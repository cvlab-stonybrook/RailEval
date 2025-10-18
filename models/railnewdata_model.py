"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from .bonnetal.config import BackboneConfig
from .bonnetal.head import DecoderConfig, HeadConfig
from .bonnetal.segmentator import Segmentator


class RailNewDataModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.
        """
        parser.set_defaults(dataset_mode='railunion', input_nc=3, output_nc=1, load_size=512, classifier='resnet18', segmodule='mobilenet', lr=1e-4, n_epochs=0, n_epochs_decay=100, batch_size=10)

        return parser

    def __init__(self, opt):
        """Initialize this model class.
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.opt = opt
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['cls', 'total']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        if self.isTrain:
            self.visual_names = ['data_A', 'segout', 'stnout']
        else:
            self.visual_names = ['data_A', 'segout', 'clsin']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        self.model_names = ['seg', 'stn', 'cls']
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        if self.opt.segmodule == 'mobilenet':
            bbone_cfg = BackboneConfig(name='mobilenetv2', 
                                        os=8,
                                        h=opt.load_size,
                                        w=opt.load_size,
                                        d=opt.input_nc,
                                        dropout=0.0,
                                        bn_d=0.05,
                                        extra={'width_mult': 1.0, 'shallow_feats': True})
            decoder_cfg = DecoderConfig(name='aspp_progressive',
                                        dropout=0.0,
                                        bn_d=0.05,
                                        extra={'aspp_channels': 32, 'last_channels': 16})
            head_cfg = HeadConfig(n_class=1,
                                dropout=0.0,
                                weights=torch.ones(1, dtype=torch.float))
            self.netseg = Segmentator(bbone_cfg, decoder_cfg, head_cfg)
            if self.isTrain:
                seg_state_dict = torch.load('/data/add_disk0/shilin/Rail/checkpoints/ttc_segstn_sep_1/latest_net_seg.pth')
                self.netseg.load_state_dict(seg_state_dict)
        else:
            norm_layer = networks.get_norm_layer(norm_type='batch')
            self.netseg = networks.UnetGenerator(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer)
            if self.isTrain:
                seg_state_dict = torch.load('/data/add_disk0/shilin/Rail/checkpoints/align_separate_1/latest_net_seg.pth')
                self.netseg.load_state_dict(seg_state_dict)
        self.netseg = networks.init_net(self.netseg, gpu_ids=self.gpu_ids)
        print('Successfully loaded seg weights')

        for param in self.netseg.parameters():
            param.requires_grad = False
        
        self.netstn = networks.STNet(input_size=(1200,1600))
        if self.isTrain:
            stn_state_dict = torch.load('/data/add_disk0/shilin/Rail/checkpoints/ttc_segstn_sep_1/latest_net_stn.pth')
            self.netstn.load_state_dict(stn_state_dict)
        self.netstn = networks.init_net(self.netstn, gpu_ids=self.gpu_ids)
        print('Successfully loaded STN weights')

        for param in self.netstn.parameters():
            param.requires_grad = False
        
        self.netcls = networks.define_classifier(classifier=opt.classifier, num_classes=opt.num_classes, gpu_ids=self.gpu_ids)
        
        if self.isTrain:  # only defined during training time
            # define your loss functions.
            self.criterionLoss = torch.nn.CrossEntropyLoss()
            # define and initialize optimizers.
            self.optimizer_cls = torch.optim.Adam(self.netcls.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_cls)

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.data_A_seg = input['A_seg'].to(self.device)
        self.data_A = input['A'].to(self.device)
        self.target = input['label'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.segout = self.netseg(self.data_A_seg)
        self.segout = F.interpolate(self.segout, size=self.data_A.shape[2:], mode='bilinear', align_corners=True)
        binary_segout = torch.ge(self.segout, 0.5).float()
        self.theta = self.netstn(binary_segout*self.data_A)
        self.grid = F.affine_grid(self.theta, self.data_A.size())
        if self.isTrain:
            self.stnout = F.grid_sample(self.data_A*binary_segout, self.grid)
        self.clsin = F.grid_sample(self.data_A, self.grid)
        self.clsin = torch.nn.functional.interpolate(self.clsin[:,:,:,532:1067], size=(448,448), mode='bilinear', align_corners=True)        
        self.clsout = self.netcls(self.clsin)

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_cls = 1*self.criterionLoss(self.clsout, self.target)
        self.loss_total = self.loss_cls

        self.loss_total.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results
        self.optimizer_cls.zero_grad()
        self.backward()
        self.optimizer_cls.step()

    def get_current_output(self):
        return self.clsout, self.target
