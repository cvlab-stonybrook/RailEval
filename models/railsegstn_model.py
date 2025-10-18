"""Model class
For Segmentation and Alignment
"""
import torch
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from .bonnetal.config import BackboneConfig
from .bonnetal.head import DecoderConfig, HeadConfig
from .bonnetal.segmentator import Segmentator
from itertools import chain


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # To avoid division by zero

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        
        return 1 - dice  # Dice loss


class RailSegStnModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.
        """
        parser.set_defaults(dataset_mode='railsegstn', input_nc=3, output_nc=1, segmodule='mobilenet', load_size=512, stnstyle='separate', lr=5e-4, n_epochs=0, n_epochs_decay=50, batch_size=10)

        return parser

    def __init__(self, opt):
        """Initialize this model class.
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.opt = opt
        # specify the training losses you want to print out.
        self.loss_names = ['seg', 'stn', 'total']
        # specify the images you want to save and display.
        self.visual_names = ['data_A', 'data_B', 'data_C', 'segout', 'stnout']
        # specify the models you want to save to the disk.
        self.model_names = ['seg', 'stn']
        # define networks
        if opt.segmodule == 'unet':
            if opt.load_size == 512:
                self.netG = networks.define_G(opt.input_nc, opt.output_nc, 64, 'unet_256', gpu_ids=self.gpu_ids)
            else:
                self.netG = networks.define_G(opt.input_nc, opt.output_nc, 64, 'unet_128', gpu_ids=self.gpu_ids)
        elif opt.segmodule == 'mobilenet':
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
            seg_state_dict = torch.load('/data/add_disk0/shilin/Rail/checkpoints/segonly_mbnet_%s_1/10_net_G.pth' % opt.load_size)
        self.netseg.load_state_dict(seg_state_dict)
        self.netseg = networks.init_net(self.netseg, gpu_ids=self.gpu_ids)
        print('Successfully loaded seg weights')
        
        self.netstn = networks.STNet(input_size=(1200,1600), style=opt.stnstyle)
        self.netstn = networks.init_net(self.netstn, gpu_ids=self.gpu_ids)
        
        if self.isTrain:  # only defined during training time
            # define your loss functions.
            self.bceLoss = torch.nn.BCELoss()
            self.diceLoss = DiceLoss()
            self.stnLoss = torch.nn.L1Loss()
            self.optimizer = torch.optim.Adam(chain(self.netseg.parameters(), self.netstn.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.data_A_seg = input['A_seg'].to(self.device)
        self.data_A = input['A'].to(self.device)
        self.data_B = input['B'].to(self.device)
        self.data_C = input['C'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.segout = self.netseg(self.data_A_seg)
        self.segout = F.interpolate(self.segout, size=self.data_A.shape[2:], mode='bilinear', align_corners=True)
        if self.opt.stnstyle == 'separate':
            binary_segout = torch.ge(self.segout, 0.5).float()
            self.segout = binary_segout + (self.segout - binary_segout).detach()
            self.theta = self.netstn(binary_segout*self.data_A)
            grid = F.affine_grid(self.theta, self.data_A.size())
            self.stnout = F.grid_sample(self.data_A*binary_segout, grid)
            if not self.isTrain:
                self.Aout = F.grid_sample(self.data_A, grid)
        if self.opt.stnstyle == 'together':
            self.segout = 0.5 * (torch.tanh(50 * (self.segout - 0.5)) + 1)
            self.theta = self.netstn(self.segout*self.data_A)
            grid = F.affine_grid(self.theta, self.data_A.size())
            self.stnout = F.grid_sample(self.data_A*self.segout, grid)
            if not self.isTrain:
                self.Aout = F.grid_sample(self.data_A, grid)

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_seg = 0.4*(0.2*self.bceLoss(self.segout, self.data_B)+self.diceLoss(self.segout, self.data_B))
        left_region = 5*self.stnLoss(self.stnout[:, :, :, 300:532], self.data_C[:, :, :, 300:532])
        right_region = 5*self.stnLoss(self.stnout[:, :, :, 1067:1300], self.data_C[:, :, :, 1067:1300])
        middle_region = self.stnLoss(self.stnout[:, :, :, 532:1067], self.data_C[:, :, :, 532:1067])
        self.loss_stn = 0.6*10*(left_region + right_region + middle_region)
        self.loss_total = (self.loss_seg) + (self.loss_stn)

        self.loss_total.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

