"""Model class template
For segmentation
"""
import torch
from .base_model import BaseModel
from . import networks
from .bonnetal.config import BackboneConfig
from .bonnetal.head import DecoderConfig, HeadConfig


class RailSegModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.
        """
        parser.set_defaults(dataset_mode='railseg', input_nc=3, output_nc=1, segmodule='mobilenet', load_size=512, lr=5e-3, n_epochs=0, n_epochs_decay=50, batch_size=10)

        return parser

    def __init__(self, opt):
        """Initialize this model class.
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out.
        self.loss_names = ['G']
        # specify the images you want to save and display.
        self.visual_names = ['data_A', 'data_B', 'output']
        # specify the models you want to save to the disk.
        self.model_names = ['G']
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
            self.netG = networks.mobilenet_segmentator(bbone_cfg, decoder_cfg, head_cfg, gpu_ids=self.gpu_ids)

        if self.isTrain:  # only defined during training time
            # define your loss functions.
            self.criterionLoss = torch.nn.BCEWithLogitsLoss()
            # define and initialize optimizers.
            self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.data_A = input['A'].to(self.device)  # get image data A
        self.data_B = input['B'].to(self.device)  # get image data B
        self.image_paths = input['A_paths'] 

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.output = self.netG(self.data_A)  # generate output image given the input data_A

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.loss_G = self.criterionLoss(self.output, self.data_B)
        self.loss_G.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results
        self.optimizer.zero_grad()   # clear network G's existing gradients
        self.backward()              # calculate gradients for network G
        self.optimizer.step()        # update gradients for network G
