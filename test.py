"""
Test script for segmentation and alignmnet module.
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import time
from torchvision import transforms
from PIL import Image
import torch
import numpy as np


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0             # test code only supports num_threads = 0
    opt.batch_size = 1              # test code only supports batch_size = 1
    opt.serial_batches = True       # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True              # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1             # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)   # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)       # create a model given opt.model and other options
    model.setup(opt)                # regular setup: load and print networks; create schedulers

    save_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
    if opt.load_iter > 0:  # load_iter is 0 by default
        save_dir = '{:s}_iter{:d}'.format(save_dir, opt.load_iter)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # test with eval mode. This only affects layers like batchnorm and dropout.
    model.eval()
    for i, data in enumerate(dataset):
        h, w = data['ori_h'].item(), data['ori_w'].item()
        model.set_input(data)       # unpack data from data loader
        model.test()                # run inference
        
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 10 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        short_path = img_path[0].split('/')[-1]
        dir_path = img_path[0].split('/')[-2]
        check_mkdir(os.path.join(save_dir, dir_path))
        name = short_path[:-4]

        for label, im_data in visuals.items():
            if label == 'output':
                im = transforms.Resize((h, w))(transforms.ToPILImage()((im_data.data.squeeze(0).cpu())))
                image_name = '%s_output.png' % (name)
                save_path = os.path.join(save_dir, dir_path, image_name)
                im.save(save_path)
            if label == 'segout':
                im = transforms.Resize((h, w))(transforms.ToPILImage()((im_data.data.squeeze(0).cpu())))
                image_name = '%s_segout.png' % (name)
                save_path = os.path.join(save_dir, dir_path, image_name)
                im.save(save_path)
            if label == 'stnout':
                im = transforms.Resize((h, w))(transforms.ToPILImage()((im_data.data.squeeze(0).cpu())))
                image_name = '%s_stnout.png' % (name)
                save_path = os.path.join(save_dir, dir_path, image_name)
                im.save(save_path)
            if label == 'data_A' or label == 'data_B' or label == 'data_C':
                im = transforms.Resize((h, w))(transforms.ToPILImage()((im_data.data.squeeze(0).cpu())))
                image_name = '%s.png' % (name+'_'+label)
                save_path = os.path.join(save_dir, dir_path, image_name)
                im.save(save_path)
            if label == 'clsin':
                im = transforms.Resize((448, 448))(transforms.ToPILImage()((im_data.data.squeeze(0).cpu())))
                image_name = '%s_clsin.png' % (name)
                save_path = os.path.join(save_dir, dir_path, image_name)
                im.save(save_path)
