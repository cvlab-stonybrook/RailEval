"""Dataset class
Load TTC data for training and testing
"""
from data.base_dataset import BaseDataset, get_transform, get_params
from PIL import Image, ImageDraw
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import torch
import numpy as np
import pandas as pd
from .augmix import RandomAugMix, GridMask
import albumentations as Album
import torchvision.transforms as transforms
import math
import cv2
import scipy.io


def make_dataset(root):
    assert os.path.isdir(root)
    imgs = []
    for folder in os.listdir(root):
        if 'TTC_' in folder:
            label = int(folder.split('_')[-1])
            for file in os.listdir(os.path.join(root, folder)):
                imgs.append([os.path.join(root, folder, file), label])
    return imgs


class RailNewDataDataset(BaseDataset):
    """A dataset class for rail images."""
    def __init__(self, opt):
        """Initialize this dataset class.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        df_ttc = pd.read_pickle(opt.dataroot)
        self.data_info = df_ttc
        self.data_info = self.data_info.sort_values(by='label').reset_index(drop=True)
            
        if self.opt.aug:
            blur_limit=(3, 7)
            sigma_limit=0.0
            self.trans_aug = Album.Compose([
                Album.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2,p=0.5),
                Album.JpegCompression(quality_lower=70),
                Album.GaussianBlur(blur_limit, sigma_limit),
                Album.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
            ])
        self.trans_res = Album.Resize(opt.load_size,opt.load_size)
        
        self.opt = opt

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        """
        img_path = self.data_info.iloc[index][0]
        label = self.data_info.iloc[index][1]
        
        A = Image.open(img_path).convert('RGB')
        w, h = A.size
        A = np.array(A)

        if self.opt.cls_type == 'ce':
            new_label = torch.tensor(label)

        if self.opt.isTrain and self.opt.aug:
            transformed = self.trans_aug(image=A)
            A = transformed['image']

        A_seg = self.trans_res(image=A.copy())['image']

        A = transforms.ToTensor()(A)
        A_seg = transforms.ToTensor()(A_seg)

        return {'A': A, 'A_seg': A_seg, 'label': new_label, 'A_paths': img_path, 'ori_w': w, 'ori_h': h}

    def __len__(self):
        """Return the total number of images."""
        return len(self.data_info)