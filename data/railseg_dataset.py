"""Dataset class
Segmentation standalone
"""
from data.base_dataset import BaseDataset
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import scipy.io
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import albumentations as Album


class RailSegDataset(BaseDataset):
    """A dataset class for rail images."""
    def __init__(self, opt):
        """Initialize this dataset class.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        
        self.data_info = pd.read_pickle(opt.dataroot)

        self.img_trans = Album.Compose([
            Album.Resize(opt.load_size,opt.load_size),
            Album.Flip(),
            Album.RandomBrightnessContrast(),
            Album.Affine(translate_px={'x':(-50,50),'y':0}, mode=4)
            ])
        self.test_trans = Album.Compose([
            Album.Resize(opt.load_size,opt.load_size),
            ])
        
        self.opt = opt

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index -- a random integer for data indexing
        Returns:
            A(tensor): rail image
            B(tensor): rail segmentation gt
            path(str): path to the image
        """
        img_path = self.data_info.iloc[index]['img_path']
        img_path = img_path.replace('/nfs/bigbrain/yashfulwani/rail/', self.opt.dataroot)
        gt_path = img_path.split('Paper_RawData')[0] + 'Paper_GT/mask' + img_path.split('Paper_RawData')[1][:-5] + '.mat'
        label = self.data_info.iloc[index]['label']
        label = torch.tensor(label)
        
        A = Image.open(img_path).convert('RGB')
        w, h = A.size
        A = np.array(A)
        B = scipy.io.loadmat(gt_path)
        B = B['mask'] * 255
        
        if self.opt.isTrain:
            transformed = self.img_trans(image=A, mask=B)
            A = transformed['image']
            B = transformed['mask']
        else:
            transformed = self.test_trans(image=A, mask=B)
            A = transformed['image']
            B = transformed['mask']
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        
        return {'A': A, 'B': B, 'label': label, 'A_paths': img_path, 'ori_w': w, 'ori_h': h}

    def __len__(self):
        """Return the total number of images."""
        return len(self.data_info)
