"""Dataset class
Segmentation + Alignment module
"""
from data.base_dataset import BaseDataset
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import cv2
import albumentations as Album
import scipy.io


class RailSegStnDataset(BaseDataset):
    """A dataset class for rail images."""
    def __init__(self, opt):
        """Initialize this dataset class.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        self.data_info = pd.read_pickle(opt.dataroot)
        self.data_info = self.data_info.reset_index().drop('index', axis=1)

        self.trans_ = Album.Compose([
            Album.RandomBrightnessContrast(p=0.5),
            Album.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
            Album.JpegCompression(quality_lower=70, quality_upper=100, p=0.3),
            Album.Flip(p=0.5),
            Album.ShiftScaleRotate(shift_limit_x=0.10, shift_limit_y=0, scale_limit=0, rotate_limit=5, border_mode=cv2.BORDER_REFLECT, p=0.7),
        ])
        
        self.opt = opt

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        """
        img_path = self.data_info.iloc[index]['img_path']
        img_path = img_path.replace('/nfs/bigbrain/yashfulwani/rail/', self.opt.dataroot)
        gt_path = img_path.split('Paper_RawData')[0] + 'Paper_GT/mask' + img_path.split('Paper_RawData')[1][:-5] + '.mat'
        
        A = Image.open(img_path).convert('RGB')
        w, h = A.size
        A = np.array(A)
        B = scipy.io.loadmat(gt_path)
        B = B['mask']

        if self.opt.isTrain:
            transformed = self.trans_(image=A, mask=B)
            A = transformed['image']
            B = transformed['mask']

            scale_factor = np.random.uniform(0.8, 1.0)
            crop_h = int(h * scale_factor)
            crop_w = int(w * scale_factor)
            y_start = np.random.randint(0, h - crop_h + 1)
            x_start = np.random.randint(0, w - crop_w + 1)
            
            A = A[y_start:y_start + crop_h, x_start:x_start + crop_w]
            A = cv2.resize(A, (w, h))
            B = B[y_start:y_start + crop_h, x_start:x_start + crop_w]
            B = cv2.resize(B, (w, h))

        bnzt = np.nonzero(B)
        left_top_x = bnzt[1][bnzt[0]==0].min()
        right_top_x = bnzt[1][bnzt[0]==0].max()
        left_bot_x = bnzt[1][bnzt[0]==(h-1)].min()
        right_bot_x = bnzt[1][bnzt[0]==(h-1)].max()
        src = np.float32([[left_top_x, 0.], [right_top_x, 0.], [right_bot_x, 1199.], [left_bot_x, 1199.]])
        dst = np.float32([[532, 0.], [1066, 0.], [1066, h-1.], [532, h-1.]])
        M = cv2.getAffineTransform(src[:3], dst[:3])
        mask = np.repeat(B[:,:,np.newaxis], 3, -1)
        C = cv2.warpAffine(A*mask, M, dsize=(w, h))
        
        A_seg = Album.Resize(self.opt.load_size,self.opt.load_size)(image=A.copy())['image']
        A = transforms.ToTensor()(A)
        A_seg = transforms.ToTensor()(A_seg)
        B = transforms.ToTensor()(B*255)
        C = transforms.ToTensor()(C)
        
        return {'A': A, 'A_seg': A_seg, 'B': B, 'C': C, 'A_paths': img_path, 'ori_w': w, 'ori_h': h}

    def __len__(self):
        """Return the total number of images."""
        return len(self.data_info)
