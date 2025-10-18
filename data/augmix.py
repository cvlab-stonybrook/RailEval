from data.base_dataset import BaseDataset, get_transform, get_params
from PIL import Image, Image, ImageOps, ImageEnhance, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import torch
import numpy as np
import cv2
import pandas as pd
import kornia as K
from kornia.geometry.transform import get_perspective_transform, warp_perspective
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
import cv2
from scipy.ndimage.interpolation import rotate as Rotate
import random
import math
import pdb
import torchvision.transforms as transforms
from itertools import permutations 
# import data.ImageSmoothingAlgorithmBasedOnGradientAnalysis.fga.filter_based_on_gradient_analysis as fga
  

def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)

def rotate_img(img, angle, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101):
    height, width = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    img = cv2.warpAffine(img, matrix, (width, height), flags=interpolation, borderMode=border_mode)
    return img


augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]


augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]

# CIFAR-10 constants
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]


def normalize(image):
  """Normalize input image channel-wise to zero mean and unit variance."""
  image = image.transpose(2, 0, 1)  # Switch to channel-first
  mean, std = np.array(MEAN), np.array(STD)
  image = (image - mean[:, None, None]) / std[:, None, None]
  return image.transpose(1, 2, 0)

def apply_op(image, op, severity):
    #   image = np.clip(image, 0, 255)
    pil_img = Image.fromarray(image)  # Convert to PIL.Image
    pil_img = op(pil_img, severity)
    return np.asarray(pil_img)

def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.):
    """Perform AugMix augmentations and compute mixture.
    Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.
    Returns:
    mixed: Augmented and mixed image.
    """
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(image).astype(np.float32)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(augmentations)
            image_aug = apply_op(image_aug, op, severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * image_aug
#         mix += ws[i] * normalize(image_aug)

    mixed = (1 - m) * image + m * mix
    return mixed


class RandomAugMix(ImageOnlyTransform):

    def __init__(self, severity=3, width=3, depth=-1, alpha=1., always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha

    def apply(self, image, **params):
        image = augment_and_mix(
            image,
            self.severity,
            self.width,
            self.depth,
            self.alpha
        )
        return image


class GridMask(DualTransform):
    """GridMask augmentation for image classification and object detection.

    Args:
        num_grid (int): number of grid in a row or column.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int
            an angle is picked from (-rotate, rotate). Default: (-90, 90)
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/2001.04086
    |  https://github.com/akuxcw/GridMask
    """

    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
        super(GridMask, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                             int(i * grid_h) : int(i * grid_h + grid_h / 2),
                             int(j * grid_w) : int(j * grid_w + grid_w / 2)
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)
                            ] = self.fill_value
                
                if self.mode == 1:
                    this_mask = 1 - this_mask

                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)

    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        h, w = image.shape[:2]
        mask = rotate_img(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask
        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)
        return image

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
        self.init_masks(height, width)

        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return ('num_grid', 'fill_value', 'rotate', 'mode')



class CustomSmooth(ImageOnlyTransform):
    def __init__(self, kernel_size=3,runs_number=2,always_apply=False,p=0.5):
        super(CustomSmooth, self).__init__(always_apply,p)
        self.kernel_size =  kernel_size                                            # set kernel_size = 3 for filtering with 3x3 kernel
        self.runs_number = runs_number                                            # set number of runs: parameter n is 1 by default
        
    def apply(self, img, **params):              
        output = fga.smooth(img, self.kernel_size, n=self.runs_number).astype(int)
        return output



class Self_Mixup(ImageOnlyTransform):
    def __init__(self, n=2,always_apply=False,p=0.5):
        super(Self_Mixup, self).__init__(always_apply,p)
        self.n = n 
        self.min_val = 400
        self.max_val = 1120
        # perm = permutations([i for i in range(n)]) 
        
        # # Print the obtained permutations 
        # count = 0
        # self.dict_perm = {} 
        # for i in list(perm):
        #     self.dict_perm[count] = list(i) 
        #     count+=1 
        # # print(self.dict_perm)

    def apply(self, img, **params):
        part = [0]*self.n

        # len0, len1, len2 = random.randint(100, 280), random.randint(100, 280), random.randint(100, 280)
        len_n = [random.randint(self.min_val//self.n,self.max_val//self.n) for i in range(self.n - 1)]

        # fetch_parts
        start = 0
        for i in range(self.n-1):
            part[i] = img[start:start+len_n[i],:,:]
            start += len_n[i]
        part[self.n-1] = img[start:,:,:]


        # part[0] = img[:len0,:,:]
        # part[1] = img[len0:len0+len1,:,:]
        # part[2] = img[len0+len1:len0+len1+len2,:,:]
        # part[3] = img[len0+len1+len2:,:,:]
        # i = random.randint(0, len(self.dict_perm)-1)
        # order = self.dict_perm[i]
        order = np.random.permutation(self.n)

        ##Concatenate
        final_img = part[order[0]]
        for i in range(1,self.n):
            final_img = np.concatenate((final_img,part[order[i]]),axis=0)

        # final_img = np.concatenate((part[order[0]],part[order[1]],part[order[2]],part[order[3]]),axis=0)
        # print(final_img.shape, part[0].shape, part[1].shape, part[2].shape, part[3].shape)
        # print(final_img.shape)
        return final_img


class Self_Mixup_Multiple(ImageOnlyTransform):
    def __init__(self,data,load_size,always_apply=False, p=0.5):
        super(Self_Mixup_Multiple,self).__init__(always_apply, p)
        self.dataset = data
        self.dfs = dict(tuple(self.dataset.groupby('label')))
        self.load_size = load_size

    @property
    def targets_as_params(self):
        return ['image','label']

    def get_params_dependent_on_targets(self, params):
        # img = params['image']
        return {'label':params['label']}

    def fetch_new_image_from_row(self,row):
        img_path = row['img_path'].item()
        img_path = img_path.replace('bigbrain','bigtoken.cs.stonybrook.edu/add_disk0')
        label = row['label'].item()
        angle = row['angle'].item()
        width = row['width'].item()
        dist = row['distance'].item()
        angle = torch.tensor(angle)
        width = torch.tensor(width)
        dist = torch.tensor(dist)
        # pdb.set_trace()
        imgo= Image.open(img_path).convert('RGB')
        imgo=np.array(imgo)
        h,w= imgo.shape[:2]
        imgA = transforms.ToTensor()(imgo).unsqueeze(0)
         
        
        left_top_x = (dist - 0 * np.sin(angle*math.pi/180)) / np.cos(angle*math.pi/180)
        left_bot_x = (dist - (h-1) * np.sin(angle*math.pi/180)) / np.cos(angle*math.pi/180)
        right_top_x = (dist + width - 0 * np.sin(angle*math.pi/180)) / np.cos(angle*math.pi/180)
        right_bot_x = (dist + width - (h-1) * np.sin(angle*math.pi/180)) / np.cos(angle*math.pi/180)
        
        resize_w, resize_h = self.load_size,2*self.load_size
        points_src = np.float32([[
            [left_top_x, 0.], [right_top_x, 0.], [right_bot_x, h-1.], [left_bot_x, h-1.],
        ]])
        points_dst = np.float32([[
            [0., 0.], [resize_w-1., 0.], [resize_w-1., resize_h-1.], [0., resize_h-1.],
        ]])
        
        M = cv2.getPerspectiveTransform(points_src, points_dst)
        imgA_warp = cv2.warpPerspective(imgo, M, dsize=(resize_w, resize_h))
        return imgA_warp

    def apply(self, img, **params):
        label = params['label']
        new_df = self.dfs[label]
        row = new_df.sample(n=1, random_state=42)
        img_new = self.fetch_new_image_from_row(row)

        #################  Divide IMG into 4 parts ############
        part = [0]*4
        len0, len1, len2 = random.randint(100, 280), random.randint(100, 280), random.randint(100, 280)
        part[0] = img[:len0,:,:]
        part[1] = img[len0:len0+len1,:,:]
        part[2] = img[len0+len1:len0+len1+len2,:,:]
        part[3] = img[len0+len1+len2:,:,:]
        i = random.randint(0, 23)
        order = dict_perm[str(i)]
        #################  Divide NEW IMG into 4 parts ############
        part_new = [0]*4
        part_new[0] = img_new[:len0,:,:]
        part_new[1] = img_new[len0:len0+len1,:,:]
        part_new[2] = img_new[len0+len1:len0+len1+len2,:,:]
        part_new[3] = img_new[len0+len1+len2:,:,:]
        #############################################################
        final_part = [part,part_new]
        choose_parts = [random.randint(0,1) for i in range(4)]
        # print(choose_parts)
        final_img = np.concatenate((final_part[choose_parts[0]][order[0]],final_part[choose_parts[0]][order[1]],
                                    final_part[choose_parts[0]][order[2]],final_part[choose_parts[0]][order[3]]),axis=0)
        # print(final_img.shape, part[0].shape, part[1].shape, part[2].shape, part[3].shape)
        return final_img
