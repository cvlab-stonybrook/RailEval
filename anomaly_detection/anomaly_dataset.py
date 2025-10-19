import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import csv





class AnomalyDataset(Dataset):
    def __init__(self, csv_file, mask=True, normal_mask_dir=None, anormal_mask_dir=None, transform=None, mask_transform=None):
        
        self.images = []
        self.transform=transform
        self.mask_transform=mask_transform
        self.mask = mask
        self.normal_mask_dir = normal_mask_dir
        self.anormal_mask_dir = anormal_mask_dir
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                img_path = row[0]
                label = int(row[1])
                if self.mask:
                    mask_name = os.path.basename(img_path)[:-4] + '_output.png'
                    self.images.append([img_path, mask_name, label])
                else:
                    self.images.append([img_path, label])
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        path_img = self.images[idx][0]
        img = Image.open(path_img).convert("RGB")

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        if self.mask:
            mask_name = self.images[idx][1]
            label = self.images[idx][2]
            if label == 0:
                mask_path = os.path.join(self.normal_mask_dir, mask_name)
            elif label == 1:
                mask_path = os.path.join(self.anormal_mask_dir, mask_name)
            mask = Image.open(mask_path).convert("RGB")
            if self.mask_transform:
                mask = self.mask_transform(mask)
            else:
                mask = transforms.ToTensor()(mask)

            return torch.cat((img, mask), dim=0), self.images[idx][2], path_img
        else:
            return img, self.images[idx][1], path_img