import requests

import numpy as np
import nibabel as nib
import os

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch 
import torch.nn.functional as F

class LabeledImageDataset(Dataset):
    def __init__(self, image_dir, label_image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        self.label_paths = [os.path.join(label_image_dir, img) for img in os.listdir(label_image_dir)]
        self.transform = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.target_size = (256, 128) 

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = np.array(nib.load(img_path).dataobj)
        image = np.expand_dims(image, axis=0)
       
        image = torch.from_numpy(image).float()

        label_path = self.label_paths[idx]
        limage = np.array(nib.load(label_path).dataobj)
        limage = np.expand_dims(limage, axis=0)
       
        limage = torch.from_numpy(limage).float()

        if image.shape[2:] != self.target_size:
             image = F.interpolate(image.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
        if limage.shape[2:] != self.target_size:
             limage = F.interpolate(limage.unsqueeze(0), size=self.target_size, mode='nearest').squeeze(0) 


        return image, limage

def load_dataset():
    transform = transforms.Compose([transforms.ToTensor(),])

    batch_size = 8
    train_loader = DataLoader(dataset=LabeledImageDataset('./data/keras_slices_data/keras_slices_train',
                                                          './data/keras_slices_data/keras_slices_seg_train',
                                                        transform=transform),
                          batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=LabeledImageDataset('./data/keras_slices_data/keras_slices_test',
                                                          './data/keras_slices_data/keras_slices_seg_test',
                                                        transform=transform),
                             batch_size=batch_size, shuffle=False)

    return train_loader, test_loader