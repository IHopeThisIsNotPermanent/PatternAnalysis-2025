import requests
import torch

import numpy as np
import nibabel as nib
import os

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


class LabeledImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = img_path.split("\\")[-1].split('_week_')[-1].split('_')[0]
        image = np.array(nib.load(img_path).dataobj)
        image = np.expand_dims(image, axis=0)
        image = np.moveaxis(image, 0, 0)
        image = torch.from_numpy(image).float()

        return image, label

def load_dataset():
    transform = transforms.Compose([transforms.ToTensor(),])

    batch_size = 50
    train_loader = DataLoader(dataset=LabeledImageDataset('./data/keras_slices_data/keras_slices_test',
                                                        transform=transform),
                          batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=LabeledImageDataset('./data/keras_slices_data/keras_slices_train',
                                                        transform=transform),
                             batch_size=batch_size, shuffle=False)

    return train_loader, test_loader