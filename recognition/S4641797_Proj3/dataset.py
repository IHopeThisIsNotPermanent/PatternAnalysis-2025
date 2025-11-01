import requests

import numpy as np
import nibabel as nib
import os

from torch.utils.data import DataLoader, Dataset

class UnlabeledImageDataset(Dataset):
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
        return {label:image}
