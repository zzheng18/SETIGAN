import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt


class CustomDataset2(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = pd.read_csv(img_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        data_name = self.img_dir.iloc[idx, 0]
        f = h5py.File('just_background.h5', 'r')
        arr = np.array(f.get(data_name))
        
        image = np.flip(arr.reshape(128,128),1)

        img = (image - image.mean()) / np.std(image)
        image2 = (img-img.min())/(img.max()-img.min())
        image3 = (image2-0.5)*2
        image = torch.from_numpy(image3)
        
        label = self.img_labels.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label