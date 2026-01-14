import os
import torch
import random
import copy
from PIL import Image, ImageOps

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import cv2

class ISICDataset2020(Dataset):

    def __init__(self, df, root, transformation=None):

        self.df = df
        self.root = root
        
        self.transform = transformation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_name = row.image_name + '.jpg'
        img_path = os.path.join(self.root, sample_name)

        image = Image.open(img_path).convert("RGB")

        width, height = image.size
        max_dim = max(width, height)

        padded_image = Image.new(image.mode, (max_dim, max_dim), (0, 0, 0))
        
        offset_w = (max_dim - width) // 2
        offset_h = (max_dim - height) // 2

        padded_image.paste(image, (offset_w, offset_h))

        label = torch.tensor(row.target, dtype=torch.long)

        if self.transform:
            padded_image = self.transform(padded_image)   
        
        return padded_image, label

class ISICDataset2018(Dataset):
    def __init__(self, df, root, transformation=None):
        self.df = df
        self.root = root
            
        self.transform = transformation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_name = row.image + '.jpg'

        img_path = os.path.join(self.root, sample_name)

        image = Image.open(img_path).convert('RGB')

        label = row[["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]].astype(float).values
        label = torch.tensor(label.argmax(), dtype=torch.long)


        if self.transform:
            image = self.transform(image)

        return image, label






















        
        

