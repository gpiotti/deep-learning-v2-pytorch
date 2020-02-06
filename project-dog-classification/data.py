#!/usr/bin/env python
# coding: utf-8
import torch
from torch.utils.data import Dataset
import glob
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_class_from_path(path):
    return torch.tensor(int(path.split(".")[0].split("\\")[-1])-1)

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(glob.glob(self.root_dir + '*/*/*.*', recursive=True))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_paths = list(glob.glob(self.root_dir + '*/*/*.*', recursive=True))
        img = Image.open(img_paths[idx])
        target = get_class_from_path(img_paths[idx])
        img_transf = self.transform(img)
        
        img_transf = img_transf

        return img_transf, target