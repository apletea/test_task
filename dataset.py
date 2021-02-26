import os
import numpy as np

import cv2
import pandas as pd
import torch
from torch.utils import data

def preprocces(img, size):
    img = cv2.resize(img, size)
    img = img / 255
    timg = np.transpose(img, (2,0,1))
    tensor = torch.from_numpy(timg)

    return tensor

class FacesDataset(data.Dataset):
    
    def __init__(self, root, size=(224,224)):
        self.root = root
        self.size = size
        self.img_dir = os.path.join(root, 'faces', 'faces')
        self.csv_path = os.path.join(root, 'train.csv')

        self.labels = pd.read_csv(self.csv_path)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        
        img_path = f'{self.img_dir}/face-{index+1}.png'
        img_raw = cv2.imread(img_path)

        tensor = preprocces(img_raw, self.size)
        label = self.labels['glasses'][index]
        return tensor, label
    
class FacesValidation(data.Dataset):

    def _load(self, path, label):
        for item in os.listdir(path):
            img_path = os.path.join(path, item)
            img_raw = cv2.imread(img_path)
            self.x.append(img_raw)
            self.y.append(label)

    def __init__(self, root, size=(224,224)):
        self.root = root
        self.size = size
        self.glasses_path = os.path.join(root, 'with_glasses')
        self.glasesles_path = os.path.join(root, 'without_glasses')

        self.x = []
        self.y = []

        self._load(self.glasses_path, 1)
        self._load(self.glasesles_path, 0)
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, index):
        tensor = preprocces(self.x[index], self.size)
        label = self.y[index]
        return tensor, label
    
class FolderData(data.Dataset):

    def __init__(self, root, size=(224,224)):
        self.root = root
        self.size = size
        self.imgs_path = os.listdir(self.root)

    def __len__(self):
        return len(self.imgs_path)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.imgs_path[index])
        img_raw = cv2.imread(img_path)
        tensor = preprocces(img_raw, self.size)
        return tensor, img_path




