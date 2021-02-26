import os
import sys
import numpy as np

import cv2
import torch
from torch.utils.data import DataLoader
from PIL import Image
from facenet_pytorch import MTCNN

from pipeline import ResultProvider
from dataset import FolderData


batch_size = 64
workers = 8

weights = 'epoch_6_glass.pth'
conf_th = 0.85

if __name__ == "__main__":
    assert(len(sys.argv) >= 2)

    in_folder = sys.argv[1]
    dataset = FolderData(in_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)

    model = ResultProvider('cuda')
    model.load_weights(weights)

    for frames, path in dataloader:
        b,c,h,w = frames.shape
        res = model(frames)
        for i in range(b):
            if (res[i] > conf_th):
                print(path[i])