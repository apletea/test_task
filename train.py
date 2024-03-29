import os
import sys
import numpy as np

import cv2
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from catalyst import dl, metrics
from catalyst.contrib.data.cv import ToTensor
from catalyst.contrib.datasets import MNIST
from PIL import Image
from facenet_pytorch import MTCNN
from tensorboardX import SummaryWriter

from pipeline import ResultProvider
from dataset import FacesDataset, FacesValidation

conf_th = 0.8
device = 'cuda'

model = ResultProvider(device)
optimizer = torch.optim.Adam(model.glass_model.parameters(), lr=0.02)

loaders = {
    "train": DataLoader(FacesDataset('kaggle'), batch_size=64),
    "valid": DataLoader(FacesValidation('example_data_glasses'), batch_size=64),
}

class CustomRunner(dl.Runner):

    def _handle_batch(self, batch):
        x, y = batch
        y = torch.unsqueeze(y, dim=1)
        y = y.float().to(device)
        y_hat = self.model(x)

        loss = F.binary_cross_entropy(y_hat, y)
        accuracy01 = metrics.multilabel_accuracy(y_hat, y, threshold=0.5)
        print(y_hat, y)
)
        self.batch_metrics.update(
            {"loss": loss,
             "accuracy01": accuracy01,
            }
        )

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

runner = CustomRunner()
# model training
runner.train(
    model=model,
    optimizer=optimizer,
    loaders=loaders,
    logdir="./logs",
    num_epochs=100,
    verbose=True,
    load_best_on_end=True,
)

torch.save(model.glass_model.state_dict, 'weights_glass.pth')