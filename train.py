import os
import sys
import numpy as np

import cv2
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from PIL import Image
from facenet_pytorch import MTCNN

from pipeline import ResultProvider
from dataset import FacesDataset, FacesValidation


batch_size = 64
workers = 8

conf_th = 0.5
device = 'cuda'

import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from catalyst import dl, metrics
from catalyst.contrib.data.cv import ToTensor
from catalyst.contrib.datasets import MNIST

model = ResultProvider('cuda')
optimizer = torch.optim.Adam(model.glass_model.parameters(), lr=0.02)

loaders = {
    "train": DataLoader(FacesDataset('kaggle'), batch_size=64),
    "valid": DataLoader(FacesValidation('example_data_glasses'), batch_size=64),
}

class CustomRunner(dl.Runner):

    def _handle_batch(self, batch):
        # model train/valid step
        x, y = batch
        y = torch.unsqueeze(y, dim=1)
        y = y.float().to('cuda')
        y_hat = self.model(x)


#       print(y_hat.dtype)
#       print(y.dtype)
        loss = F.binary_cross_entropy(y_hat, y)
        accuracy01 = metrics.multilabel_accuracy(y_hat, y, threshold=0.5)
        print(y_hat, y)
#       print(accuracy01)
#       auc = metrics.auc(y_hat, y)
#       avg_precision = metrics.average_precision(y_hat, y)
#       print(loss)
#       print(accuracy01)
        self.batch_metrics.update(
            {"loss": loss,
             "accuracy01": accuracy01,
#            'auc':auc,
#           'avg_precision': avg_precision}
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
# model inference
#for prediction in runner.predict_loader(loader=loaders["valid"]):
#   assert prediction.detach().cpu().numpy().shape[-1] == 10
# model tracing
#traced_model = runner.trace(loader=loaders["valid"])