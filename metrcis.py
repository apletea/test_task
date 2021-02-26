import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import cv2
import torch
from torch.utils.data import DataLoader
from PIL import Image
from facenet_pytorch import MTCNN
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score


from pipeline import ResultProvider
from dataset import FacesValidation 


batch_size = 64
workers = 8

weights = 'epoch_6_glass.pth'
conf_th = 0.5

root_folder = 'example_data_glasses'

if __name__ == "__main__":

    dataset = FacesValidation(root_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)

    model = ResultProvider('cuda')
    model.load_weights(weights)

    for x, y in dataloader:
        y_hat = model(x)

        y = y.cpu().numpy()
        y_hat = torch.squeeze(y_hat, dim=1)
        y_hat = y_hat.detach().cpu().numpy()
        fpr, tpr, _ = roc_curve(y, y_hat)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
#       plt.show()
        plt.savefig('roc_curve.png')


        precision, recall, _ = precision_recall_curve(y, y_hat)
        average_precision = average_precision_score(y, y_hat)

        plt.figure()
        plt.step(recall, precision, where='post')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(
            'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision))
#       plt.show()
        plt.savefig('precision_recall_curve.png')

        accuracies = []
        thresholds = np.arange(0,1,0.01)
        for threshold in thresholds:
            y_pred = np.greater(y_hat, threshold).astype(int)
            accuracy = accuracy_score(y, y_pred)
            accuracies.append(accuracy)

        accuracies = np.array(accuracies)
        max_accuracy = accuracies.max() 
        max_accuracy_threshold =  thresholds[accuracies.argmax()]
        print(max_accuracy, max_accuracy_threshold)

        plt.figure()
        plt.step(thresholds, accuracies, where='post')
        plt.xlabel('Confidence threshold')
        plt.ylabel('Accuracy')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
#       plt.show()
        plt.savefig('accuracy.png')
        




        
        
