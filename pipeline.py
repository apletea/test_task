import numpy as np

import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN

from model import EfficientNetBase

class ResultProvider(torch.nn.Module):

    def __init__(self, device, glass_model=None , detector=None):
        super(ResultProvider, self).__init__()
        if glass_model is None:
            glass_model = EfficientNetBase()
        if detector is None:
            detector = MTCNN(image_size=224)

        self.glass_model = glass_model
        self.glass_model.to(device)
        self.detector = detector
        self.detector.to('cpu')

        self.gsize = (96,96)
        self.device = device
    
    def load_weights(self, path: str) -> None:
        self.glass_model.load_state_dict(torch.load(path))
    
    def _totensor(self, img: np.array) -> torch.Tensor:
        nimg = img / 255
        timg = np.transpose(nimg, (2,0,1))

        tensor = torch.from_numpy(timg)
        tensor.to(self.device)
        return tensor

    def _tonumpy(self, tensor: torch.Tensor) -> np.array:
        numpy_arr = tensor.cpu().numpy()
        numpy_arr = np.transpose(( numpy_arr * 255).astype(np.uint8),(1,2,0))
        return numpy_arr

    def _convertTensorToPil(self, tensor: torch.Tensor) -> Image:
        numpy_arr = self._tonumpy(tensor) 
        img = Image.fromarray(cv2.resize(numpy_arr, (224,224)))
        return img

    def _cropAndResize(self, tensor: torch.Tensor ,bbox: np.array) -> torch.Tensor:
        c, h, w = tensor.shape
        numpy_arr = self._tonumpy(tensor)
        
        abs_x0 = bbox[0] / 224
        abs_x1 = bbox[2] / 224
        abs_y0 = bbox[1] / 224
        abs_y1 = bbox[3] / 224

        y0 = int(abs_x0 * w)
        y1 = int(abs_x1 * w)
        x0 = int(abs_y0 * h)
        x1 = int(abs_y1 * h)

        crop = numpy_arr[x0:x1, y0:y1]
        crop = cv2.resize(crop, self.gsize) 
        
        crop_tensor = self._totensor(crop)
        return crop_tensor


    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        b,c,h,w = tensor.shape
        crops = []
        for i in range(b):
            self.detector.to('cpu')
            bboxes, _ = self.detector.detect(self._convertTensorToPil(tensor[i])) 
            if bboxes is None:
#               crops.append(self._crostensor[i])
                bboxes = [[0,0,224,224]]
            bbox = bboxes[0]
            crops.append(self._cropAndResize(tensor[i], bbox))
#       print(crops.shape)
        cropTensor = torch.stack(crops) 
        self.glass_model.to(self.device)
        res = self.glass_model(cropTensor.to(self.device).float())
        return res
        
#       batch_size