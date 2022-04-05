import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random


from models.experimental import attempt_load
from utils.datasets import  letterbox
from utils.general import check_img_size,  non_max_suppression, scale_coords
from utils.torch_utils import select_device
import pdb
# class number 33
# 0-32
class Net():
    def __init__(self, weights='./best.pt',
                 view_img=False, imgsz=640, device='0',
                 conf_thres=0.5, iou_thres=0.3):

        self.weights = weights
        self.imgsz = imgsz
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agnostic_nms = False
        self.augment = False
        self.classes = [i for i in range(34)]
        # Initialize
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(
            self.weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(
            self.imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16
        # Get names and colors
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)]
                       for _ in range(len(self.names))]
        self.img = torch.zeros(
            (1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
        _ = self.model(self.img.half(
        ) if self.half else self.img) if self.device.type != 'cpu' else None  # run once

    def detect(self, im0s):


        img = letterbox(im0s, new_shape=self.imgsz)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres)
        pred = pred[0]# bs = 1
        dets = [[] for _ in self.classes]
        if len(pred):
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0s.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(pred):
                xyxy = torch.tensor(xyxy).view(-1).tolist()
                dets[int(cls.item())].append(xyxy)
        return dets

if __name__ == '__main__':
    det = Net()
    im = cv2.imread('/home/fht/data/ocrtoc/scenes/20210512154044/rgb_undistort/0000.png')
    res= det.detect(im)
