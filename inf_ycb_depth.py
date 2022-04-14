import os
import copy
import pdb
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math
import open3d as o3d
import torch
import cv2
from scipy.spatial.transform import Rotation as Ro
from yolo_det import Net as detNet

CLASSES = ['clear_box', 'plastic_apple', 'plastic_banana', 'plastic_orange', 'plastic_peach', 'plastic_plum',
      'plastic_strawberry', 'orion_pie', 'potato_chip', 'suger', 'flat_screwdriver', 'large_clamp', 'mini_claw_hammer',
      'power_drill', 'scissors', 'two_color_hammer', 'square_plate', 'green_bowl', 'blue_plate', 'bleach_cleanser',
      'blue_cup', 'blue_moon', 'blue_tea_box', 'book', 'book_holder', 'bowl', 'cleanser', 'conditioner',
      'correction_fuid', 'cracker_box', 'doraemon_bowl', 'doraemon_plate', 'extra_large_clamp', 'fork', 'gelatin_box',
      'glue', 'grey_plate', 'knife', 'large_marker', 'lipton_tea', 'magic_clean', 'medium_clamp', 'mug', 'pen_container',
      'phillips_screwdriver', 'pitcher', 'plate', 'plate_holder', 'poker', 'pudding_box', 'red_marker', 'remote_controller',
      'round_plate', 'small_clamp', 'small_marker', 'soap', 'soap_dish', 'spoon', 'stapler', 'suger_box', 'toothpaste',
      'yellow_cup', 'tbd', 'green_cup', 'orange_cup', 'pink_tea_box', 'plastic_pear', 'plastic_lemon', 'black_marker',
      'blue_marker', 'repellent', 'shampoo', 'yellow__bowl', 'book_holder_1']



class PoseDet():
    def __init__(self, path='best.pt', desk='desk_depth.npy'):
        self.path = path
        self.cam_cx = 639.5
        self.cam_cy = 359.5
        self.cam_fx = 917.94347345
        self.cam_fy = 917.94347345
        self.cam_scale = 1.0
        self.img_width = 720
        self.img_length = 1280
        desk = np.load(desk)
        self.desk = (desk / self.cam_scale).reshape(1, 720, 1280)
        # xmap = np.array([[j for i in range(1280)] for j in range(720)])
        # ymap = np.array([[i for i in range(1280)] for j in range(720)])

        # pt0 = ((ymap - self.cam_cx) * desk / self.cam_fx).reshape(1, 720, 1280)
        # pt1 = ((xmap - self.cam_cy) * desk / self.cam_fy).reshape(1, 720, 1280)

        # self.desk_cld = np.concatenate((pt0, pt1, desk), axis=0).reshape(3,-1).T
        
        self.init_net()


    def init_net(self):
        self.detector = detNet(self.path)




    def inference(self, img, depth):

        bboxes= self.detector.detect(img)
        xmap = np.array([[j for i in range(1280)] for j in range(720)])
        ymap = np.array([[i for i in range(1280)] for j in range(720)])

        print(depth.shape)
        pt2 = (depth / self.cam_scale).reshape(1, 720, 1280)
        pt0 = ((ymap - self.cam_cx) * pt2 / self.cam_fx).reshape(1, 720, 1280)
        pt1 = ((xmap - self.cam_cy) * pt2 / self.cam_fy).reshape(1, 720, 1280)
        mask = np.zeros((1, 720, 1280))
        # pt2_crop = pt2
        pt2_crop = abs(pt2 - self.desk) #max(np.median(pt2[0,224:404, 624:724]), np.mean(pt2[0,224:404, 624:724]))
        # print(crop_value)
        pt2_crop[np.where(pt2_crop > 1)] = pt2[np.where(pt2_crop > 1)]
        mask[np.where(pt2_crop > 1)] = 1.0
        # cld = np.concatenate((pt0, pt1, pt2_crop), axis=0).reshape(3,-1).T

        my_result = [[] for _ in CLASSES]
        
        
        for idx in range(len(bboxes)):
            if len(bboxes[idx]) == 0:
                continue
            # print(f"class name : {self.class_list[idx]}")
            # print(f'index: {idx}')
            for bbox in bboxes[idx]:
                print(bbox)
                x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2],bbox[3]
                num_ = mask[0, y_min:y_max, x_min:x_max].sum()
                x_avg = (pt0[0, y_min:y_max, x_min:x_max] * mask[0, y_min:y_max, x_min:x_max]).sum() / num_
                y_avg = (pt1[0, y_min:y_max, x_min:x_max] * mask[0, y_min:y_max, x_min:x_max]).sum() / num_
                z_avg = (pt2_crop[0, y_min:y_max, x_min:x_max] * mask[0, y_min:y_max, x_min:x_max]).sum() / num_
                print(f"after:{x_avg, y_avg, z_avg}")
                my_result[idx].append([x_avg, y_avg, z_avg])
        return my_result


                    

if __name__ == "__main__":
    net = PoseDet()
    img = cv2.imread('./0000.png')
    dep = np.load('./0000.npy')
    net.inference(img, dep)