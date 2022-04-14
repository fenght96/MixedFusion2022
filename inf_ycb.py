import os
import copy
import random
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
import json
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from scipy.spatial.transform import Rotation as Ro
from yolo_det import Net as detNet


class PoseDet():
    def __init__(self, dataset_root = '/home/fht/data/ocrtoc', model = '/home/fht/code/DenseFusion-1-Pytorch-1.6/trained_models/ycb/pose_model_current.pth', refine_model = '/home/fht/code/DenseFusion-1-Pytorch-1.6/trained_models/ycb/pose_refine_model_current.pth'):
        self.root = dataset_root
        self.model = model
        self.refine_model = refine_model
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800, 840, 880, 920, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280]
        self.xmap = np.array([[j for i in range(1280)] for j in range(720)])
        self.ymap = np.array([[i for i in range(1280)] for j in range(720)])
        self.cam_cx = 649.62834545
        self.cam_cy = 367.13343954
        self.cam_fx = 907.19708252
        self.cam_fy = 924.39801025
        self.cam_scale = 1000.0
        self.num_obj = 33
        self.img_width = 760
        self.img_length = 1280
        self.num_points = 1000
        self.num_points_mesh = 500
        self.iteration = 2


        class_file = open('{0}/object_name_list.txt'.format(self.root))
        self.class_list = []
        class_id = 1
        self.cld = {}
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break
            class_input = class_input.strip()
            self.class_list.append(class_input)

            input_file = o3d.io.read_point_cloud('{0}/rgb_pcd/{1}.ply'.format(self.root, class_input))
            input_file = np.asarray(input_file.points)
            self.cld[class_id] = []
            for i in range(input_file.shape[0]):
                input_line = input_file[i]
                self.cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])

            self.cld[class_id] = np.array(self.cld[class_id])
            class_id += 1
        self.init_net()


    def init_net(self):
        self.estimator = PoseNet(num_points = self.num_points, num_obj = self.num_obj)
        self.estimator.cuda()
        self.estimator.load_state_dict(torch.load(self.model))
        self.estimator.eval()

        self.refiner = PoseRefineNet(num_points = self.num_points, num_obj = self.num_obj)
        self.refiner.cuda()
        self.refiner.load_state_dict(torch.load(self.refine_model))
        self.refiner.eval()

        self.detector = detNet()

    def get_bbox(self, posecnn_rois):
        rmin = int(posecnn_rois[0]) + 1
        rmax = int(posecnn_rois[2]) - 1
        cmin = int(posecnn_rois[1]) + 1
        cmax = int(posecnn_rois[3]) - 1
        r_b = rmax - rmin
        for tt in range(len(self.border_list)):
            if r_b > self.border_list[tt] and r_b < self.border_list[tt + 1]:
                r_b = self.border_list[tt + 1]
                break
        c_b = cmax - cmin
        for tt in range(len(self.border_list)):
            if c_b > self.border_list[tt] and c_b < self.border_list[tt + 1]:
                c_b = self.border_list[tt + 1]
                break
        center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
        rmin = center[0] - int(r_b / 2)
        rmax = center[0] + int(r_b / 2)
        cmin = center[1] - int(c_b / 2)
        cmax = center[1] + int(c_b / 2)
        if rmin < 0:
            delt = -rmin
            rmin = 0
            rmax += delt
        if cmin < 0:
            delt = -cmin
            cmin = 0
            cmax += delt
        if rmax > self.img_width:
            delt = rmax - self.img_width
            rmax = self.img_width
            rmin -= delt
        if cmax > self.img_length:
            delt = cmax - self.img_length
            cmax = self.img_length
            cmin -= delt
        return rmin, rmax, cmin, cmax





    def inference(self, img, depth):

        bboxes= self.detector.detect(img)

        my_result_wo_refine = []
        my_result = []
        
        for idx in range(len(bboxes)):
            if len(bboxes[idx]) == 0:
                continue
            # print(f"class name : {self.class_list[idx]}")
            # print(f'index: {idx}')
            for bbox in bboxes[idx]:
                try:
                    print(bbox)
                    rmin, rmax, cmin, cmax = self.get_bbox(bbox)
                    mask = np.zeros(img.shape[:-1])
                    mask[rmin:rmax, cmin:cmax] = 1.0

                    # dep_min = np.min(depth[rmin:rmax,  cmin:cmax])
                    # dep_max = np.min(depth[rmin:rmax,  cmin:cmax])
                    # dep_median = np.median(depth[rmin:rmax,  cmin:cmax])

                    choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
                
                    if len(choose) > self.num_points:
                        c_mask = np.zeros(len(choose), dtype=int)
                        c_mask[:self.num_points] = 1
                        np.random.shuffle(c_mask)
                        choose = choose[c_mask.nonzero()]
                    else:
                        choose = np.pad(choose, (0, self.num_points - len(choose)), 'wrap')

                    depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                    xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                    ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                    choose = np.array([choose])

                    pt2 = depth_masked / self.cam_scale
                    pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
                    pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
                    cloud = np.concatenate((pt0, pt1, pt2), axis=1)

                    img_masked = np.array(img)[:, :, :3]
                    img_masked = np.transpose(img_masked, (2, 0, 1))
                    img_masked = img_masked[:, rmin:rmax, cmin:cmax]

                    cloud = torch.from_numpy(cloud.astype(np.float32))
                    choose = torch.LongTensor(choose.astype(np.int32))
                    img_masked = self.norm(torch.from_numpy(img_masked.astype(np.float32)))
                    index = torch.LongTensor([idx])

                    cloud = Variable(cloud).cuda()
                    choose = Variable(choose).cuda()
                    img_masked = Variable(img_masked).cuda()
                    index = Variable(index).cuda()

                    cloud = cloud.view(1, self.num_points, 3)
                    img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

                    pred_r, pred_t, pred_c, emb = self.estimator(img_masked, cloud, choose, index)
                    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, self.num_points, 1)

                    pred_c = pred_c.view(1, self.num_points)
                    how_max, which_max = torch.max(pred_c, 1)
                    pred_t = pred_t.view(1 * self.num_points, 1, 3)
                    points = cloud.view(1 * self.num_points, 1, 3)

                    my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
                    my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
                    print(f"my_r:{my_r}\t my_t:{my_t}")
                    my_pred = np.append(my_r, my_t)
                    print(f"my_pred:{my_pred.tolist()}")
                    my_result_wo_refine.append(my_pred.tolist())

                    for ite in range(0, self.iteration):
                        T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(self.num_points, 1).contiguous().view(1, self.num_points, 3)
                        my_mat = quaternion_matrix(my_r)
                        R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                        my_mat[0:3, 3] = my_t
                        
                        new_cloud = torch.bmm((cloud - T), R).contiguous()
                        pred_r, pred_t = self.refiner(new_cloud, emb, index)
                        pred_r = pred_r.view(1, 1, -1)
                        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                        my_r_2 = pred_r.view(-1).cpu().data.numpy()
                        my_t_2 = pred_t.view(-1).cpu().data.numpy()
                        my_mat_2 = quaternion_matrix(my_r_2)

                        my_mat_2[0:3, 3] = my_t_2

                        my_mat_final = np.dot(my_mat, my_mat_2)
                        my_r_final = copy.deepcopy(my_mat_final)
                        my_r_final[0:3, 3] = 0
                        my_r_final = quaternion_from_matrix(my_r_final, True)
                        my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                        my_pred = np.append(my_r_final, my_t_final)
                        my_r = my_r_final
                        my_t = my_t_final
                    # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)
                    # r = Ro.from_quat(my_r)
                    # print(r.as_matrix())
                    my_pred = np.append(my_r, my_t)
                    my_result.append(my_pred.tolist())
                    print(f'my_pred final:{my_pred}')
                    break
                except ZeroDivisionError:
                    print("PoseCNN Detector Lost at No.{1} keyframe".format(idx))
                    my_result_wo_refine.append([0.0 for i in range(7)])
                    my_result.append([0.0 for i in range(7)])

if __name__ == "__main__":
    net = PoseDet()
    img = cv2.imread('/home/fht/data/ocrtoc/scenes/20210512154044/rgb_undistort/0000.png')
    dep = cv2.imread('/home/fht/data/ocrtoc/scenes/20210512154044/depth_undistort/0000.png')
    net.inference(img, dep)