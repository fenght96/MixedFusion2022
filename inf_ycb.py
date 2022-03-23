import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '/home/fht/data/ocrtoc', help='dataset root dir')
parser.add_argument('--model', type=str, default = '/home/fht/code/DenseFusion-1-Pytorch-1.6/trained_models/ycb/pose_model_current.pth',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '/home/fht/code/DenseFusion-1-Pytorch-1.6/trained_models/ycb/pose_refine_model_current.pth',  help='resume PoseRefineNet model')
opt = parser.parse_args()

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800, 840, 880, 920, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280]
xmap = np.array([[j for i in range(1280)] for j in range(720)])
ymap = np.array([[i for i in range(1280)] for j in range(720)])
cam_cx = 649.62834545
cam_cy = 367.13343954
cam_fx = 907.19708252
cam_fy = 924.39801025
cam_scale = 1000.0
num_obj = 33
img_width = 760
img_length = 1280
num_points = 1000
num_points_mesh = 500
iteration = 2
bs = 1
dataset_config_dir = 'datasets/ycb/dataset_config'
root = '/home/fht/data/ocrtoc'
ycb_toolbox_dir = 'YCB_Video_toolbox'
result_wo_refine_dir = 'experiments/eval_result/ycb/Densefusion_wo_refine_result'
result_refine_dir = 'experiments/eval_result/ycb/Densefusion_iterative_result'

def get_bbox(posecnn_rois):
    rmin = int(posecnn_rois[idx][3]) + 1
    rmax = int(posecnn_rois[idx][5]) - 1
    cmin = int(posecnn_rois[idx][2]) + 1
    cmax = int(posecnn_rois[idx][4]) - 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
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
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax




estimator = PoseNet(num_points = num_points, num_obj = num_obj)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.model))
estimator.eval()

refiner = PoseRefineNet(num_points = num_points, num_obj = num_obj)
refiner.cuda()
refiner.load_state_dict(torch.load(opt.refine_model))
refiner.eval()


testlist = []
input_file = open('{0}/test.txt'.format(dataset_config_dir))
while 1:
    input_line = input_file.readline()
    if not input_line:
        break
    if input_line[-1:] == '\n':
        input_line = input_line[:-1]
    testlist.append(input_line.replace('/home/casia/code/icra/ocrtoc',root))
input_file.close()


class_file = open('/home/fht/data/ocrtoc/object_name_list.txt'.format(dataset_config_dir))
class_list = []
class_id = 1
cld = {}
while 1:
    class_input = class_file.readline()
    if not class_input:
        break
    class_input = class_input.strip()
    class_list.append(class_input)

    input_file = o3d.io.read_point_cloud('{0}/rgb_pcd/{1}.ply'.format(root, class_input))
    input_file = np.asarray(input_file.points)
    cld[class_id] = []
    for i in range(input_file.shape[0]):
        input_line = input_file[i]
        cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])

    cld[class_id] = np.array(cld[class_id])
    class_id += 1

for now in range(0, 1):
    testlist[now] = '/home/fht/data/ocrtoc/scenes/20210512171024/rgb_undistort/0000'

    print(f'test path :{testlist[now]}')
    with open(testlist[now].replace('rgb_undistort', 'obj_out_pose') + '.txt', 'r') as f:
        lines = f.readlines()
    for line in lines:
         line = line.strip().split(':')
         pose = json.loads(line[1])
         pose = np.array(pose)
         r = Ro.from_matrix(pose[:-1,:-1])
         print(f'name:{line[0]}')
         print(r.as_quat())




    img = Image.open('{0}.png'.format(testlist[now]))
    depth = np.array(Image.open('{0}.png'.format(testlist[now].replace('rgb_undistort', 'depth_undistort'))))

    label = np.load('{0}.npy'.format(testlist[now].replace('rgb_undistort', 'seg_masks')))
    mask_back = ma.getmaskarray(ma.masked_equal(label, 0))

    obj_file = open(testlist[now][:-4].replace('rgb_undistort/', 'object_list.txt'))
    obj_list = obj_file.readlines()
    obj_file.close()
    lst = [1 +  class_list.index(ob.strip()) for ob in obj_list]


    my_result_wo_refine = []
    my_result = []
    
    for idx in range(len(lst)):
        itemid = lst[idx]
        print(f"class name : {class_list[lst[idx] - 1]}")
        print(f'index: {lst[idx]}')
        try:
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, lst[idx]))
            mask = mask_label * mask_depth

            if len(mask.nonzero()[0]) <= num_points:
                continue

            rmin, rmax, cmin, cmax = get_bbox(mask_label)



            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose) > num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')



            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])

            pt2 = depth_masked / cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)

            img_masked = np.array(img)[:, :, :3]
            img_masked = np.transpose(img_masked, (2, 0, 1))
            img_masked = img_masked[:, rmin:rmax, cmin:cmax]

            cloud = torch.from_numpy(cloud.astype(np.float32))
            choose = torch.LongTensor(choose.astype(np.int32))
            img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
            index = torch.LongTensor([itemid - 1])

            cloud = Variable(cloud).cuda()
            choose = Variable(choose).cuda()
            img_masked = Variable(img_masked).cuda()
            index = Variable(index).cuda()

            cloud = cloud.view(1, num_points, 3)
            img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

            pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

            pred_c = pred_c.view(bs, num_points)
            how_max, which_max = torch.max(pred_c, 1)
            pred_t = pred_t.view(bs * num_points, 1, 3)
            points = cloud.view(bs * num_points, 1, 3)

            my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
            my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
            print(f"my_r:{my_r}\t my_t:{my_t}")
            my_pred = np.append(my_r, my_t)
            print(f"my_pred:{my_pred.tolist()}")
            my_result_wo_refine.append(my_pred.tolist())

            for ite in range(0, iteration):
                T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
                my_mat = quaternion_matrix(my_r)
                R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                my_mat[0:3, 3] = my_t
                
                new_cloud = torch.bmm((cloud - T), R).contiguous()
                pred_r, pred_t = refiner(new_cloud, emb, index)
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
            r = Ro.from_quat(my_r)
            print(r.as_matrix())
            my_pred = np.append(my_r, my_t)
            my_result.append(my_pred.tolist())
            print(f'my_pred final:{my_pred}')
            break
        except ZeroDivisionError:
            print("PoseCNN Detector Lost {0} at No.{1} keyframe".format(itemid, now))
            my_result_wo_refine.append([0.0 for i in range(7)])
            my_result.append([0.0 for i in range(7)])

    