import glob
import math
import os
import json
from pyparsing import col
import torch
import cv2
import h5py
import numpy as np
import scipy.io as io
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter

import warnings 
warnings.filterwarnings("ignore")

'''change your path'''

root = './BCData/'

part_A_train = os.path.join(root, 'train')
part_A_val = os.path.join(root, 'val')
part_A_test = os.path.join(root, 'test')

path_sets = [part_A_train, part_A_val, part_A_test]
# path_sets = [part_A_train, part_A_val]


def generate_point_map(img_path, coordinates, color='red'):
    '''
    输入图像路径和点的坐标集合
    输出打点后的图像
    '''
    # print(type(img_path))
    if isinstance(img_path, str):
        Img_data = cv2.imread(img_path)
    else:
        Img_data = img_path
    ori_Img_data = Img_data.copy()

    point_size = 1
    if color == 'red':
        point_color = (0, 0, 255)
    elif color == 'blue':
        point_color = (0, 255, 255)
    thickness = 4
    for index, pt in enumerate(coordinates):
        pt2d = np.zeros([640, 640], dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
       
        Img_data = cv2.circle(Img_data, (int(pt[0]),int(pt[1])), point_size, point_color, thickness)

    # return Img_data
    return ori_Img_data, Img_data


def fidt_generate1(im_data, gt_data, lamda):
    size = im_data.shape
    new_im_data = cv2.resize(im_data, (lamda * size[1], lamda * size[0]), 0)

    new_size = new_im_data.shape
    # print(new_size[0], new_size[1]) # 1024
    d_map = (np.zeros([new_size[0], new_size[1]]) + 255).astype(np.uint8) # 变成全是255的图像
    gt = lamda * gt_data

    for o in range(0, len(gt)):
        x = np.max([1, math.floor(gt[o][1])]) # floor向下取整
        y = np.max([1, math.floor(gt[o][0])])
        if x >= new_size[0] or y >= new_size[1]: # 点超出范围
            # print("#####\n")
            continue
        d_map[x][y] = d_map[x][y] - 255 # 将有人头的点的值变为0，黑色
    # d_map：二值图像，要么0，要么255
    # 计算二值图像内任意点到人头点（值为0）的距离, 将前景对象提取出来。cv2.DIST_L2表示使用Euclid距离。
    # Calculates the distance to the closest zero pixel for each pixel of the source image.
    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 0)
    # print(len(distance_map[0]), distance_map)
    
    input_max = np.max(distance_map)
    distance_map = torch.from_numpy(distance_map)
    # EDT map
    distance_map = 1 / (1 + torch.pow(distance_map, 10 * (distance_map/input_max) + 0.5))

    distance_map = distance_map.numpy()
    distance_map[distance_map < 1e-2] = 0 # 过滤掉一些杂点

    return distance_map


img_paths = []
for path in path_sets:
    for filepath,dirnames,filenames in os.walk(path):
        # print(dirnames)
        for filename in filenames:
            # print(filename)
            if filename.split('.')[-1] == 'png':
                save_path = os.path.join(filepath, filename)
                # print(save_path)
                if "image" in save_path:
                    img_paths.append(save_path)
    # break
print(len(img_paths))

img_paths.sort()

K  = 0
for img_path in img_paths:
    # print(img_path)
    K += 1

    gt_file_path = img_path.replace("images", "annotations__").replace("png", "h5")
    paths = gt_file_path.split("__")
    gt_file_path_P = paths[0] + "/positive" + paths[1]
    gt_file_path_N = paths[0] + '/negative' + paths[1]
    # print(gt_file_path_P, gt_file_path_N)

    gt_file_P = h5py.File(gt_file_path_P)
    gt_file_N = h5py.File(gt_file_path_N)
    # print(gt_file)
    coordinates_P = np.asarray(gt_file_P['coordinates'])
    coordinates_N = np.asarray(gt_file_N['coordinates']) m
    # print(coordinates_P, len(coordinates_P))
    # print(len(coordinates_P),  len(coordinates_N))
    # 避免其中一个没有值
    if len(coordinates_N) and len(coordinates_P):
        coordinates = np.vstack([coordinates_N, coordinates_P])
    else:
        coordinates = coordinates_P if len(coordinates_P) else coordinates_N

    # coordinates = np.vstack([coordinates_N, coordinates_P])
    # print(len(coordinates_P),  len(coordinates_N), len(coordinates))
    # img, img_return = generate_point_map(img_path, coordinates, 'red')
    # print(img_return.shape)
    Img_data = cv2.imread(img_path)
    Img_data = cv2.resize(Img_data, (512, 512))

    # print(coordinates)
    re_coordinates = []
    for cor in coordinates:
        cor[0] = int(cor[0] / 640 * 512)
        cor[1] = int(cor[1] / 640 * 512)
        re_coordinates.append([cor[0], cor[1]])
    # print(re_coordinates)
    # break
    fidt_map1 = fidt_generate1(Img_data, re_coordinates, 1)

    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
    for i in range(0, len(re_coordinates)):
        if int(re_coordinates[i][1]) < Img_data.shape[0] and int(re_coordinates[i][0]) < Img_data.shape[1]:
            kpoint[int(re_coordinates[i][1]), int(re_coordinates[i][0])] = 1

    h5_path = img_path.replace('.png', '.h5').replace('images', 'location_map')
    # print(h5_path)
    # print(os.path.join(h5_path.split('/')[:-1]))
    # if not os.path.exists(os.path.join(h5_path.split('/')[:-1])):
    #     os.makedirs(os.path.join(h5_path.split('/')[:-1]))

    with h5py.File(h5_path, 'w') as hf:
        hf['fidt_map'] = fidt_map1
        hf['kpoint'] = kpoint

    fidt_map1 = fidt_map1
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2) # 生成伪彩色图

    '''for visualization'''
    
    cv2_path = img_path.replace('images', 'location_map_show')
    # print(cv2_path)
    cv2.imwrite(cv2_path, fidt_map1)
    cv2.imwrite(img_path, Img_data)
    # break

print(K)