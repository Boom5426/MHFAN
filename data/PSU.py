from email.mime import image
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
from scipy.ndimage import gaussian_filter

import warnings 
warnings.filterwarnings("ignore")

root = './PSU/PSU_dataset/'

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
    # FIDTM
    # distance_map = 1 / (1 + torch.pow(distance_map, 0.02 * distance_map + 0.75))
    distance_map = 1 / (1 + torch.pow(distance_map, 10 * (distance_map/input_max) + 0.5))

    distance_map = distance_map.numpy()
    distance_map[distance_map < 1e-2] = 0 # 过滤掉一些杂点

    return distance_map

img_paths = []
for filepath,dirnames,filenames in os.walk(root):
    for filename in filenames:
        if filename.split('.')[-1] == 'bmp':
            save_path = os.path.join(filepath, filename)
            img_paths.append(save_path)

# img_paths.sort()
print(len(img_paths))

k = 0
for img_path in img_paths:
    # print(img_path)
    k += 1
    Img_data = cv2.imread(img_path)

    mat = io.loadmat(img_path.replace('.bmp', '_detection.mat'))
    Gt_data = mat['detection']

    Img_data = cv2.resize(Img_data, (512, 512))
    # print(coordinates)
    re_coordinates = []
    for cor in Gt_data:
        cor[0] = int(cor[0] / 612 * 512)
        cor[1] = int(cor[1] / 452 * 512)
        re_coordinates.append([cor[0], cor[1]])
    # print(re_coordinates)

    fidt_map1 = fidt_generate1(Img_data, re_coordinates, 1)

    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
    for i in range(0, len(re_coordinates)):
        if int(re_coordinates[i][1]) < Img_data.shape[0] and int(re_coordinates[i][0]) < Img_data.shape[1]:
            kpoint[int(re_coordinates[i][1]), int(re_coordinates[i][0])] = 1

    # 前面84个放进训练集，后面放进测试集
    path = 'train' if k<=84 else 'test'
    
    h5_path = img_path.replace('.bmp', '.h5').replace('PSU_dataset', path).replace(img_path.split('/')[-2],'location_map', 1)

    # print(h5_path)
    with h5py.File(h5_path, 'w') as hf:
        hf['fidt_map'] = fidt_map1
        hf['kpoint'] = kpoint

    img_path = h5_path.replace('h5', 'jpg').replace('location_map', 'images')
    # print(img_path)
    cv2.imwrite(img_path, Img_data)

    fidt_map1 = fidt_map1
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2) # 生成伪彩色图

    '''for visualization'''
    # location_map_path = root.split('\\')[0].replace('PSU_dataset', 'location_map_show')
    # if not os.path.exists(location_map_path):
    #     os.mkdir(location_map_path)
    cv2_path = h5_path.replace('location_map', 'location_map_show').replace('h5', 'jpg')

    # print(cv2_path)
    cv2.imwrite(cv2_path, fidt_map1)


train_files = os.listdir('./PSU/train/images')
test_files = os.listdir('./PSU/test/images')

print(len(train_files), len(test_files))