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

import scipy.spatial as T
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
    print(paths)
    gt_file_path_P = paths[0] + "/positive" + paths[1]
    gt_file_path_N = paths[0] + '/negative' + paths[1]
    # print(gt_file_path_P, gt_file_path_N)

    gt_file_P = h5py.File(gt_file_path_P)
    gt_file_N = h5py.File(gt_file_path_N)
    # print(gt_file)
    coordinates_P = np.asarray(gt_file_P['coordinates'])
    coordinates_N = np.asarray(gt_file_N['coordinates'])

    # coordinates = np.vstack([coordinates_N, coordinates_P])
    # print(len(coordinates_P),  len(coordinates_N), len(coordinates))
    img, img_return = generate_point_map(img_path, coordinates_P, 'red')
    # print(img_return.shape)

    _, img_return = generate_point_map(img_return, coordinates_N, 'blue')
    # print(img_return.shape)

    # res = np.hstack((img, img_return))
    img_save_path = img_path.replace("images", "IMG_Annotations")
    print(img_save_path)
    # if not os.path.exists(img_save_path):
    #     os.makedirs(img_save_path)
    cv2.imwrite(img_save_path, img_return)
    
    # break
    # Gt_data = np.asarray(gt_file['density'])
    # print(Gt_data.shape)
    # break
    # mat = io.loadmat(img_path.replace('.png', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
    # Gt_data = mat["image_info"][0][0][0][0][0]

    # fidt_map1 = fidt_generate1(Img_data, Gt_data, 1)

    # kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
    # for i in range(0, len(Gt_data)):
    #     if int(Gt_data[i][1]) < Img_data.shape[0] and int(Gt_data[i][0]) < Img_data.shape[1]:
    #         kpoint[int(Gt_data[i][1]), int(Gt_data[i][0])] = 1
    # h5_path = img_path.replace('.png', '.h5').replace('images', 'gt_fidt_map')
    # print(h5_path)
    # with h5py.File(h5_path, 'w') as hf:
    #     hf['fidt_map'] = fidt_map1
    #     hf['kpoint'] = kpoint

    # fidt_map1 = fidt_map1
    # fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    # fidt_map1 = fidt_map1.astype(np.uint8)
    # fidt_map1 = cv2.applyColorMap(fidt_map1, 2)

    # '''for visualization'''
    # cv2_path = img_path.replace('images', 'gt_show').replace('png', 'jpg')
    # print(cv2_path)
    # cv2.imwrite(cv2_path, fidt_map1)

print(K)