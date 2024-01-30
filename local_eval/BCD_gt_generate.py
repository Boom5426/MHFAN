import os
import time
import glob
import cv2
import json
import h5py
import numpy as np
import scipy.io
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter
import math
import scipy.io as io
from matplotlib import pyplot as plt
import sys

'''please set your dataset path'''
root = '../data/CoNIC/'

part_A_test = os.path.join(root, 'train', 'images')
path_sets = [part_A_test]

if not os.path.exists(part_A_test):
    sys.exit("The path is wrong, please check the dataset path.")

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.png')):
        img_paths.append(img_path)

img_paths.sort()

f = open('./gt_file/BCD_train_gt.txt', 'w+')
k = 1
for img_path in img_paths:

    print(img_path)
    # loc = json.load(open(img_path.replace('.jpg', '.json').replace('images','masks'), 'r', encoding="utf-8"))['shapes']

    gt_file_path = img_path.replace("images", "annotations__").replace("png", "h5")
    paths = gt_file_path.split("__")
    gt_file_path_P = paths[0] + "/positive" + paths[1]
    gt_file_path_N = paths[0] + '/negative' + paths[1]

    # print(gt_file_path_P, gt_file_path_N)

    gt_file_P = h5py.File(gt_file_path_P)
    gt_file_N = h5py.File(gt_file_path_N)
    # print(gt_file)
    coordinates_P = np.asarray(gt_file_P['coordinates'])
    coordinates_N = np.asarray(gt_file_N['coordinates'])
    # print(coordinates_P, len(coordinates_P))
    # print(len(coordinates_P),  len(coordinates_N))
    # 避免其中一个没有值
    if len(coordinates_N) and len(coordinates_P):
        coordinates = np.vstack([coordinates_N, coordinates_P])
    else:
        coordinates = coordinates_P if len(coordinates_P) else coordinates_N

    # print(loc)
    re_coordinates = []
    for cor in coordinates:
        cor[0] = int(cor[0] / 640 * 512)
        cor[1] = int(cor[1] / 640 * 512)
        re_coordinates.append([cor[0], cor[1]])
    # mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
    # Gt_data = mat["image_info"][0][0][0][0][0]
    f.write('{} {} '.format(k, len(re_coordinates)))

    for data in re_coordinates:
        sigma_s = 5
        sigma_l = 10
        f.write('{} {} {} {} {} '.format(math.floor(data[0]), math.floor(data[1]), sigma_s, sigma_l, 1))
    f.write('\n')

    k = k + 1
f.close()
