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


# '''please set your dataset path'''
# root = '../data/PSU/'

# part_A_test = os.path.join(root, 'test', 'images')
# path_sets = [part_A_test]

# if not os.path.exists(part_A_test):
#     sys.exit("The path is wrong, please check the dataset path.")

# img_paths = []
# for path in path_sets:
#     for img_path in glob.glob(os.path.join(path, '*.jpg')):
#         img_paths.append(img_path)

# img_paths.sort()

# f = open('./gt_file/PSU_test_gt.txt', 'w+')
# k = 1
# for img_path in img_paths:
#     print(img_path)  # ../data/UW/test/images/img13.jpg

#     # ../data/UW/PSU_dataset/img13/img13_detection.mat

#     mat_path = img_path.replace('.jpg', '_detection.mat').replace('test/images','PSU_dataset')
#     # ../data/PSU/PSU_dataset/img97_detection.mat
#     tmp_path = mat_path.split('/')[-1].split('_')[0]
#     tmp = mat_path.split('/')
#     mat_path = os.path.join(tmp[0], tmp[1], tmp[2], tmp[3], tmp_path, tmp[-1])

#     print(mat_path)
#     # break
#     mat = io.loadmat(mat_path)
#     Gt_data = mat['detection']

#     re_coordinates = []
#     for cor in Gt_data:
#         # print(cor)
#         cor[0] = int(cor[0] / 612 * 512)
#         cor[1] = int(cor[1] / 452 * 512)
#         # print([cor[0], cor[1]])
#         re_coordinates.append([cor[0], cor[1]])
#     # break
#     f.write('{} {} '.format(k, len(re_coordinates)))

#     for data in re_coordinates:
#         sigma_s = 5
#         sigma_l = 10
#         f.write('{} {} {} {} {} '.format(math.floor(data[0]), math.floor(data[1]), sigma_s, sigma_l, 1))
#     f.write('\n')

#     k += 1
# f.close()


'''please set your dataset path'''
root = '../data/UW/'

part_A_test = os.path.join(root, 'fold2', 'images')
path_sets = [part_A_test]

if not os.path.exists(part_A_test):
    sys.exit("The path is wrong, please check the dataset path.")

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

img_paths.sort()

f = open('./gt_file/UW_test2_gt.txt', 'w+')
k = 1
for img_path in img_paths:
    print(img_path)  # ../data/UW/test/images/img13.jpg

    # ../data/UW/PSU_dataset/img13/img13_detection.mat

    mat_path = img_path.replace('.jpg', '_detection.mat').replace('fold2/images','UW_dataset')
    # ../data/PSU/PSU_dataset/img97_detection.mat
    tmp_path = mat_path.split('/')[-1].split('_')[0]
    tmp = mat_path.split('/')
    mat_path = os.path.join(tmp[0], tmp[1], tmp[2], tmp[3], 'Detection', tmp_path, tmp[-1])

    print(mat_path)
    # break
    mat = io.loadmat(mat_path)
    Gt_data = mat['detection']

    re_coordinates = []
    for cor in Gt_data:
        cor[0] = int(cor[0] / 500 * 512)
        cor[1] = int(cor[1] / 500 * 512)
        re_coordinates.append([cor[0], cor[1]])

    f.write('{} {} '.format(k, len(re_coordinates)))

    for data in re_coordinates:
        sigma_s = 6
        sigma_l = 10
        f.write('{} {} {} {} {} '.format(math.floor(data[0]), math.floor(data[1]), sigma_s, sigma_l, 1))
    f.write('\n')

    k += 1
f.close()
