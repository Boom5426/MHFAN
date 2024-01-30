import os
import time
import glob
import cv2
import json
import h5py

import ast
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
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

img_paths.sort()


txt_path = "/home/zy/libo/Cell_localization/data/CoNIC/train/location_train.txt"
f = open('./gt_file/CoNIC_train_gt.txt', 'w+')
k = 1
for img_path in img_paths:

    # print(img_path)

    try:
        with open(txt_path, 'r') as file:
            for line in file:
                cleaned_line = line.strip()
                # print(cleaned_line.split(':')[0], os.path.basename(img_path))
                if cleaned_line.split(':')[0] == os.path.basename(img_path).split('.')[0]:
                    print(cleaned_line.split(':')[0], os.path.basename(img_path).split('.')[0])
                    break
    except FileNotFoundError:
        print(f"文件 '{txt_path}' 未找到")
    
    re_coordinates = cleaned_line.split(':')[1]
    # re_coordinates = parts[1].strip()
    # print(re_coordinates)
    # re_coordinates = list(re_coordinates)
    re_coordinates = ast.literal_eval(re_coordinates)
    # loc = json.load(open(img_path.replace('.jpg', '.json').replace('images','masks'), 'r', encoding="utf-8"))['shapes']

    # print(loc)
    # re_coordinates = []

    f.write('{} {} '.format(k, len(re_coordinates)))

    for data in re_coordinates:
        sigma_s = 5
        sigma_l = 10
        f.write('{} {} {} {} {} '.format(math.floor(data[0]), math.floor(data[1]), sigma_s, sigma_l, 1))
    f.write('\n')

    k = k + 1
    # break
f.close()
