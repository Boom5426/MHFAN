import warnings 
warnings.filterwarnings("ignore")

import math
import os
import ast
from pyparsing import col
import torch
import cv2
import h5py
import numpy as np


'''change your path'''

root = './'

part_A_train = os.path.join(root, 'train')
part_A_test = os.path.join(root, 'test')

path_sets = [part_A_train, part_A_test]


def EDT_generate(im_data, gt):
    '''
    code of EDT map: Exponential Distance Transform Maps for Cell Localization
    '''
    size = im_data.shape
    new_im_data = cv2.resize(im_data, (size[1], size[0]), 0)

    new_size = new_im_data.shape
    d_map = (np.zeros([new_size[0], new_size[1]]) + 255).astype(np.uint8)

    for o in range(0, len(gt)):
        x = np.max([1, math.floor(gt[o][1])])
        y = np.max([1, math.floor(gt[o][0])])
        if x >= new_size[0] or y >= new_size[1]:
            continue
        d_map[x][y] = d_map[x][y] - 255
    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 0)
    
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
            if filename.split('.')[-1] == 'jpg':
                save_path = os.path.join(filepath, filename)
                # print(save_path)
                if "image" in save_path:
                    img_paths.append(save_path)
    # break
print("Images:",len(img_paths))

img_paths.sort()

K  = 0
train_path = "./train/location_train.txt"
test_path = "./test/location_test.txt"
for img_path in img_paths:
    print(img_path)
    if "train" in img_path:
        txt_path = train_path
    else:
        txt_path = test_path
    K += 1

    Img_data = cv2.imread(img_path)

    try:
        with open(txt_path, 'r') as file:
            for line in file:
                cleaned_line = line.strip()
                if cleaned_line.split(':')[0] == os.path.basename(img_path).split('.')[0]:
                    print(cleaned_line.split(':')[0], os.path.basename(img_path).split('.')[0])
                    break
    except FileNotFoundError:
        print(f"File '{txt_path}' not find")
    
    re_coordinates = cleaned_line.split(':')[1]
    re_coordinates = ast.literal_eval(re_coordinates)
    # print(len(re_coordinates))

    # paper: 
    fidt_map = EDT_generate(Img_data, re_coordinates)

    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
    for i in range(0, len(re_coordinates)):
        if int(re_coordinates[i][1]) < Img_data.shape[0] and int(re_coordinates[i][0]) < Img_data.shape[1]:
            kpoint[int(re_coordinates[i][1]), int(re_coordinates[i][0])] = 1

    location_map_path = img_path.replace('.jpg', '.h5').replace('images', 'location_map')
    print(location_map_path)

    # with h5py.File(location_map_path, 'w') as hf:
    #     hf['fidt_map'] = fidt_map
    #     hf['kpoint'] = kpoint

    fidt_map = fidt_map
    fidt_map = fidt_map / np.max(fidt_map) * 255
    fidt_map = fidt_map.astype(np.uint8)
    fidt_map = cv2.applyColorMap(fidt_map, 2) # 生成伪彩色图

    # '''for visualization'''    
    cv2_path = img_path.replace('images', 'location_map_show')
    print(cv2_path)
    # cv2.imwrite(cv2_path, fidt_map)
    # break

print("Processd data:",K)


# Reference of EDT map
'''
@article{li2023exponential,
  title={Exponential Distance Transform Maps for Cell Localization},
  author={Li, Bo and Chen, Jie and Yi, Hang and Feng, Min and Yang, Yongquan and Bu, Hong},
  year={2023},
  publisher={TechRxiv}
}
'''