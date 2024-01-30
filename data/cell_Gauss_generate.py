import glob
import math
import os
import json
import torch
import cv2
import h5py
import numpy as np
import scipy.io as io
import scipy.spatial
from scipy.ndimage import gaussian_filter

'''change your path'''
root = './cell_dataset/'

part_A_train = os.path.join(root, 'x40S/training', 'images')
part_A_test = os.path.join(root, 'x40S/validation', 'images')
# part_B_train = os.path.join(root, 'x40/training', 'images')
# part_B_test = os.path.join(root, 'x40/validation', 'images')

path_sets = [part_A_train, part_A_test]
# path_sets = [ part_A_test]

if not os.path.exists(part_A_train.replace('images', 'gt_Gus_map')):
    os.makedirs(part_A_train.replace('images', 'gt_Gus_map'))

if not os.path.exists(part_A_test.replace('images', 'gt_Gus_map')):
    os.makedirs(part_A_test.replace('images', 'gt_Gus_map'))

if not os.path.exists(part_A_train.replace('images', 'gt_Gus_show')):
    os.makedirs(part_A_train.replace('images', 'gt_Gus_show'))

if not os.path.exists(part_A_test.replace('images', 'gt_Gus_show')):
    os.makedirs(part_A_test.replace('images', 'gt_Gus_show'))

# if not os.path.exists(part_B_train.replace('images', 'gt_fidt_map')):
#     os.makedirs(part_B_train.replace('images', 'gt_fidt_map'))

# if not os.path.exists(part_B_test.replace('images', 'gt_fidt_map')):
#     os.makedirs(part_B_test.replace('images', 'gt_fidt_map'))

# if not os.path.exists(part_B_train.replace('images', 'gt_show')):
#     os.makedirs(part_B_train.replace('images', 'gt_show'))

# if not os.path.exists(part_B_test.replace('images', 'gt_show')):
#     os.makedirs(part_B_test.replace('images', 'gt_show'))

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

img_paths.sort()


def fidt_generate1(im_data, gt_data, lamda):
    size = im_data.shape
    new_im_data = cv2.resize(im_data, (lamda * size[1], lamda * size[0]), 0)

    new_size = new_im_data.shape
    d_map = (np.zeros([new_size[0], new_size[1]]) + 255).astype(np.uint8)
    gt = lamda * gt_data

    for o in range(0, len(gt)):
        x = np.max([1, math.floor(gt[o][1])])
        y = np.max([1, math.floor(gt[o][0])])
        if x >= new_size[0] or y >= new_size[1]:
            continue
        d_map[x][y] = d_map[x][y] - 255

    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 0)
    distance_map = torch.from_numpy(distance_map)
    distance_map = 1 / (1 + torch.pow(distance_map, 0.02 * distance_map + 0.75))
    distance_map = distance_map.numpy()
    distance_map[distance_map < 1e-2] = 0

    return distance_map

def gaussian_filter_density(gt):
    # print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    # 数组array中非零元素的位置(数组索引)的函数。
    pts = list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))
    
    leafsize = 2048
    # build kdtree
    # pts = list(pts)
    # KDTree 快速查找 nearest-neighbor
    tree = scipy.spatial.KDTree(pts, leafsize=leafsize)
    # query kdtree
    # pts = np.array(pts)
    # 查询kd-tree附近的邻居,4个邻居
    distances, locations = tree.query(pts, k=4)

    # print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            # 取周围3个点的距离来作为滤波，这样越松散的地方扩散越多，越紧密的地方越紧密
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        # sigma： 高斯函数里面的\sigma，\sigma越大滤波越厉害
        density += gaussian_filter(pt2d, sigma, mode='constant')
    # print('done.')
    return density

K  = 0
for img_path in img_paths:
    print(img_path)
    K += 1
    Img_data = cv2.imread(img_path)

    loc = json.load(open(img_path.replace('.jpg', '.json').replace('images','masks'), 'r', encoding="utf-8"))['shapes']
    # print(loc)
    Gt_data = []
    for points in loc:
        tmp_point = points['points']
        Gt_data.append(tmp_point[0])

    number = len(Gt_data)

    # break
    # mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
    # Gt_data = mat["image_info"][0][0][0][0][0]

    # fidt_map1 = fidt_generate1(Img_data, Gt_data, 1)

    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
    for i in range(0, len(Gt_data)):
        if int(Gt_data[i][1]) < Img_data.shape[0] and int(Gt_data[i][0]) < Img_data.shape[1]:
            kpoint[int(Gt_data[i][1]), int(Gt_data[i][0])] = 1
    # print(kpoint.shape)
    Guass_map = gaussian_filter_density(kpoint)
    # print(Guass_map.shape)

    h5_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_Gus_map')
    # print(h5_path)
    with h5py.File(h5_path, 'w') as hf: 
        hf['Guass_map'] = Guass_map
        hf['kpoint'] = kpoint
        hf['number'] = number

    Guass_map = Guass_map
    Guass_map = Guass_map / np.max(Guass_map) * 255
    Guass_map = Guass_map.astype(np.uint8)
    Guass_map = cv2.applyColorMap(Guass_map, 2)
    # print(Guass_map.shape)

    '''for visualization'''
    # cv2_path = '../guss.jpg'
    cv2_path = img_path.replace('images', 'gt_Gus_show').replace('jpg', 'jpg')
    # print(cv2_path)
    cv2.imwrite(cv2_path, Guass_map)
    # break

print(K)