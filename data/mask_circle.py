from PIL import  Image
import os
import cv2 as cv
import matplotlib.pyplot as plt
import glob
from pylab import plot
import numpy as np
import json
import math
import scipy.spatial as T
from functions import euclidean_dist,  generate_cycle_mask, average_del_min
mode = 'train'

img_path = './cell_dataset/BCData/train/images'
json_path = './cell_dataset/BCData//jsons'
mask_path = './cell_dataset/BCData//masks'

img_path = './cell_dataset/x40/training/images'
json_path = './cell_dataset/x40/training/jsons'
mask_path = './cell_dataset/x40/training/masks'

cycle  =False
if  not os.path.exists(mask_path):
    os.makedirs(mask_path)

def generate_mask(height, width):
    x, y = np.ogrid[-height:height + 1, -width:width + 1]
    # ellipse mask
    mask = ((x) ** 2 / (height ** 2) + (y) ** 2 / (width ** 2) <= 1)
    mask.dtype = 'uint8'
    return mask

def remove_cover(mask_map,kernel=3):
    mask_map [mask_map>1] = 2
    delet_map = np.zeros_like(mask_map,dtype='uint8')
    delet_map[mask_map==2]=2

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(kernel,kernel))
    delet_map = cv.dilate(delet_map,kernel)
    mask_map[delet_map == 2] = 0
    return mask_map, delet_map

def generate_masks():
    for idx, img_id in enumerate(os.listdir(img_path)):

        dst_mask_path = os.path.join(mask_path, img_id.replace('jpg', 'png'))
        if os.path.exists(dst_mask_path):
            continue
        else:
            ImgInfo = {}
            ImgInfo.update({"img_id": img_id})
            img_ori = os.path.join(img_path, img_id)
            img_ori = Image.open(img_ori)
            w, h = img_ori.size
            print(img_id)
            print(w, h)

            mask_map = np.zeros((h, w),dtype='uint8')
            gt_name = os.path.join(json_path, img_id.split('.')[0] + '.json')
            print(gt_name)

            with open(gt_name) as f:
                ImgInfo = json.load(f)

            centroid_list = []
            wh_list = []
            for id,(w_start, h_start, w_end, h_end) in enumerate(ImgInfo["boxes"],0):
                centroid_list.append([(w_end + w_start) / 2, (h_end + h_start) / 2])
                wh_list.append([max((w_end - w_start) / 2, 3), max((h_end - h_start) / 2, 3)])
            # print(len(centroid_list))
            centroids = np.array(centroid_list.copy(),dtype='int')
            wh        = np.array(wh_list.copy(),dtype='int')
            wh[wh>25] = 25
            human_num = ImgInfo["human_num"]
            for point in centroids:
                point = point[None,:]

                dists = euclidean_dist(point, centroids)
                dists = dists.squeeze()
                id = np.argsort(dists)

                for start, first in enumerate(id, 0):
                    if start > 0 and start < 5:
                        src_point = point.squeeze()
                        dst_point = centroids[first]

                        src_w, src_h = wh[id[0]][0], wh[id[0]][1]
                        dst_w, dst_h = wh[first][0], wh[first][1]

                        count = 0
                        if (src_w + dst_w) - np.abs(src_point[0] - dst_point[0]) > 0 and (src_h + dst_h) - np.abs(src_point[1] - dst_point[1]) > 0:
                            w_reduce = ((src_w + dst_w) - np.abs(src_point[0] - dst_point[0])) / 2
                            h_reduce = ((src_h + dst_h) - np.abs(src_point[1] - dst_point[1])) / 2
                            threshold_w, threshold_h = max(-int(max(src_w - w_reduce, dst_w - w_reduce) / 2.), -60), max(
                                -int(max(src_h - h_reduce, dst_h - h_reduce) / 2.), -60)

                        else:
                            threshold_w, threshold_h = max(-int(max(src_w, dst_w) / 2.), -60), max(-int(max(src_h, dst_h) / 2.), -60)
                        # threshold_w, threshold_h = -5, -5
                        while (src_w + dst_w) - np.abs(src_point[0] - dst_point[0]) > threshold_w and (src_h + dst_h) - np.abs(
                                src_point[1] - dst_point[1]) > threshold_h:

                            if (dst_w * dst_h) > (src_w * src_h):
                                wh[first][0] = max(int(wh[first][0] * 0.9), 2)
                                wh[first][1] = max(int(wh[first][1] * 0.9), 2)
                                dst_w, dst_h = wh[first][0], wh[first][1]
                            else:
                                wh[id[0]][0] = max(int(wh[id[0]][0] * 0.9), 2)
                                wh[id[0]][1] = max(int(wh[id[0]][1] * 0.9), 2)
                                src_w, src_h = wh[id[0]][0], wh[id[0]][1]

                            if human_num >= 3:
                                dst_point_ = centroids[id[start + 1]]
                                dst_w_, dst_h_ = wh[id[start + 1]][0], wh[id[start + 1]][1]
                                if (dst_w_ * dst_h_) > (src_w * src_h) and (dst_w_ * dst_h_) > (dst_w * dst_h):
                                    if (src_w + dst_w_) - np.abs(src_point[0] - dst_point_[0]) > -3 and (src_h + dst_h_) - np.abs(
                                            src_point[1] - dst_point_[1]) > -3:
                                        wh[id[start + 1]][0] = max(int(wh[id[start + 1]][0] * 0.9), 2)
                                        wh[id[start + 1]][1] = max(int(wh[id[start + 1]][1] * 0.9), 2)

                            count += 1
                            if count > 40:
                                break
            for (center_w, center_h), (width, height) in zip(centroids, wh):
                assert (width > 0 and height > 0)

                if (0 < center_w < w) and (0 < center_h < h):
                    h_start = (center_h - height)
                    h_end = (center_h + height )

                    w_start = center_w - width
                    w_end = center_w + width
                    #
                    if h_start < 0:
                        h_start = 0

                    if h_end > h:
                        h_end = h

                    if w_start < 0:
                        w_start = 0

                    if w_end > w:
                        w_end = w

                    if cycle:
                        mask = generate_cycle_mask(height, width)
                        mask_map[h_start:h_end, w_start: w_end] = mask

                    else:
                        mask_map[h_start:h_end, w_start: w_end] = 1

            mask_map = mask_map*255

            cv.imwrite(dst_mask_path, mask_map, [cv.IMWRITE_PNG_BILEVEL, 1])

def generate_masks_with_points():
    max_sigma = 7.5
    max_kernel_size = int(2 * max_sigma)
    max_kernel_width = 2 * max_kernel_size + 1

    file_list = glob.glob(os.path.join(img_path, '*.jpg'))

    print(len(file_list))
    for idx, path in enumerate(file_list):  # 108.jpg is the wrong labeled image
        # print(path)

        img_id = path.split('\\')[-1].split('.')[0]
        img_ori = Image.open(path)
        w, h = img_ori.size
        # print(img_id, w, h)

        mask_map = np.zeros((h, w), dtype='float32')

        gt_name = os.path.join(json_path, img_id.split('.')[0] + '.json')

        # with open(gt_name) as f:
        #     ImgInfo = json.load(f)

        # points = ImgInfo["shapes"]
        loc = json.load(open(gt_name, 'r', encoding="utf-8"))['shapes']
        points = []
        for point in loc:
            tmp_point = point['points']
            points.append(tmp_point[0])

        gt_count = len(points)
        # print(gt_count)
        leafsize = 2048
        if gt_count>0:
            # build kdtree
            tree = T.KDTree(points.copy(), leafsize=leafsize)
            distances, locations = tree.query(points, k=2)  # 查询最近的两个邻居

            for i, pt in enumerate(points):
                if pt[0]>=w or pt[1]>=h:
                    continue
                center_h, center_w = int(pt[1]), int(pt[0])

                radius = 8
                h_start = max(0, center_h - radius)
                h_end = min(1024, center_h + radius )

                w_start = max(0, center_w - radius)
                w_end = min(1024, center_w + radius)

                h2 = (h_end - h_start)/2
                w2 = (w_end - w_start)/2

                mask = generate_cycle_mask(h2, w2)
                # y,x=np.ogrid[0:8,0:8]
                # mask = (x-center_h)**2+(y-center_w)**2<=4**2

                # print(mask.shape,'\n', h_start,h_end, w_start,w_end)
                mask_map[h_start:h_end, w_start:w_end] = mask

        mask_map = mask_map.astype(np.uint8)*255
        # print(os.path.join(mask_path, img_id+'.png'))
        cv.imwrite(os.path.join(mask_path, img_id+'.png'), mask_map, [cv.IMWRITE_PNG_BILEVEL, 1])
        # print(mask_map.sum())
        # break


if __name__ == '__main__':
    generate_masks_with_points()
