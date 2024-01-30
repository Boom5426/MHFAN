import scipy.spatial
from PIL import Image
import scipy.io as io
import scipy
import numpy as np
import h5py
import cv2


def load_data_fidt(img_path, args, train=True):
    ######### mode
    # print(img_path)
    # if train == True:
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'location_map').replace('.png', '.h5')
    # gt_path = img_path.replace('.jpg', '.h5').replace('images', 'our_gt_fidt_map')
    # gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_Gus_map')
    # else:
    #     gt_path = img_path.replace('.jpg', '.h5').replace('512images', 'our_gt_fidt_map')
        # gt_path = img_path.replace('.jpg', '.h5').replace('512images', '512gt_fidt_map')
    # print(gt_path)
    img = Image.open(img_path).convert('RGB')

    while True:
        try:
            gt_file = h5py.File(gt_path)
            k = np.asarray(gt_file['kpoint'])
            fidt_map = np.asarray(gt_file['fidt_map'])
            # fidt_map = np.asarray(gt_file['Guass_map'])
            break
        except OSError:
            print("path is wrong, can not load ", img_path)
            cv2.waitKey(1000)  # Wait a bit

    img = img.copy()
    fidt_map = fidt_map.copy()
    k = k.copy()

    return img, fidt_map, k
