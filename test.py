from __future__ import division
import warnings
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import dataset
import math
from image import *
from utils import *
import logging
import nni
from nni.utils import merge_parameter
from config import return_args, args
warnings.filterwarnings('ignore')
setup_seed(args.seed)
logger = logging.getLogger('mnist_AutoML')
from local_eval.eval import location_main

# from Networks.CMU import cmunext_l
# from Networks.UNet.U_net import U_Net
# from Networks.HR_Net.Cat_GCN123_hrnet import get_seg_model
# from Networks.UNet.U_net_ghost import U_Net, U_Net_best
# from Networks.UNet.UNet_GCN import U_Net_HGN, U_Net_GPA
# from Networks.CMUNet import CMUNet
# from Networks.VGG import VGG16_FPN
from Networks.HR_Net.seg_hrnet import get_seg_model

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(args):
    # test_file = './npydata/PSU_test.npy'
    # test_file = './npydata/BCData_test.npy'
    # gt_location_file = './local_eval/gt_file/BCD_test_gt.txt'
    
    test_file = './npydata/ShanghaiB_test.npy'
    gt_location_file = './local_eval/gt_file/B_gt.txt'
    # test_file = './npydata/UW_test.npy' 

    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_id']

    # model = UNext(1)
    # model = U_Net_base()
    model = get_seg_model()
    args['pre'] = './save_file/SHB/HRNet_up/model_best.pth'

    # model = get_seg_model()
    # args['pre'] = '/home/jlt/Crowd_counting/cell_Location/save_file/BCData/model_best.pth'

    # model = U_Net_HGN()
    # args['pre'] = './save_file/BCData/mlp1/model_best.pth'

    # model = U_Net_GNN()
    # args['pre'] = './save_file/BCData/UNet_GNN/model_best.pth'

    # model = CSRNet()
    # args['pre'] = './save_file/BCD/CSRNet/model_best.pth'

    # model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args['lr']},])

    print(args['pre'])

    args['save_path'] = './save_file/visual/'
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    if args['pre']:
        if os.path.isfile(args['pre']):
            print("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            print("=> no checkpoint found at '{}'".format(args['pre']))

    torch.set_num_threads(args['workers'])
    print(args['best_pred'], args['start_epoch'])

    if args['preload_data'] == True:
        test_data = pre_data(val_list, args, train=False)
    else:
        test_data = val_list

    '''inference '''
    # a = time_count(test_data, model, args, gt_location_file)
    prec1, visi = Location_validate(test_data, model, args, gt_location_file)

    # is_best = prec1 < args['best_pred']
    # args['best_pred'] = min(prec1, args['best_pred'])

    # print('\nThe visualizations are provided in ', args['save_path'])
    # save_checkpoint({
    #     'arch': args['pre'],
    #     'state_dict': model.state_dict(),
    #     'best_prec1': args['best_pred'],
    #     'optimizer': optimizer.state_dict(),
    # }, visi, is_best, args['save_path'])

def pre_data(train_list, args, train):
    print("Pre_load dataset ......")
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        # print(fname)
        img, fidt_map, kpoint = load_data_fidt(Img_path, args, train)

        blob = {}
        blob['img'] = img
        blob['kpoint'] = np.array(kpoint)
        blob['fidt_map'] = fidt_map
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1

    return data_keys

def Location_validate(Pre_data, model, args, gt_location_file):
    print('begin test')   
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),

                            ]),
                            args=args, train=False),
        batch_size=1)

    model.eval()

    mae = 0.0
    mse = 0.0
    visi = []
    index = 0

    if not os.path.exists('./local_eval/loc_file'):
        os.makedirs('./local_eval/loc_file')

    '''output coordinates'''
    f_loc = open("./local_eval/A_localization.txt", "w+")

    for i, (fname, img, fidt_map, kpoint) in enumerate(test_loader):
        count = 0
        img = img.cuda()

        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(fidt_map.shape) == 5:
            fidt_map = fidt_map.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if len(fidt_map.shape) == 3:
            fidt_map = fidt_map.unsqueeze(0)

        with torch.no_grad():
            # print(img.shape)
            # 测试推理时间
            d6 = model(img)

            '''return counting and coordinates'''
            _, pred_kpoint, f_loc = LMDS_counting(d6, i + 1, f_loc, args)
            _ = generate_point_map(pred_kpoint, f_loc, rate=1)
            # f_loc.close()
            if i % 20 == 0:
                # print('{fname} Gt {gt:.2f} Pred {pred}'.format(fname=fname[0], gt=gt_count, pred=count))
                visi.append(
                    [img.data.cpu().numpy(), d6.data.cpu().numpy(), fidt_map.data.cpu().numpy(),
                    fname])

    f_loc.close()
    f1m_s, f1m_l, mae, mse = location_main("./local_eval/A_localization.txt", gt_location_file)

    return f1m_s + f1m_l, visi

import time
def time_count(Pre_data, model, args, gt_location_file):
    print('begin test')   
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),

                            ]),
                            args=args, train=False),
        batch_size=1)

    model.eval()

    number = 0
    
    for i, (fname, img, fidt_map, kpoint) in enumerate(test_loader):
        number += 1
        img = img.cuda()

        with torch.no_grad():
            # 测试推理时间
            if number == 2:
                begin = time.time()
            d6 = model(img)
                
    end = time.time()
    print("time:", (end - begin), number, (end - begin)/400)

    return

def libo_counting(input, w_fname, f_loc, args):
    # input_max = torch.max(input).item()  # 获取了一个全局的最大值

    ''' find local maxima'''
    keep = nn.functional.max_pool2d(input, (11, 11), stride=1, padding=5)
    keep = nn.functional.max_pool2d(keep, (11, 11), stride=1, padding=5)

    keep = (keep == input).float()
    input = keep * input

    '''set the pixel value of local maxima as 1 for counting'''
    # print(input_max)
    # input[input < 0.1 ] = 0            ######### 细胞核检测的效果
    # input[input > 0] = 1
    # input[input < 50.0 / 255.0 * input_max] = 0            ######### 细胞核检测的效果:100
    input[input < 0.06] = 0      
    input[input > 0] = 1

    # ''' negative sample'''
    # if input_max < 0.1:
    #     input = input * 0

    count = int(torch.sum(input).item())
    kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()

    f_loc.write('{} {} '.format(w_fname, count))
    return count, kpoint, f_loc

def LMDS_counting(input, w_fname, f_loc, args):
    input_max = torch.max(input).item()

    ''' find local maxima'''
    if args['dataset'] == 'QNRF' :
        input = nn.functional.avg_pool2d(input, (3, 3), stride=1, padding=1)
        keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    else:
        keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input

    '''set the pixel valur of local maxima as 1 for counting'''
    input[input < 100.0 / 255.0 * input_max] = 0
    input[input > 0] = 1

    ''' negative sample'''
    if input_max < 0.1:
        input = input * 0

    count = int(torch.sum(input).item())

    kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()

    f_loc.write('{} {} '.format(w_fname, count))
    return count, kpoint, f_loc


def generate_point_map(kpoint, f_loc, rate=1):
    '''obtain the location coordinates'''
    pred_coor = np.nonzero(kpoint)
    # print('###', len(pred_coor[0]))

    point_map = np.zeros((int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8") + 255  # 22
    # count = len(pred_coor[0])
    coord_list = []
    for i in range(0, len(pred_coor[0])): # 1024
        h = int(pred_coor[0][i] * rate)
        w = int(pred_coor[1][i] * rate)
        coord_list.append([w, h])
        # 在point_map上绘图，半径为2的圆
        cv2.circle(point_map, (w, h), 2, (0, 0, 0), -1)

    for data in coord_list:
        f_loc.write('{} {} '.format(math.floor(data[0]), math.floor(data[1])))
    f_loc.write('\n')

    return point_map

def generate_bounding_boxes(kpoint, fname):
    '''change the data path'''
    path = os.path.join('./data/cell_dataset/x40S/validation/images', fname[0])
    print(path)
    Img_data = cv2.imread(path)
    ori_Img_data = Img_data.copy()

    '''generate sigma'''
    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)

    distances, locations = tree.query(pts, k=4)
    for index, pt in enumerate(pts):
        pt2d = np.zeros(kpoint.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if np.sum(kpoint) > 1:
            sigma = (distances[index][1] + distances[index][2] + distances[index][3]) * 0.1
        else:
            sigma = np.average(np.array(kpoint.shape)) / 2. / 2.  # case: 1 point
        sigma = min(sigma, min(Img_data.shape[0], Img_data.shape[1]) * 0.05)

        if sigma < 6:
            t = 2
        else:
            t = 2
        Img_data = cv2.rectangle(Img_data, (int(pt[0] - sigma), int(pt[1] - sigma)),
                                 (int(pt[0] + sigma), int(pt[1] + sigma)), (0, 255, 0), t)

    return ori_Img_data, Img_data

def show_map(input):
    input[input < 0] = 0
    input = input[0][0]
    fidt_map1 = input
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2)
    return fidt_map1

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))
    print(params)

    main(params)
