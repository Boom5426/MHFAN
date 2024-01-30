from __future__ import division
import warnings
warnings.filterwarnings('ignore')

from operator import mod

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision
import dataset
import math
from image import *
from utils import *

import logging
import nni
from nni.utils import merge_parameter
from config import return_args, args
import time
from local_eval.eval import location_main
import json

# from Networks.VGG.VGG import VGG16_FPN
from Networks.VGG.VGG_GNN import VGG16_FPN


'''fixed random seed '''
setup_seed(args.seed)

logger = logging.getLogger('mnist_AutoML')

def main(args):
    if args['dataset'] == 'BCData':
        train_file = './npydata/BCData_train.npy'
        test_file = './npydata/BCData_val.npy' 
        gt_location_file = './local_eval/gt_file/BCD_val_gt.txt'
    elif args['dataset'] == 'seg':
        train_file = './npydata/seg_train.npy'
        test_file = './npydata/seg_val.npy'
        gt_location_file = './local_eval/gt_file/seg_val_gt.txt'
    elif args['dataset'] == 'CoNIC':
        train_file = './npydata/CoNIC_train.npy'
        test_file = './npydata/CoNIC_test.npy'
        gt_location_file = './local_eval/gt_file/CoNIC_test_gt.txt'

    with open(train_file, 'rb') as outfile:
        train_list = np.load(outfile).tolist()
    with open(test_file, 'rb') as outfile:
        test_list = np.load(outfile).tolist()

    print(len(train_list))
    print(len(test_list))
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_id']
    # model = get_seg_model()
    model = VGG16_FPN()

    model = model.cuda()

    optimizer = torch.optim.Adam(
        [
            {'params': model.parameters(), 'lr': args['lr']},
        ], lr=args['lr'], weight_decay=args['weight_decay'])

    criterion = nn.MSELoss(size_average=False).cuda()

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1500,], gamma=0.1)

    print("Pre: ",args['pre'])

    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    if args['pre']:
        if os.path.isfile(args['pre']):
            print("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            # model.load_state_dict(checkpoint['state_dict'], strict=False)
            try:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            except KeyError as e:
                print(f"Ignored KeyError: {e}")

            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            print("=> no checkpoint found at '{}'".format(args['pre']))

    torch.set_num_threads(args['workers'])
    # args['best_pred'] = 0
    print(args['best_pred'], args['start_epoch'])

    if args['preload_data'] == True:
        train_data = pre_data(train_list, args, train=True)
        test_data = pre_data(test_list, args, train=False)
    else:
        train_data = train_list
        test_data = test_list
    
    if args['pre']:
        _, _, _, _, _ = Location_validate(test_data, model, args, gt_location_file)
        # exit()
    
    record = {}
    for epoch in range(args['start_epoch'], args['epochs']):
        start = time.time()
        train(train_data, model, criterion, optimizer, epoch, args, scheduler)
        end1 = time.time()

        '''inference '''
        
        # if epoch > 4 and epoch % 5 == 0:
        if epoch >= -1:
            f1m_s, f1m_l, mae, mse, visi = Location_validate(test_data, model, args, gt_location_file)
            
            re = [f1m_s, f1m_l, mae, mse]
            
            F1_m = f1m_s + f1m_l

            is_best = F1_m >= args['best_pred']
            if is_best:
                record['best'] = re
            
            record[str(epoch)] = re
            
            with open('./record.json', 'w') as f:
                f.write(json.dumps(record, indent=2))

            end2 = time.time()
            
            args['best_pred'] = max(F1_m, args['best_pred'])
            # args['best_pred'] = max(F1_m, args['best_pred'])

            print(' * best F1_m:  ', args['best_pred'], args['save_path'])

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args['pre'],
                'state_dict': model.state_dict(),
                'best_prec1': args['best_pred'],
                'optimizer': optimizer.state_dict(),
            }, visi, is_best, args['save_path'])

def pre_data(train_list, args, train):
    print("Pre_load dataset ......")
    print(len(train_list))
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        # print(fname)
        img, fidt_map, kpoint = load_data_fidt(Img_path, args, train)

        if min(fidt_map.shape[0], fidt_map.shape[1]) < 256 and train == True:
            # ignore some small resolution images
            continue
        # print(img.size, fidt_map.shape)
        blob = {}
        blob['img'] = img
        blob['kpoint'] = np.array(kpoint)
        blob['fidt_map'] = fidt_map
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1
    print("load finished ...")

    return data_keys

def train(Pre_data, model, criterion, optimizer, epoch, args, scheduler):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),

                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            batch_size=args['batch_size'],
                            num_workers=args['workers'],
                            args=args),
        batch_size=args['batch_size'], drop_last=False)
    args['lr'] = optimizer.param_groups[0]['lr']
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args['lr']))

    model.train()
    end = time.time()

    for i, (fname, img, fidt_map, kpoint) in enumerate(train_loader):
        data_time.update(time.time() - end)
        img = img.cuda()
        fidt_map = fidt_map.type(torch.FloatTensor).unsqueeze(1).cuda()

        d6 = model(img)

        if d6.shape != fidt_map.shape:
            print("the shape is wrong, please check. Both of prediction and GT should be [B, C, H, W].")
            exit()
        loss = criterion(d6, fidt_map)

        losses.update(loss.item(), img.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
    scheduler.step()

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

    # if not os.path.exists('./local_eval/loc_file'):
    #     os.makedirs('./local_eval/loc_file')

    '''output coordinates'''
    f_loc = open("./local_eval/B_localization.txt", "w+")

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
            d6 = model(img)
            '''return counting and coordinates'''
            _, pred_kpoint, f_loc = LMDS_counting(d6, i + 1, f_loc, args)
            point_map = generate_point_map(pred_kpoint, f_loc, rate=1)
            if args['visual'] == True:
                if not os.path.exists(args['save_path'] + '_box/'):
                    os.makedirs(args['save_path'] + '_box/')
                ori_img, box_img = generate_bounding_boxes(pred_kpoint, fname)
                show_fidt = show_map(d6.data.cpu().numpy())
                gt_show = show_map(fidt_map.data.cpu().numpy())
                res = np.hstack((ori_img, gt_show, show_fidt, point_map, box_img))
                cv2.imwrite(args['save_path'] + '_box/' + fname[0], res)
            # f_loc.close()
            if i % 20 == 0:
                # print('{fname} Gt {gt:.2f} Pred {pred}'.format(fname=fname[0], gt=gt_count, pred=count))
                visi.append(
                    [img.data.cpu().numpy(), d6.data.cpu().numpy(), fidt_map.data.cpu().numpy(),
                    fname])

    f_loc.close()
    f1m_s, f1m_l, mae, mse = location_main("./local_eval/B_localization.txt", gt_location_file)

    return f1m_s, f1m_l, mae, mse, visi

def LMDS_counting(input, w_fname, f_loc, args):
    input_max = torch.max(input).item()

    ''' find local maxima'''
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

    point_map = np.zeros((int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8") + 255  # 22
    # count = len(pred_coor[0])
    coord_list = []
    for i in range(0, len(pred_coor[0])):
        h = int(pred_coor[0][i] * rate)
        w = int(pred_coor[1][i] * rate)
        coord_list.append([w, h])
        cv2.circle(point_map, (w, h), 2, (0, 0, 0), -1)

    for data in coord_list:
        f_loc.write('{} {} '.format(math.floor(data[0]), math.floor(data[1])))
    f_loc.write('\n')
    # f_loc.close()

    return point_map

def generate_bounding_boxes(kpoint, fname):
    '''change the data path'''
    Img_data = cv2.imread(
        './data/UW/fold2/images/' + fname[0])
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
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))
    print(params)

    main(params)
