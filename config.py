import argparse

# from sqlalchemy import true 

parser = argparse.ArgumentParser(description='FIDTM')

# ['BCData', 'SHB', 'CoNIC', 'SHA', 'QNRF']
parser.add_argument('--dataset', type=str, default='CoNIC',
                    help='choice train dataset')
parser.add_argument('--save_path', type=str, default='./save_file/CoNIC/VGG_GNN',
                    help='save checkpoint directory')
parser.add_argument('--gpu_id', type=str, default='0',
                    help='gpu id')
parser.add_argument('--batch_size', type=int, default=12,
                    help='input batch size for training')
parser.add_argument('--crop_size', type=int, default=512,  # 256 for A, 512 for others 
                    help='crop size for training')
parser.add_argument('--visual', type=bool, default=False,
                    help='visual for bounding box. ')
parser.add_argument('--preload_data', type=bool, default=False,  # 一次性加载到内存中。以内存换时间
                    help='preload data. ')
parser.add_argument('--lr', type=float, default= 1e-4,
                    help='learning rate')
parser.add_argument('--pre', type=str, default=None,
                    help='pre-trained model directory')

# parser.add_argument('--pre', type=str, default='./save_file/CoNIC/VGG_GNN/model_best.pth',
#                     help='pre-trained model directory')


parser.add_argument('--workers', type=int, default=16,                      # 降内存
                    help='load data workers')
parser.add_argument('--print_freq', type=int, default=200,
                    help='print frequency')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='start epoch for training')
parser.add_argument('--epochs', type=int, default=1000,                      # 降内存
                    help='end epoch for training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--best_pred', type=int, default=0,
                    help='best pred')
parser.add_argument('--weight_decay', type=float, default=5 * 1e-4,
                    help='weight decay')

'''video demo'''
parser.add_argument('--video_path', type=str, default=None,
                    help='input video path ')

args = parser.parse_args()
return_args = parser.parse_args()
