from  torchvision import models
import sys
import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch_geometric.nn import TransformerConv, LayerNorm, GATConv, GCNConv
import math
from torch.nn.modules.module import Module
from torch import FloatTensor
from torch.nn.parameter import Parameter
# sys.path.append('/home/zcy/CrowdCounting/IIM-main/')
# from misc.utils import *
BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
def initialize_weights(models):
    for model in models:
        real_init_weights(model)

def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, nn.Conv2d):    
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m,nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print( m )

class Cos_Adj_topK(Module):
    def __init__(self, in_features=1024, out_features=1024, top_k=9):
        super(Cos_Adj_topK, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.top_k = top_k

        self.weight0 = Parameter(FloatTensor(in_features, out_features))
        self.weight1 = Parameter(FloatTensor(in_features, out_features))

        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight0)
        nn.init.xavier_uniform_(self.weight1)

    def forward(self, input):
        seq_len = torch.sum(torch.abs(input), 1)
        soft = nn.Softmax(1)
        theta = torch.matmul(input, self.weight0)
        # print(theta.shape) # [2, 512, 32]
        phi = torch.matmul(input, self.weight1)
        # print(phi.shape) # [2, 512, 32]
        phi2 = phi.permute(1, 0)
        sim_graph = torch.matmul(theta, phi2)

        theta_norm = torch.norm(theta, p=2, dim=1, keepdim=True)  # B*T*1
        # print(theta_norm.shape) # [2, 576, 1]
        phi_norm = torch.norm(phi, p=2, dim=1, keepdim=True)  # B*T*1
        # print(phi_norm.shape) # [2, 576, 1]
        x_norm_x = theta_norm.matmul(phi_norm.permute(1, 0))
        # print(x_norm_x.shape) # [2, 576, 576]
        sim_graph = sim_graph / (x_norm_x + 1e-20)

        # output = torch.zeros_like(sim_graph)
        # # .cpu().detach().numpy()
        # for i in range(sim_graph.shape[0]):
        #     # print(sim_graph[i][:10])
        #     sorted_indices = np.argsort(sim_graph[i].cpu().detach().numpy())
        #     # print(sorted_indices)
        #     selected_indices = sorted_indices[:self.top_k]
        #     # output[i][selected_indices] = 1
        #     output[i][selected_indices] = sim_graph[i][selected_indices]
        return sim_graph

def create_adjacency_matrix(tensor, Top_K=3):
    # Assuming tensor is a 3D tensor [w, h, c]
    c, w, h = tensor.shape

    # Extract the [1, C] part as nodes
    nodes = tensor[:, :].view(w * h, c)

    # Compute spatial coordinates
    y, x = torch.meshgrid(torch.arange(w), torch.arange(h))
    coords = torch.stack([y.flatten(), x.flatten()], dim=1)
    # print('coords:', coords.shape, coords) #对应图中每一点的坐标

    # Compute pairwise distances between spatial coordinates
    spatial_distances = torch.cdist(coords.float(), coords.float()).to('cuda:0')
    # print('spatial_distances:', spatial_distances.shape, spatial_distances[0])  # spatial_distances[0]表示当前第一点对其他所有点的坐标距离

    # Compute pairwise distances between node features
    feature_distances = torch.cdist(nodes, nodes)

    # Norm: 整个矩阵norm，而不是行或者列
    # 对整个特征距离矩阵进行归一化
    feature_distances = feature_distances / feature_distances.norm()
    # 对整个空间距离矩阵进行归一化
    spatial_distances = spatial_distances / spatial_distances.norm()

    # 对 feature_distances 进行排序并取最近的前 10 个节点
    _, sorted_feature_indices = torch.topk(feature_distances, k=feature_distances.size(1), largest=False)
    adjacency_matrix_feature = torch.zeros_like(feature_distances, dtype=torch.int)
    for i in range(adjacency_matrix_feature.size(0)):
        adjacency_matrix_feature[i, sorted_feature_indices[i, :Top_K]] = 1

    # 对 spatial_distances 进行排序并取最近的前 10 个节点
    _, sorted_spatial_indices = torch.topk(spatial_distances, k=spatial_distances.size(1), largest=False)
    adjacency_matrix_spatial = torch.zeros_like(spatial_distances, dtype=torch.int)
    for i in range(adjacency_matrix_spatial.size(0)):
        adjacency_matrix_spatial[i, sorted_spatial_indices[i, :Top_K]] = 1

    # adjacency_matrix = adjacency_matrix_feature
    adjacency_matrix = adjacency_matrix_feature & adjacency_matrix_spatial
    # 聚合你附近和你相似的
    # print('adjacency_matrix:', adjacency_matrix.shape, adjacency_matrix[0])

    return adjacency_matrix

def crop(d, g):
    g_h, g_w = g.size()[2:4]
    d_h, d_w = d.size()[2:4]
    d1 = d[:, :, int(math.floor((d_h - g_h) / 2.0)):int(math.floor((d_h - g_h) / 2.0)) + g_h,
         int(math.floor((d_w - g_w) / 2.0)):int(math.floor((d_w - g_w) / 2.0)) + g_w]
    return d1

mode = 'Vgg_bn'
class VGG16_FPN(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16_FPN, self).__init__()
        vgg = models.vgg16_bn(pretrained=pretrained)
        features = list(vgg.features.children())
        if  mode == 'Vgg_bn':
            self.layer1 = nn.Sequential(*features[0:23])
            self.layer2 = nn.Sequential(*features[23:33])
            self.layer3 = nn.Sequential(*features[33:43])

        in_dim3 = 512 # 通道数量固定，隐层随便，太大会爆炸，就和通道保持一致
        # self.Cos_Adj3 = Cos_Adj_topK(in_dim3, in_dim3)
        # graphTransformer Conv
        self.GNNconv3_1 = TransformerConv(in_dim3, in_dim3)
        self.GNNconv3_2 = TransformerConv(in_dim3, in_dim3)
        self.norm = nn.LayerNorm(in_dim3)
        self.sigmod = nn.Sigmoid()
        
        self.de_pred = nn.Sequential(
            nn.Conv2d(in_channels=1280,out_channels=640,kernel_size=1,stride=1,padding=0),
            BatchNorm2d(640, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(640, 64, 4, stride=2, padding=1, output_padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, output_padding=0, bias=True),
        )
        initialize_weights(self.de_pred)

    def forward(self, x):
        gt = x.clone()

        f = []
        x = self.layer1(x)        
        x0_h, x0_w = x.size(2), x.size(3)

        f.append(x)
        x = self.layer2(x)
        f.append(x)
        x = self.layer3(x)
        f.append(x)

        # f = self.neck(f)

        # GNN        
        x3 = f[2]

        output_list = []
        # print(x3.shape)
        for x3_in in x3:
            # get Dist Adj
            Dist_adj = create_adjacency_matrix(x3_in)
            # print(Dist_adj.shape, Dist_adj[0][:20])   # 64, 64
            x3_in = x3_in.view(x3_in.shape[2] * x3_in.shape[1], x3_in.shape[0])
            
            edge_index_temp = sp.coo_matrix(Dist_adj.cpu().detach().numpy())
            values = edge_index_temp.data  # 边上对应权重值weight
            indices = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
            edge_index_A = torch.LongTensor(np.array(indices)).cuda()
            # print(edge_index_A.shape, edge_index_A[0][:10])            # [2, 4096] , 缩减边后锐减：[2, 2240])
            # x and edges
            h1 = self.GNNconv3_1(x3_in, edge_index_A)
            h1 = F.dropout(self.norm(torch.relu(h1)), 0.5)
            h2 = self.GNNconv3_2(h1, edge_index_A)

            # print(h2.shape)
            output_list.append(h2.unsqueeze(0))

        output = torch.cat(output_list, dim=0)
        x2_weight = output.view(f[2].shape[0], f[2].shape[1], f[2].shape[2], -1)

        x2 = self.sigmod(x2_weight) * x3
        # x2 = x3 + x2_weight
        # print(x2.shape)

        x0 = F.upsample(f[0], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x1 = F.upsample(f[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.upsample(x2, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        
        f = torch.cat([x0, x1, x2], 1)

        x = self.de_pred(f)

        x = crop(x, gt)
        # print(x.shape)

        return x

class FPN(nn.Module):
    """
    Feature Pyramid Network.

    This is an implementation of - Feature Pyramid Networks for Object
    Detection (https://arxiv.org/abs/1612.03144)

    Args:
        in_channels (List[int]):
            number of input channels per scale

        out_channels (int):
            number of output channels (used at each scale)

        num_outs (int):
            number of output scales

        start_level (int):
            index of the first input scale to use as an output scale

        end_level (int, default=-1):
            index of the last input scale to use as an output scale

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print('outputs[{}].shape = {!r}'.format(i, outputs[i].shape))
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,in_channels,out_channels,num_outs,start_level=0,end_level=-1,
                extra_convs_on_inputs=True,bn=True):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = Conv2d( in_channels[i], out_channels,1,bn=bn, bias=not bn,same_padding=True)

            fpn_conv = Conv2d( out_channels, out_channels,3,bn=bn, bias=not bn,same_padding=True)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        self.init_weights()
    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)


    def forward(self, inputs):

        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [ self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels) ]


        return tuple(outs)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=True, bias=True):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)

        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=False)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

if __name__ == "__main__":
    net = VGG16_FPN(pretrained=False).cuda()
    # print(net)
    # summary(net,(3,64 ,64 ),batch_size=4)
    out = net(torch.rand(1,3,512,512).cuda())
    print(out.shape)
