from  torchvision import models
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import fvcore.nn.weight_init as weight_init
import math
from torch.nn.modules.module import Module
from torch import FloatTensor
from torch.nn.parameter import Parameter
import numpy as np

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

mode = 'Vgg_bn'
class VGG16_FPN(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16_FPN, self).__init__()
        if mode == 'Vgg_bn':
            vgg = models.vgg16_bn(pretrained=pretrained)
        features = list(vgg.features.children())
        if  mode == 'Vgg_bn':
            self.layer1 = nn.Sequential(*features[0:23])
            self.layer2 = nn.Sequential(*features[23:33])
            self.layer3 = nn.Sequential(*features[33:43])
            self.layer4 = nn.Sequential(*features[43:])

        # add
        self.hyperAdj0 = SimilarityAdj(16384, 256)
        self.hyperAdj1 = SimilarityAdj(4096, 512)
        self.hyperAdj2 = SimilarityAdj(1024, 512)
        self.hyperAdj3 = SimilarityAdj(256, 512)
        
        self.att1 = SAF(channels=1024+512+256)
        self.att2 = SAF(channels=1024+512)
        self.att3 = SAF(channels=512+512)

        self.sigmod3 = nn.Sigmoid()
        
        self.de_pred = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1792,128,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,1,kernel_size=3,stride=1,padding=1,bias=True)
        )
        
        initialize_weights(self.de_pred)

    def forward(self, x):
        f = []
        x = self.layer1(x)
        # print(x.shape)
        f.append(x)
        x = self.layer2(x)
        # print(x.shape)
        f.append(x)
        x = self.layer3(x)
        # print(x.shape)
        f.append(x)
        x = self.layer4(x)
        # print(x.shape)
        f.append(x)

        # for i in range(len(f)):
        #     print(f[i].shape)
        # feature =torch.cat([f[0],  F.interpolate(f[1],scale_factor=2),F.interpolate(f[2],scale_factor=4), F.interpolate(f[3],scale_factor=8)], dim=1)
        
        # add hyper, print(feature.shape)
        x0_h, x0_w = f[0].size(2), f[0].size(3)
        x0, x1, x2, x3 = f[0], f[1], f[2], f[3]
        
        x0_hgn = x0.view(x0.shape[0], x0.shape[1], x0.shape[2] * x0.shape[3])
        x1_hgn = x1.view(x1.shape[0], x1.shape[1], x1.shape[2] * x1.shape[3])
        x2_hgn = x2.view(x2.shape[0], x2.shape[1], x2.shape[2] * x2.shape[3])
        x3_hgn = x3.view(x3.shape[0], x3.shape[1], x3.shape[2] * x3.shape[3])
        # print(x0_hgn.shape, x1_hgn.shape, x2_hgn.shape, x3_hgn.shape)
        # [1, 96, 2304]) torch.Size([1, 192, 576]) torch.Size([1, 384, 144])
        
        # add hypergraph
        x0_hgn = self.hyperAdj0(x0_hgn)
        # print(x1_hgn.shape)
        x0_hgn = x0_hgn.view(x0_hgn.shape[0], x0_hgn.shape[1], x0_h, x0_w)
        
        x1_hgn = self.hyperAdj1(x1_hgn)
        # print(x1_hgn.shape)
        x1_hgn = x1_hgn.view(x1_hgn.shape[0], x1_hgn.shape[1], x0_h//2, x0_w//2)

        x2_hgn = self.hyperAdj2(x2_hgn)
        # print(x2.shape)
        x2_hgn = x2_hgn.view(x2_hgn.shape[0], x2_hgn.shape[1], x0_h//4, x0_w//4)

        x3_hgn = self.hyperAdj3(x3_hgn)
        # print(x3.shape)
        x3_hgn = x3_hgn.view(x3_hgn.shape[0], x3_hgn.shape[1], x0_h//8, x0_w//8)
        
        x0 = self.sigmod3(x0_hgn) * x0
        x1 = self.sigmod3(x1_hgn) * x1
        x2 = self.sigmod3(x2_hgn) * x2
        x3 = self.sigmod3(x3_hgn) * x3
            
        x2 = self.att3(x3,x2)
        # print(x2.shape)
        x1 = self.att2(x2,x1)
        # print(x1.shape)
        x0 = self.att1(x1,x0)
        # print(x0.shape)
                
        x = self.de_pred(x0)
        return x


# class VGG16_FPN(nn.Module):
#     def __init__(self, pretrained=True):
#         super(VGG16_FPN, self).__init__()
#         if mode == 'Vgg_bn':
#             vgg = models.vgg16_bn(pretrained=pretrained)
#         features = list(vgg.features.children())
#         if  mode == 'Vgg_bn':
#             self.layer1 = nn.Sequential(*features[0:23])
#             self.layer2 = nn.Sequential(*features[23:33])
#             self.layer3 = nn.Sequential(*features[33:43])

#         in_channels = [256,512,512]
#         self.neck =  FPN(in_channels,256,len(in_channels))

#         self.de_pred = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(768,128,kernel_size=3,stride=1,padding=1,bias=True),
# 		    nn.BatchNorm2d(128),
# 			nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128,1,kernel_size=3,stride=1,padding=1,bias=True)
#         )
        
#         self.hyper = SimilarityAdj(16384, 768)
#         self.sigmod = nn.Sigmoid()
        
#         initialize_weights(self.de_pred)

#     def forward(self, x):
#         f = []
#         x = self.layer1(x)
#         f.append(x)
#         x = self.layer2(x)
#         f.append(x)
#         x = self.layer3(x)
#         f.append(x)

#         f = self.neck(f)
#         feature =torch.cat([f[0],  F.interpolate(f[1],scale_factor=2),F.interpolate(f[2],scale_factor=4)], dim=1)
#         # print(feature.shape)
#         x0_h, x0_w = feature.size(2), feature.size(3)
        
#         feature_hgn = feature.view(feature.shape[0], feature.shape[1], feature.shape[2] * feature.shape[3])
#         # print(feature_hgn.shape)

#         # add hypergraph
#         feature_hgn = self.hyper(feature_hgn)
#         # print(x1_hgn.shape)
#         feature = feature.view(feature.shape[0], feature.shape[1], x0_h, x0_w)
        
#         feature = self.sigmod(feature) * feature
#         # print(feature.shape)
        
#         x = self.de_pred(feature)
        
#         return x

class SAF(nn.Module):
    def __init__(self, channels=720, r=4):
        super(SAF, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, high_stage, low_stage):
        # print(x_c, x1_c, x2_c, x3_c)  # 48 96 192 384
        h, w = low_stage.size(2), low_stage.size(3)
        # 为了特征对齐，所以align_corners为TRUE，密集更友好
        high_stage = F.interpolate(input=high_stage, size=(h, w), mode='bilinear', align_corners=True)
        
        xa = torch.cat((low_stage, high_stage), 1)
        # print(xa.shape)  # [1, 720, 128, 128])
        xl = self.local_att(xa)
        
        wei = self.sigmoid(xl)
        out = xa * wei
        return out

class HGNN(nn.Module):
    def __init__(self, in_ch, n_hid, n_class, dropout=0.5, momentum=0.1):
        super(HGNN, self).__init__()
        self.dropout = dropout
        # self.batch_normalzation1 = nn.BatchNorm1d(in_ch, momentum=momentum)
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        # print(x.shape)
        # x = self.batch_normalzation1(x)
        x = F.relu(self.hgc1(x, G))
        # print(x.shape)
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        # print(x.shape)
        return x

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    # 计算内积，行求和， 成 1 * C
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T  # C * C
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat) # 开方
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat

def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G

def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G

def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=False, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    # print(dis_mat.shape)
    n_obj = dis_mat.shape[0] # 闁跨喐鏋婚幏鐑芥晸閺傘倖瀚归柨鐔诲Ν绾板瀚归柨鐔告灮閹风兘鏁撻弬銈嗗N_object
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj): # 闁跨喐鏋婚幏鐑芥晸閺傘倖瀚瑰В蹇庣闁跨喐鏋婚幏鐑芥晸閼哄倻顣幏锟�
        dis_mat[center_idx, center_idx] = 0 # 闁跨喓娈曢弬銈嗗闁跨喐鏋婚幏铚傝礋0
        dis_vec = dis_mat[center_idx] # 闁跨喐鏋婚幏宄板闁跨喕濡喊澶嬪闁跨喐鏋婚幏鐑芥晸閺傘倖瀚规惔鏃堟晸閺傘倖瀚归柨鐔告灮閹凤拷
        # print(dis_vec.shape)
        res_vec = list(reversed(np.argsort(dis_vec)))
        nearest_idx = np.array(res_vec).squeeze()
        avg_dis = np.average(dis_vec) # 閸欐牠鏁撻弬銈嗗閸婏拷
        # print("***\n", avg_dis)
        # any闁跨喓娈曢幘鍛闁跨喐鏋婚幏鐑芥晸閺傘倖瀚归柨鐔告灮閹风兘鏁撻弬銈嗗閸忓啴鏁撻弬銈嗗闁跨喐鏋婚幏鐑芥晸閺傘倖瀚归柨鐔告灮閹风兘鏁撴慨鎰剁礉闁跨喐鏋婚幏鐑芥晸閺傘倖瀚筎rue闁跨喐瑙︽潻鏂剧串閹风ǖrue
        if not np.any(nearest_idx[:k_neig] == center_idx):
            # 闁跨喐鏋婚幏鐑芥晸閺傘倖瀚归柨鐔活潡閿燂拷10闁跨喐鏋婚幏宄板帗闁跨喐鏋婚幏鐑芥晸閺傘倖瀚归柨鐔活潡閸氾缚绗夌敮顔藉闁跨喐鏋婚幏鐑芥晸閺傘倖瀚归崜宥夋晸閼哄倻鍋ｉ敍宀勬晸閺傘倖瀚归柨鐔告灮閹风兘鏁撻弬銈嗗闁跨喖鍙洪弬銈嗗
            nearest_idx[k_neig - 1] = center_idx

        # print(nearest_idx[:k_neig])
        for node_idx in nearest_idx[:k_neig]:
            if is_probH: # True, 閸欐牠鏁撻弬銈嗗闁跨喓绮搁敐蹇斿闁跨喐鏋婚幏鐑芥晸閺傘倖瀚归柨鐔告灮閹疯渹绔撮柨鐔告灮閹风兘鏁撻弬銈嗗绾噣鏁撴笟銉ь劜閹风兘鏁撻弬銈嗗
                H[node_idx, center_idx] = np.exp(-dis_vec[node_idx] ** 2 / (m_prob * avg_dis) ** 2)
                # print(H[node_idx, center_idx])
            else:
                # print("#")
                H[node_idx, center_idx] = 1.0
        # print(H)
    return H

class SimilarityAdj(Module):
    def __init__(self, in_features, out_features):
        super(SimilarityAdj, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight0 = Parameter(FloatTensor(in_features, out_features))
        self.weight1 = Parameter(FloatTensor(in_features, out_features))
        
        self.gc1_0 = HGNN(in_features, out_features, out_features)
        self.gc1_1 = HGNN(out_features, out_features, in_features)
        self.gc2_0 = HGNN(in_features, out_features, out_features)
        self.gc2_1 = HGNN(out_features, out_features, in_features)
        self.gc3_0 = HGNN(in_features, out_features, out_features)
        self.gc3_1 = HGNN(out_features, out_features, in_features)
        
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / sqrt(self.weight0.size(1))
        nn.init.xavier_uniform_(self.weight0)
        nn.init.xavier_uniform_(self.weight1)

    def forward(self, input):
        seq_len = torch.sum(torch.max(torch.abs(input), dim=2)[0]>0, 1)
        # To support batch operations
        soft = nn.Softmax(1)
        # print(input.shape) # [1, 384, 512]
        theta = torch.matmul(input, self.weight0)
        # print(theta.shape) # [2, 512, 32]
        phi = torch.matmul(input, self.weight1)
        # print(phi.shape) # [2, 512, 32]
        phi2 = phi.permute(0, 2, 1)
        sim_graph = torch.matmul(theta, phi2)
        # print(sim_graph.shape) # [2, 576, 576]

        theta_norm = torch.norm(theta, p=2, dim=2, keepdim=True)  # B*T*1
        # print(theta_norm.shape) # [2, 576, 1]
        phi_norm = torch.norm(phi, p=2, dim=2, keepdim=True)  # B*T*1
        # print(phi_norm.shape) # [2, 576, 1]
        x_norm_x = theta_norm.matmul(phi_norm.permute(0, 2, 1))
        # print(x_norm_x.shape) # [2, 576, 576]
        sim_graph = sim_graph / (x_norm_x + 1e-20)
        
        # print(sim_graph.shape)
        output = torch.zeros_like(sim_graph)
        for i in range(len(seq_len)):
            tmp = sim_graph[i, :seq_len[i], :seq_len[i]]
            adj2 = tmp
            # print(adj2.shape)
            # adj2 = F.threshold(adj2, 0.7, 0)
            adj2 = soft(adj2)
            # print(adj2.shape)
            output[i, :seq_len[i], :seq_len[i]] = adj2
       
        # distance
        input_dis = input.cpu().detach().numpy()
        dis_mats = []
        for i in range(len(input_dis)):
            dis_mat = Eu_dis(input_dis[i])
            dis_mats.append(dis_mat)
        
        dis_mats = torch.Tensor(np.array(dis_mats))
        # print(sim_graph.shape,  dis_mats.shape)
        output1 = torch.zeros_like(dis_mats)
        for i in range(len(seq_len)):
            tmp = dis_mats[i, :seq_len[i], :seq_len[i]]
            adj2 = tmp
            # print(adj2.shape)
            # adj2 = F.threshold(adj2, 0.7, 0)
            adj2 = soft(adj2)
            # print(adj2.shape)
            output1[i, :seq_len[i], :seq_len[i]] = adj2
        # print(output.shape) # [2, 512, 512]
        
        dis_mats = np.array(output1)
        sim_mats = np.array(output.cpu().detach().numpy())

        # print(sim_graph.shape,  dis_mats.shape)
        # 鐩稿叧鍏崇郴鐭╅樀
        sim_graph = dis_mats + sim_mats * 0.5
        
        H3_all, H9_all, H15_all = [], [], []
        for i in range(len(sim_graph)):
            # print(sim_graph[i].shape)
            # add multi scale
            H3 = construct_H_with_KNN_from_distance(sim_graph[i], 3, False, 1)
            H9 = construct_H_with_KNN_from_distance(sim_graph[i], 9, False, 1)
            H15 = construct_H_with_KNN_from_distance(sim_graph[i],15, False, 1)
            # print(H[0])
            H3_all.append(H3)
            H9_all.append(H9)
            H15_all.append(H15)
        
        # print("#################",(np.array(H9_all)).shape) # (2, 3, 144, 144)
        # hypergraph = np.stack((H3_all, H9_all, H15_all), axis=0)
        # print(hypergraph.shape) # [2,576,576]
        G3 = generate_G_from_H(H3_all, variable_weight=False)
        G9 = generate_G_from_H(H9_all, variable_weight=False)
        G15 = generate_G_from_H(H15_all, variable_weight=False)
        for i in range(len(G3)):
            G3[i] = G3[i].A
        G3 = torch.Tensor(G3).cuda()
        for i in range(len(G9)):
            G9[i] = G9[i].A
        G9 = torch.Tensor(G9).cuda()
        for i in range(len(G15)):
            G15[i] = G15[i].A
        G15 = torch.Tensor(G15).cuda()
        # print(G.shape) 
        
        # print(adj1.shape)
        x1 = torch.relu(self.gc1_0(input, G3))
        x1 = torch.relu(self.gc1_1(x1, G3))
        # print(x1.shape)
        
        x2 = torch.relu(self.gc2_0(input, G9))
        x2 = torch.relu(self.gc2_1(x2, G9)) 

        x3 = torch.relu(self.gc3_0(input, G15))
        x3 = torch.relu(self.gc3_1(x3, G15))

        x123 = x1 + x2 + x3

        return x123

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


if __name__ == "__main__":
    net = VGG16_FPN(pretrained=False).cuda()
    # print(net)
    # summary(net,(3,64 ,64 ),batch_size=4)
    out = net(torch.rand(1,3,512,512).cuda())
    print(out.shape)
