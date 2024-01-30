from tokenize import group
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from torch.nn import init
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import numpy as np
from torch import FloatTensor
BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x    

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class GhostBottleneck_CBAM(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, cbam=True):
        super(GhostBottleneck_CBAM, self).__init__()
        has_cbam = True if cbam else False
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        self.cbam = CBAM(mid_chs)

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )


    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Squeeze-and-excitation
        if self.cbam is not None:
            x = self.cbam(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        x += self.shortcut(residual)
        return x 

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out, groups_num=1):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True, groups=groups_num),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True, groups=groups_num),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out, groups_num=1):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True, groups=groups_num),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=1.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x
        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        x += self.shortcut(residual)
        return x 

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class GNN(nn.Module):
    def __init__(self, in_ch, n_hid, n_class, dropout=0.5, momentum=0.1):
        super(GNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = GNN_conv(in_ch, n_hid)
        self.hgc2 = GNN_conv(n_hid, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        # print(x.shape)
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        # print(x.shape)
        return x

class GNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GNN_conv, self).__init__()

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

class SimilarityAdj(Module):
    
    def __init__(self, in_features, out_features):
        super(SimilarityAdj, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight0 = Parameter(FloatTensor(in_features, out_features))
        self.weight1 = Parameter(FloatTensor(in_features, out_features))
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / sqrt(self.weight0.size(1))
        nn.init.xavier_uniform_(self.weight0)
        nn.init.xavier_uniform_(self.weight1)

    def forward(self, input, seq_len):
        # To support batch operations
        soft = nn.Softmax(1)
        theta = torch.matmul(input, self.weight0)
        phi = torch.matmul(input, self.weight0)
        phi2 = phi.permute(0, 2, 1)
        sim_graph = torch.matmul(theta, phi2)

        theta_norm = torch.norm(theta, p=2, dim=2, keepdim=True)  # B*T*1
        phi_norm = torch.norm(phi, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = theta_norm.matmul(phi_norm.permute(0, 2, 1))
        sim_graph = sim_graph / (x_norm_x + 1e-20)

        # output = torch.zeros_like(sim_graph)
        # if seq_len is None:
        #     for i in range(sim_graph.shape[0]):
        #         tmp = sim_graph[i]
        #         adj2 = tmp
        #         adj2 = F.threshold(adj2, 0.7, 0)
        #         adj2 = soft(adj2)
        #         output[i] = adj2
        # else:
        #     for i in range(len(seq_len)):
        #         tmp = sim_graph[i, :seq_len[i], :seq_len[i]]
        #         adj2 = tmp
        #         adj2 = F.threshold(adj2, 0.7, 0)
        #         adj2 = soft(adj2)
        #         output[i, :seq_len[i], :seq_len[i]] = adj2

        return sim_graph

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Inception(nn.Module):
    def __init__(self, basic_conv = Conv2d_cd):
        super(Inception, self).__init__()
        self.branch_0 = nn.Sequential(
            basic_conv(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True))

        self.branch_1 = nn.Sequential(
            basic_conv(3, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.branch_2 = nn.Sequential(
            basic_conv(3,  16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.brance_3 = nn.Sequential(
            nn.AvgPool2d(3, 1, padding=1, count_include_pad=False),
            basic_conv(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.brance_4 = nn.Sequential(
            nn.Conv2d(32 * 4, 32, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.brance_3(x)
        x4 = torch.cat((x0, x1, x2, x3), dim=1)
        return self.brance_4(x4)

class Lite_UNet(nn.Module):
    def __init__(self, basic_conv = Conv2d_cd):
        super(Lite_UNet,self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Conv1 =  Inception()

        self.Conv2 = GhostBottleneck_CBAM(32, 64, 64)
        self.Conv3 = GhostBottleneck_CBAM(64, 128, 128)
        self.Conv4 = GhostBottleneck_CBAM(128, 256, 256)
        self.Conv5 = GhostBottleneck_CBAM(256, 512, 512)

        self.sigmoid = nn.Sigmoid()

        self.GNN = SimilarityAdj(1024, 512)
        self.gc1 = GNN(1024, 512, 512)
        self.gc2 = GNN(512, 1024, 1024)

        self.Up5 = up_conv(ch_in=512,ch_out=256, groups_num=4)
        self.Up4 = up_conv(ch_in=256,ch_out=128, groups_num=4)
        self.Up3 = up_conv(ch_in=128,ch_out=64, groups_num=4)
        self.Up2 = up_conv(ch_in=64,ch_out=32, groups_num=4)

        self.Up_conv5 = GhostBottleneck_CBAM(512, 256, 256)
        self.Up_conv4 = GhostBottleneck_CBAM(256, 128, 128)
        self.Up_conv3 = GhostBottleneck_CBAM(128, 64, 64)
        self.Up_conv2 = GhostBottleneck_CBAM(64, 32, 32)

        self.Conv_1x1 = nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        x6 = x5.reshape(x5.shape[0],-1,1024)
        seq_len = torch.sum(torch.max(torch.abs(x6), dim=2)[0]>0, 1)
        G = self.GNN(x6, seq_len)
        x6 = torch.relu(self.gc1(x6, G))
        x6 = torch.relu(self.gc2(x6, G))

        x6 = x6.reshape(x6.shape[0],512,32,32)

        x5 = x5 * self.sigmoid(x6)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv_1x1(d2)
        
        return out

if __name__ == "__main__":
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    net = Lite_UNet().cuda()
    # print(net)
    input = torch.randn(1, 3, 512, 512).cuda()
    x= net(input)
    print(x.shape)
    # from thop import profile
    # flops, params = profile(net, inputs=(input, ))
    # # summary(net,(3,64 ,64 ),batch_size=4)
    # print('Total params: %.2fM' % (params/1000000.0))
    # print('Total flops: %.2fG' % (flops/1000000000.0))
