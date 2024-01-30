from  torchvision import models
import sys
import torch.nn.functional as F
import torch.nn as nn
import torch
# from misc.utils import *
import math
from torch.nn.parameter import Parameter
import numpy as np
from torch import FloatTensor

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
            # [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            # print(self.conv.weight.shape)
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            # print(kernel_diff)
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

class CBG(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, basic_conv = Conv2d_cd):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        # self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups)
        self.conv = basic_conv(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        # output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output

class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1, basic_conv = Conv2d_cd):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        # self.conv = nn.Conv2d(nIn, nOut,kSize, stride=stride, padding=padding, bias=False,
        #                       dilation=d, groups=groups)
        self.conv = basic_conv(nIn, nOut,kSize, stride=stride, padding=padding, bias=False,
                              dilation=d, groups=groups)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class MDC(nn.Module):
    def __init__(self, nIn, nOut, stride=1, k=4, r_lim=9, down_method='esp'): #down_method --> ['avg' or 'esp']
        super().__init__()
        self.stride = stride
        n = int(nOut / k) # 得出每个分支的 channel
        n1 = nOut - (k - 1) * n
        assert down_method in ['avg', 'esp'], 'One of these is suppported (avg or esp)'
        assert n == n1, "n(={}) and n1(={}) should be equal for Depth-wise Convolution ".format(n, n1)
        self.proj_1x1 = CBG(nIn, n, 1, stride=1, groups=1)

        # (For convenience) Mapping between dilation rate and receptive field for a 3x3 kernel
        map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        self.k_sizes = list()
        for i in range(k): # 4
            ksize = int(3 + 2 * i)
            # After reaching the receptive field limit, fall back to the base kernel size of 3 with a dilation rate of 1
            ksize = ksize if ksize <= r_lim else 3
            self.k_sizes.append(ksize) # 3 5 7 3
        # sort (in ascending order) these kernel sizes based on their receptive field
        # This enables us to ignore the kernels (3x3 in our case) with the same effective receptive field in hierarchical
        # feature fusion because kernels with 3x3 receptive fields does not have gridding artifact.
        self.k_sizes.sort() # 3 5 7 9
        self.spp_dw = nn.ModuleList()
        for i in range(k):
            d_rate = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(CDilated(n, n, kSize=3, stride=stride, groups=1, d=d_rate))
        #  3 5 7 9 对应 1 2 3 4

    def forward(self, input):
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        # compute the output for each branch and hierarchically fuse them
        # i.e. Split --> Transform --> HFF
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            # HFF
            out_k = out_k + output[k - 1]
            # out_k = out_k + output[0]
            output.append(out_k)
        # output.append(output1) # [1, 256, 64, 64]
        # for i in range(len(output)):
        #     print(output[i].shape)
        # Merge
        expanded = torch.cat(output, 1) # concatenate the output of different branches
        # [1, 1280, 64, 64]
        return expanded

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
            vgg = models.vgg16_bn(pretrained=False)
        features = list(vgg.features.children())

        # self.MGA = MDC(nIn=3,nOut=64)
        if  mode == 'Vgg_bn':
            self.layer1 = nn.Sequential(*features[0:23])
            self.layer2 = nn.Sequential(*features[23:33])
            self.layer3 = nn.Sequential(*features[33:43])
        in_channels = [256,512,512]
        self.neck =  FPN(in_channels,256,len(in_channels))
        self.de_pred = nn.Sequential(
            nn.Conv2d(
                in_channels=768,
                out_channels=768,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(768, 64, 4, stride=2, padding=1, output_padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, output_padding=0, bias=True),
            nn.Sigmoid()
        )
        initialize_weights(self.de_pred)

    def forward(self, x):
        f = []
        # x = self.MGA(x)
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        f.append(x)
        x = self.layer2(x)
        f.append(x)
        x = self.layer3(x)
        f.append(x)

        f = self.neck(f)
        feature =torch.cat([f[0],  F.interpolate(f[1],scale_factor=2),F.interpolate(f[2],scale_factor=4)], dim=1)
        x = self.de_pred(feature)
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
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    net = VGG16_FPN(pretrained=False).cuda()
    # net = U_Net_MDC_GNN().cuda()
    # print(net)
    input = torch.randn(1, 3, 512, 512).cuda()
    # x= net(input)
    # print(x.shape)
    from thop import profile
    flops, params = profile(net, inputs=(input, ))
    # summary(net,(3,64 ,64 ),batch_size=4)
    print('Total params: %.2fM' % (params/1000000.0))
    print('Total flops: %.2fG' % (flops/1000000000.0))