import torch.nn as nn
import torch
from torchvision import models
# from utils import save_net,load_net
import torch.nn.functional as F
from torch.nn import init
import math

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
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups)
        # self.conv = basic_conv(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups)
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
        self.conv = nn.Conv2d(nIn, nOut,kSize, stride=stride, padding=padding, bias=False,
                              dilation=d, groups=groups)
        # self.conv = basic_conv(nIn, nOut,kSize, stride=stride, padding=padding, bias=False,
        #                       dilation=d, groups=groups)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class DSP(nn.Module):
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

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        # self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self.last_layer = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=0, bias=True),
            nn.BatchNorm2d(32, momentum= 0.01),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, output_padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1, output_padding=0, bias=True),
        )

        if not load_weights:
            # mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            # for i in range(len(self.frontend.state_dict().items())):
            #     list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
    
    def forward(self,x):

        x = self.frontend(x)
        # print(x.shape)
        x = self.backend(x)
        # print(x.shape)
        x = self.last_layer(x)

        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False, basic_conv=Conv2d_cd):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # conv2d = basic_conv(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)                

class CSRNet_CDC(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet_CDC, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat,in_channels = 64)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        # self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.MGA = DSP(nIn=3,nOut=64)

        self.last_layer = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=0, bias=True),
            nn.BatchNorm2d(32, momentum= 0.01),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, output_padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1, output_padding=0, bias=True),
        )

        self._initialize_weights()
        # if not load_weights:
        #     mod = models.vgg16(pretrained = True)
        #     self._initialize_weights()
        #     for i in range(len(self.frontend.state_dict().items())):
        #         list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
    def forward(self,x):
        x = self.MGA(x)
        x = self.frontend(x)
        # print(x.shape)
        x = self.backend(x)
        # print(x.shape)
        x = self.last_layer(x)

        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
  

if __name__ == "__main__":
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '5'

    net = CSRNet().cuda()
    # print(net)
    input = torch.randn(1, 3, 512, 512).cuda()
    out = net(input)
    print(out.shape)
    # from thop import profile
    # flops, params = profile(net, inputs=(input, ))
    # # summary(net,(3,64 ,64 ),batch_size=4)
    # print('Total params: %.2fM' % (params/1000000.0))
    # print('Total flops: %.2fG' % (flops/1000000000.0))
