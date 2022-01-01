# dcd and DDPM
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import fusion_strategy
from args_fusion import args


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.

class SEModule_small(nn.Module):
    def __init__(self, channel):
        super(SEModule_small, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y

class conv_dy(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride):
        super(conv_dy, self).__init__()

        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)

        self.conv = nn.Conv2d(inplanes, planes, kernel_size, stride, bias=False)
        self.dim = int(math.sqrt(inplanes))
        squeeze = max(inplanes, self.dim ** 2) // 16

        self.q = nn.Conv2d(inplanes, self.dim, 1, stride, 0, bias=False)
        self.p = nn.Conv2d(self.dim, planes, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(inplanes, squeeze, bias=False),
            SEModule_small(squeeze),
        )
        self.fc_phi = nn.Linear(squeeze, self.dim ** 2, bias=False)
        self.fc_scale = nn.Linear(squeeze, planes, bias=False)
        self.hs = Hsigmoid()

    def forward(self, x):
        # x = self.reflection_pad(x)
        # print("x size:  ", x.size())
        x0 = self.reflection_pad(x)
        r = self.conv(x0)
        # print("r size:  ", r.size())
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        phi = self.fc_phi(y).view(b, self.dim, self.dim)
        scale = self.hs(self.fc_scale(y)).view(b, -1, 1, 1)
        r = scale.expand_as(r) * r

        out = self.bn1(self.q(x))
        _, _, h, w = out.size()

        out = out.view(b, self.dim, -1)
        out = self.bn2(torch.matmul(phi, out)) + out
        out = out.view(b, -1, h, w)
        # print("out size:",out.size())
        # print("r size:  ",r.size())
        out = self.p(out) +r
        return out

# Dense convolution unit
class DY_DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DY_DenseConv2d, self).__init__()
        self.dy_dense_conv = conv_dy(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dy_dense_conv(x)
        out = F.relu(out, inplace=True)
        out = torch.cat([x, out], 1)
        return out

# dy Block unit
class DY_DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels_def,kernel_size, stride):
        super(DY_DenseBlock, self).__init__()
        # out_channels_def = 16

        self.dy_block1= DY_DenseConv2d(in_channels, out_channels_def, kernel_size, stride)
        self.dy_block2= DY_DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride)
        self.dy_block3= DY_DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride)
        denseblock = []
        denseblock += [DY_DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DY_DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride),
                       DY_DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out


# Dense convolution unit
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


# Dense Block unit
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels_def,kernel_size, stride):
        super(DenseBlock, self).__init__()
        # out_channels_def = 16

        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


# DenseFuse network
class DenseFuse_net(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(DenseFuse_net, self).__init__()
        denseblock = DenseBlock
        nb_filter = [16, 64, 32, 16]
        kernel_size = 3
        stride = 1

        # encoder

        self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.DB1 = denseblock(nb_filter[0],nb_filter[0], kernel_size, stride)


        # self.conv2 = ConvLayer(input_nc2, nb_filter[0], kernel_size, stride)
        # self.DB2 = denseblock(nb_filter[0], kernel_size, stride)

        # decoder
        self.conv2 = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)##concate之后为128通道
        self.conv3 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.conv4 = ConvLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.conv5 = ConvLayer(nb_filter[3], output_nc, kernel_size, stride)

    def encoder(self, input_s1,input_s2):
        x1 = self.conv1(input_s1)
        x_DB = self.DB1(x1)
        x2 = self.conv1(input_s2)
        x_DB2 = self.DB1(x2)
        # db_fusion = torch.cat((x_DB, x_DB2),dim = 1) ##concate在一起
        db_fusion = torch.max(x_DB, x_DB2)
        return [db_fusion]
        # return x_DB,x_DB2

    def fusion(self, en1, en2, strategy_type='addition'):
        # addition
        if strategy_type is 'attention_weight':
            # attention weight
            fusion_function = fusion_strategy.attention_fusion_weight
        else:
            fusion_function = fusion_strategy.addition_fusion

        f_0 = fusion_function(en1[0], en2[0])
        return [f_0]

    # def fusion(self, en1, en2, strategy_type='addition'):
    #     f_0 = (en1[0] + en2[0])/2
    #     return [f_0]

    def decoder(self, f_en):
        x2 = self.conv2(f_en[0])
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        output = self.conv5(x4)

        return [output]


# DenseFuse network
# class MedFuse_dcd_net(nn.Module):
#     def __init__(self,input_nc=1, output_nc=1):
#         super(MedFuse_dcd_net, self).__init__()
#         denseblocks = DY_DenseBlock
#         # denseblocks = DenseBlock
#
#         nb_filter = [16, 64, 32, 16]
#         kernel_size = 3
#         stride = 1
#
#         # decoder
#         self.conv2 = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)  ##concate之后为128通道
#         self.conv3 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
#         self.conv4 = ConvLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
#         self.conv5 = ConvLayer(nb_filter[3], output_nc, kernel_size, stride)
#
#         # Initialize parameters for other parameters
#         # for m in self.modules():
#         #     if isinstance(m, nn.Conv2d):
#         #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#         #         m.weight.data.normal_(0, math.sqrt(2. / n))
#
#         # self.selfdc_4 = DDPM(64, 64, 64, 3, 4)
#
#         ## encoder
#         # encoder
#         self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
#         self.DB1 = denseblocks(nb_filter[0], nb_filter[0], kernel_size, stride)
#
#         inplanes = nb_filter[0]*6
#         self.dy_conv = conv_dy(inplanes, nb_filter[0], 1, stride)
#
#         # Initialize conv1 with the pretrained model and freeze its parameters
#         # for p in pretrained_dict.parameters():
#         #     p.requires_grad = False
#         # self.conv1 = pretrained_dict.conv1
#         # self.conv1.stride = stride
#         # self.conv1.padding = (0, 0)
#         # #
#         # self.DB1 = pretrained_dict.DB1
#         # self.DB1.stride = stride
#         # self.DB1.padding = (0, 0)
#     # self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
#     # self.DB1 = denseblock(nb_filter[0], kernel_size, stride)
#
#     def encoder(self, input_s1, input_s2):
#         x1 = self.conv1(input_s1)
#         x_DB = self.DB1(x1)
#         x2 = self.conv1(input_s2)
#         x_DB2 = self.DB1(x2)
#         # db_fusion = torch.cat((x_DB, x_DB2),dim = 1) ##concate在一起
#         # db_fusion = torch.max(x_DB, x_DB2)
#         # return [db_fusion]
#         return x_DB, x_DB2
#
#     # def fusion(self, en1, en2, strategy_type='addition'):
#     #     # addition
#     #     if strategy_type is 'attention_weight':
#     #         # attention weight
#     #         fusion_function = fusion_strategy.attention_fusion_weight
#     #     else:
#     #         fusion_function = fusion_strategy.addition_fusion
#     #
#     #     f_0 = fusion_function(en1[0], en2[0])
#     #     return [f_0]
#
#     def fusion(self, en1, en2, strategy_type='addition'):
#         f_0 = (en1[0] + en2[0]) / 2
#         return [f_0]
#
#     # 特征动态融合
#     def dy_fusion(self,en1,en2):
#
#         # 动态分解方法实现通道融合
#         x = torch.cat([en1,en2], 1)
#         dy_fuse = dy_conv(x)
#
#         #DDPM
#
#
#         # 耦合反馈
#
#         return dy_fuse
#
#
#
#
#     def decoder(self, f_en):
#         x2 = self.conv2(f_en[0])
#         x3 = self.conv3(x2)
#         x4 = self.conv4(x3)
#         output = self.conv5(x4)
#
#         return [output]

def medtrain(input_nc, output_nc):
    # pre-trained model
    if args.resume is not None:
        print('pretrained model using weight from {}.'.format(args.resume))  ##恢复与初始化权重
        pretrained_model = DenseFuse_net(input_nc, output_nc)
        pretrained_model.load_state_dict(torch.load(args.resume))
        # pretrained_dict = torch.load(args.resume)
        # print(pretrained_dict.items())
    # ##继续训练
    trained_model = DY_MedFuse_net(pretrained_model,None,input_nc, output_nc)
    # trained_model.load_state_dict(torch.load(args.trained_model)['model'])

    # our model
    net = DY_MedFuse_net(pretrained_model,trained_model,input_nc, output_nc)
    # model_dict = net.state_dict()
    # print(model_dict.items())

    # dy_fuse
    # net = MedFuse_dcd_net(pretrained_model,input_nc, output_nc)
    # model_dict = net.state_dict()

    #
    # pretrained_dict = {k: v for k, v in torch.load(args.trained_model)['model'].items() if k in model_dict}
    # model_dict.update(pretrained_dict)  # 用预训练模型参数更新new_model中的部分参数
    # net.load_state_dict(model_dict)  # 将更新后的model_dict加载进new model中
    # #
    # # ##### 冻结部分参数
    # for param in net.parameters():
    #     param.requires_grad = False  # 设置所有参数不可导，下面选择设置可导的参数
    # for param in net.conv1.parameters():
    #     print("param: ",param.requires_grad)
    # for param in net.DB1.parameters():
    #     print("param: ", param.requires_grad)
    #     param.requires_grad = True

    return net

class DY_MedFuse_net(nn.Module):
    def __init__(self, pretrained_dict,trained_model,input_nc=1, output_nc=1):
        super(DY_MedFuse_net, self).__init__()
        denseblocks = DY_DenseBlock
        # denseblocks = DenseBlock

        nb_filter = [16, 64, 32, 16]
        kernel_size = 3
        stride = 1
        self.DB1 = denseblocks(nb_filter[0], nb_filter[0],kernel_size, stride)
        self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)

        if trained_model:
            self.conv1 = trained_model.conv1
            self.conv1.stride = stride
            self.conv1.padding = (0, 0)
            self.DB1 = trained_model.DB1
            # self.conv2 = trained_model.conv2
            # # self.conv3 =trained_model.conv3
            # # self.conv4 =trained_model.conv4
            # self.conv5 = trained_model.conv5
            # self.conv6 = trained_model.conv6
            # self.DB2 = trained_model.DB2
        else:
            # self.conv1 = pretrained_dict.conv1
            self.conv1.stride = stride
            self.conv1.padding = (0, 0)

            # self.DB1.denseblock[0].dy_dense_conv.conv = pretrained_dict.DB1.denseblock[0].dense_conv.conv2d
            # self.DB1.denseblock[1].dy_dense_conv.conv = pretrained_dict.DB1.denseblock[1].dense_conv.conv2d
            # self.DB1.denseblock[2].dy_dense_conv.conv = pretrained_dict.DB1.denseblock[2].dense_conv.conv2d


        # self.DB2 = denseblocks(nb_filter[0], nb_filter[0],kernel_size, stride)

        # decoder
        self.conv2 = conv_dy(nb_filter[1], nb_filter[1], kernel_size, stride)  ##concate之后为128通道
        self.conv3 = conv_dy(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.conv4 = conv_dy(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.conv5 = ConvLayer(nb_filter[3], output_nc, kernel_size, stride)

        # Initialize parameters for other parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        # self.conv6 = pretrained_dict.conv1
        # self.conv6.stride = stride
        # self.conv6.padding = (0, 0)

        # self.DB2.denseblock[0].dy_dense_conv.conv = pretrained_dict.DB1.denseblock[0].dense_conv.conv2d
        # self.DB2.denseblock[1].dy_dense_conv.conv = pretrained_dict.DB1.denseblock[1].dense_conv.conv2d
        # self.DB2.denseblock[2].dy_dense_conv.conv = pretrained_dict.DB1.denseblock[2].dense_conv.conv2d

    def encoder(self, input_s1, input_s2):
        x1 = self.conv1(input_s1)
        x_DB = self.DB1(x1)

        x2 = self.conv1(input_s2)
        x_DB2 = self.DB1(x2)
        # db_fusion = torch.cat((x_DB, x_DB2),dim = 1) ##concate在一起
        db_fusion = torch.max(x_DB, x_DB2)
        return [db_fusion]
        # return x_DB,x_DB2

    def fusion(self, en1, en2, strategy_type='addition'):
        # addition
        if strategy_type is 'attention_weight':
            # attention weight
            fusion_function = fusion_strategy.attention_fusion_weight
        else:
            fusion_function = fusion_strategy.addition_fusion

        f_0 = fusion_function(en1, en2)
        return [f_0]

    # def fusion(self, en1, en2, strategy_type='addition'):
    #     f_0 = (en1[0] + en2[0]) / 2
    #     return [f_0]

    def decoder(self, f_en):
        x2 = self.conv2(f_en[0])
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        output = self.conv5(x4)

        return [output]
