'''

mode: Jie LI
Kiktech
waterljwant@gmail.com
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb


input_size=[480, 640]
width_mult = 1.

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# class FrozenBatchNorm2d(nn.Module):
#    """
#    BatchNorm2d where the batch statistics and the affine parameters
#    are fixed
#    """
#
#    def __init__(self, n):
#        super(FrozenBatchNorm2d, self).__init__()
#        self.register_buffer("weight", torch.ones(n))
#        self.register_buffer("bias", torch.zeros(n))
#        self.register_buffer("running_mean", torch.zeros(n))
#        self.register_buffer("running_var", torch.ones(n))
#
#    def forward(self, x):
#        #modified as suggested by jiewei
#        scale = self.weight * (self.running_var + 0.000000001).rsqrt()
#        bias = self.bias - self.running_mean * scale
#        scale = scale.reshape(1, -1, 1, 1)
#        bias = bias.reshape(1, -1, 1, 1)
#        return x * scale + bias

## Mod by Minming to not froze
FrozenBatchNorm2d = nn.BatchNorm2d

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        FrozenBatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        FrozenBatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

# filter size : 1x1 conv
# dimension   : 65 x 65 x 320 --> 65 x 65 x 256
class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes):
        super(ASPP_module, self).__init__()


        self.aspp0 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1,
                                             stride=1, padding=0, dilation=1, bias=False),
                                   nn.BatchNorm2d(planes))
        self.aspp1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                             stride=1, padding=6, dilation=6, bias=False),
                                   nn.BatchNorm2d(planes))
        self.aspp2 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                             stride=1, padding=12, dilation=12, bias=False),
                                    nn.BatchNorm2d(planes))
        self.aspp3 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                             stride=1, padding=18, dilation=18, bias=False),
                                   nn.BatchNorm2d(planes))

    def forward(self, x):
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)

        return torch.cat((x0, x1, x2, x3), dim=1)


class DeepLabv_v3_plus_decoder(nn.Module):
    def __init__(self, n_classes=1, upsample=False):
        super(DeepLabv_v3_plus_decoder, self).__init__()
        ic1 = 1024
        # ic2 = 256
        # ic1 = 320
        ic2 = 24
        self.upsample = upsample

        # ASPP
        self.aspp = ASPP_module(ic1, 256)
        # global pooling
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(ic1, 256, 1, stride=1, bias=False))

        self.conv = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn = nn.BatchNorm2d(256)

        # low_level_features to 48 channels
        self.conv2 = nn.Conv2d(ic2, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

        # init weights
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, low_level_features):
        # bs, c, h, w = x.size()
        # # padding
        # # if h == 720:  # 720*1280 -->736*1280
        # if h % 32 != 0:
        #     m, n = divmod(h, 32)
        #     ph = int(((m+1)*32-h)/2)
        #     x = F.pad(x, (0, 0, ph, ph), "constant", 0)
        # print(x.size())

        # x : 1/32 16 x 16
        # x, low_level_features = self.feature_extractor(x)
        # print('feature_extractor', x.size(), low_level_features.size())

        if self.upsample:  # TODO 1/16 ---> 1/8
            # NOTE ! !
            # 1/32 16 x 16 too small, up first and then ASPP
            x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)  # For pytorch 0.4.0
            # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # For pytorch 0.4.1
            # print(x.size())

        x_aspp = self.aspp(x)

        x_ = self.global_avg_pool(x)
        x_ = F.upsample(x_, size=x.size()[2:], mode='bilinear', align_corners=True)  # For pytorch 0.4.0
        # x_ = F.interpolate(x_, size=x.size()[2:], mode='bilinear', align_corners=True)

        # print('x_aspp.size(), x_.size()', x_aspp.size(), x_.size())
        x = torch.cat((x_aspp, x_), dim=1)

        x = self.conv(x)
        x = self.bn(x)
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)  # For pytorch 0.4.0
        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        # 1/4 128 x 128
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)

        # print('x.size(), low_level_features.size()', x.size(), low_level_features.size())
        x = torch.cat((x, low_level_features), dim=1)

        x = self.last_conv(x)
        x = F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True)  # For pytorch 0.4.0
        # x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

        # if h % 32 != 0:
        #     x = x[:, :, ph:h+ph, :]
        return x


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                FrozenBatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                FrozenBatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                FrozenBatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                FrozenBatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                FrozenBatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                FrozenBatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                FrozenBatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                FrozenBatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)



class ShuffleNetV2_8(nn.Module):
    def __init__(self, n_class=2, width_mult=1.0):
        super(ShuffleNetV2_8, self).__init__()

        # assert (input_size[0] % 32 == 0 and input_size[1] % 32 == 0)

        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []

        self.stage_2 = []
        self.stage_3 = []
        self.stage_4 = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        self.stage_2 = nn.Sequential(self.features[0], self.features[1], self.features[2], self.features[3])

        self.stage_3 = nn.Sequential(self.features[4], self.features[5], self.features[6], self.features[7],
                                     self.features[8], self.features[9], self.features[10], self.features[11])

        self.stage_4 = nn.Sequential(
            InvertedResidual(232, 464, 1, 2),
            InvertedResidual(232, 464, 1, 1),
            InvertedResidual(232, 464, 1, 1),
            InvertedResidual(232, 464, 1, 1)
        )

        # building last several layers
        self.conv_last = conv_1x1_bn(464, 1024)

        # self.top = nn.Conv2d(232, 2048, 1, 1)
        self.freeze_bn()



    def freeze_bn(self):
        ##TODO to fix bn, now test fix all, next only fix some part
        print("called freeze bn, but this not working as imagined")
        self.freeze_bn_affine = True
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if self.freeze_bn_affine:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def forward(self, x):
        # print('x-1', x.size())

        x = self.conv1(x)
        # print('x-2', x.size())

        x = self.maxpool(x)
        # print('x-3', x.size())

        #low_level_feature = x
        #test4
        fx = self.stage_2(x)
        # print('stage_2', fx.size())

        # y = self.stage_3(fx)
        # print('stage_3', y.size())

        # y = self.stage_4(y)
        # print('stage_4', y.size())

        return fx

class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super(UpsamplerBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        #print(input.size())
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super(non_bottleneck_1d, self).__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output + input)  # +input = identity (residual connection)

if width_mult == 0.5:
    stage_out_channels = [-1, 24, 48, 96, 192, 1024]
elif width_mult == 1.0:
    stage_out_channels = [-1, 24, 116, 232, 464, 1024]
elif width_mult == 1.5:
    stage_out_channels = [-1, 24, 176, 352, 704, 1024]
elif width_mult == 2.0:
    stage_out_channels = [-1, 24, 224, 488, 976, 2048]

class Decoder (nn.Module):
    def __init__(self, num_classes=2):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList()
        '''
        self.layers.append(UpsamplerBlock(128,64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64,16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        '''

        # self.layers.append(UpsamplerBlock(304,116))
        # self.layers.append(non_bottleneck_1d(116, 0, 1))
        # self.layers.append(non_bottleneck_1d(116, 0, 1))

        self.layers.append(UpsamplerBlock(208, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d( 16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)


        self.aspp16 = ASPP_module(stage_out_channels[3], 256)
        self.aspp8 = ASPP_module(stage_out_channels[2], 256)

        self.global_avg_pool16 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(stage_out_channels[3], 256, 1, stride=1, bias=False))

        self.global_avg_pool8 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(stage_out_channels[2], 256, 1, stride=1, bias=False))

        self.last_conv = nn.Sequential(nn.Conv2d(304, num_classes, kernel_size=1, stride=1))

        self.conv = nn.Conv2d(1280, 128, 1, bias=False)
        self.bn = nn.BatchNorm2d(128)

        self.conv_f8 = nn.Conv2d(116, 64, 1, bias=False)
        self.bn_f8 = nn.BatchNorm2d(64)

        self.conv_f4 = nn.Conv2d(24, 64, 3, stride=2,bias=False, padding=1)
        self.bn_f4 = nn.BatchNorm2d(64)
        self.conv_f4_last = nn.Conv2d(64, 16, 1, bias=False)
        self.bn_f4_last = nn.BatchNorm2d(16)

        self.relu = nn.ReLU()


    #x is 1/4 320*176 , y is 1/8 160 * 88 and z is 1/16 80 * 44
    def forward(self, x, y, z):
        # print("x.size", x.size())
        # print("z.size", low_level_features.size())

        #1/16 feature
        z_aspp = self.aspp16(z)
        z_ = self.global_avg_pool16(z)
        z_ = F.upsample(z_, size=z.size()[2:], mode='bilinear')
        z = torch.cat((z_aspp, z_), dim=1)
        #print(z.size())
        z = self.conv(z)
        z = self.bn(z)
        z = self.relu(z)
        z = F.upsample(z, scale_factor=2, mode='bilinear')


        #1/8 feature
        #y_aspp = self.aspp8(y)
        #y_ = self.global_avg_pool8(y)
        #y_ = F.upsample(y_, size=y.size()[2:], mode='bilinear')
        y = self.conv_f8(y)
        y = self.bn_f8(y)
        y = self.relu(y)


        #1/4 feature
        x = self.conv_f4(x)
        x = self.bn_f4(x)
        x = self.relu(x)
        x = self.conv_f4_last(x)
        x = self.bn_f4_last(x)
        x = self.relu(x)

        #print(x.size())
        # 1/4 320 x 176
        # x = self.conv2(x)
        # x = self.bn2(x)

        # print("low level", low_level_features.size())
        x = torch.cat((x, y, z), dim=1)
        #print(x.size())
        # x = self.last_conv(x)
        # x = F.upsample(x, scale_factor=4, mode='bilinear')
        # print("x size", x.size())

        output = x

        for layer in self.layers:
            output = layer(output)
        output = self.output_conv(output)

        # print("output size", output.size())
        return output

class shufflenet(_fasterRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False):  # TODO
        self.model_path = ''
        self.dout_base_model = 232
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        _fasterRCNN.__init__(self, classes, class_agnostic)

        self.dlb = True  # Use mobilenet/shufflenet + deeplabv3 for segmentation

    def _init_modules(self):
        #build the layers
        shufflenet = ShuffleNetV2_8()

        # Deleted by Minming, not train from this model anymore

        self.RCNN_low_base = nn.Sequential(shufflenet.conv1,
                                           shufflenet.maxpool,)  # 1/4

        self.RCNN_mid_base = nn.Sequential(shufflenet.stage_2)    # 1/8

        self.RCNN_base = nn.Sequential(shufflenet.stage_3)  # 1/16

        self.RCNN_top = nn.Sequential(shufflenet.stage_4,
                                      shufflenet.conv_last)


        self.SegDecoder = Decoder(2)  # Drive Line Segmentation, mobilenetv2_32,  TODO: 1/16--->1/8

        #self.SegDecoder = DeepLabv_v3_plus_decoder(n_classes=2, upsample=True)  # Drive Line Segmentation, mobilenetv2_32

        self.RCNN_cls_score = nn.Linear(1024, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(1024, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(1024, 4 * self.n_classes)

        # Added by Minming, load the model from pretrained detection model
        if self.pretrained == False and self.training:
            self.model_path = "./models/faster_rcnn_1_3_5475.pth"
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)

            pretrained_dict = {}
            for k, v in state_dict['model'].items():
                pretrained_dict.update({k: v})

            model_dict = self.state_dict()
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

            #todo here we fix the parameters load from the pretrained detection model
            fix = False
            if fix and self.training:
                for p in self.RCNN_rpn.RPN_Conv.parameters(): p.requires_grad = False
                for p in self.RCNN_rpn.RPN_cls_score.parameters(): p.requires_grad = False
                for p in self.RCNN_rpn.RPN_bbox_pred.parameters(): p.requires_grad = False
                for p in self.RCNN_low_base[0].parameters(): p.requires_grad = False
                for p in self.RCNN_low_base[1].parameters(): p.requires_grad = False
                for p in self.RCNN_mid_base[0].parameters(): p.requires_grad = False
                for p in self.RCNN_base[0].parameters(): p.requires_grad = False
                for p in self.RCNN_top[0].parameters(): p.requires_grad = False
                for p in self.RCNN_top[1].parameters(): p.requires_grad = False
                for p in self.RCNN_bbox_pred.parameters(): p.requires_grad = False
                for p in self.RCNN_cls_score.parameters(): p.requires_grad = False

                # self.RCNN_rpn.RPN_Conv.apply(set_bn_fix)
                # self.RCNN_rpn.RPN_cls_score.apply(set_bn_fix)
                # self.RCNN_rpn.RPN_bbox_pred.apply(set_bn_fix)
                # self.RCNN_low_base[0].apply(set_bn_fix)
                # self.RCNN_low_base[1].apply(set_bn_fix)
                # self.RCNN_base[0].apply(set_bn_fix)
                # self.RCNN_base[1].apply(set_bn_fix)
                # self.RCNN_top[0].apply(set_bn_fix)
                # self.RCNN_top[1].apply(set_bn_fix)
                # self.RCNN_bbox_pred.apply(set_bn_fix)
                # self.RCNN_cls_score.apply(set_bn_fix)

        #todo here to check the weights, and one epoch later to check again
        #print(self.state_dict())
        # self.freeze_bn()


    def print_paras(self):
        to_output = False
        if to_output:
            print("self.RCNN_rpn.RPN_Conv")
            for p in self.RCNN_rpn.RPN_Conv.parameters(): print(p.data)
            print("self.RCNN_rpn.RPN_cls_score")
            for p in self.RCNN_rpn.RPN_cls_score.parameters(): print(p.data)
            print("self.RCNN_rpn.RPN_bbox_pred")
            for p in self.RCNN_rpn.RPN_bbox_pred.parameters(): print(p.data)
            print("self.RCNN_low_base[0]")
            for p in self.RCNN_low_base[0].parameters(): print(p.data)
            print("self.RCNN_low_base[1]")
            for p in self.RCNN_low_base[1].parameters(): print(p.data)
            print("self.RCNN_base[0]")
            for p in self.RCNN_base[0].parameters(): print(p.data)
            print("self.RCNN_base[1]")
            for p in self.RCNN_base[1].parameters(): print(p.data)
            print("self.RCNN_top[0]")
            for p in self.RCNN_top[0].parameters(): print(p.data)
            print("self.RCNN_top[1]")
            for p in self.RCNN_top[1].parameters(): print(p.data)
            print("self.RCNN_bbox_pred")
            for p in self.RCNN_bbox_pred.parameters(): print(p.data)
            print("self.RCNN_cls_score")
            for p in self.RCNN_cls_score.parameters(): print(p.data)

    def _head_to_tail(self, pool5):
        fc7 = self.RCNN_top(pool5).mean(3).mean(2)
        return fc7
