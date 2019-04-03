'''
    MobileNet_v2_OS_32
    these codes brow from https://github.com/tonylins/pytorch-mobilenet-v2
    but little different from original architecture :
    +-------------------------------------------+-------------------------+
    |                                               output stride
    +===========================================+=========================+
    |       original MobileNet_v2_OS_32         |          32             |
    +-------------------------------------------+-------------------------+
    |   self.interverted_residual_setting = [   |                         |
    |       # t, c, n, s                        |                         |
    |       [1, 16, 1, 1],                      |  pw -> dw -> pw-linear  |
    |       [6, 24, 2, 2],                      |                         |
    |       [6, 32, 3, 2],                      |                         |
    |       [6, 64, 4, 2],                      |       stride = 2        |
    |       [6, 96, 3, 1],                      |                         |
    |       [6, 160, 3, 2],                     |       stride = 2        |
    |       [6, 320, 1, 1],                     |                         |
    |   ]                                       |                         |
    +-------------------------------------------+-------------------------+
    |          MobileNet_v2_OS_8                |          8              |
    +-------------------------------------------+-------------------------+
    |   self.interverted_residual_setting = [   |                         |
    |       # t, c, n, s                        |                         |
    |       [1, 16, 1, 1],                      |    dw -> pw-linear      |
    |       [6, 24, 2, 2],                      |                         |
    |       [6, 32, 3, 2],                      |                         |
    |       [6, 64, 4, 1],                      |       stride = 1        |
    |       [6, 96, 3, 1],                      |                         |
    |       [6, 160, 3, 1],                     |       stride = 1        |
    |       [6, 320, 1, 1],                     |                         |
    |   ]                                       |                         |
    +-------------------------------------------+-------------------------+

    Notation! I throw away last layers.


Author: Zhengwei Li
Data: July 1 2018

mode: Jie LI
Kiktech
waterljwant@gmail.com
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.faster_rcnn.faster_rcnn import _fasterRCNN

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



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
        ic1 = 2048
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


# -------------------------------------------------------------------------------------------------
# MobileNet_v2_os_32
# --------------------
class MobileNet_v2_os_32(nn.Module):
    def __init__(self, nInputChannels=3):
        super(MobileNet_v2_os_32, self).__init__()
        # 1/2
        # 256 x 256
        self.head_conv = conv_bn(nInputChannels, 32, 2)

        # 1/2
        # 256 x 256
        self.block_1 = InvertedResidual(32, 16, 1, 1)
        # 1/4 128 x 128
        self.block_2 = nn.Sequential(
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6)
            )
        # 1/8 64 x 64
        self.block_3 = nn.Sequential(
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6)
            )
        # 1/16 32 x 32
        self.block_4 = nn.Sequential(
            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6)
            )
        # 1/16 32 x 32
        self.block_5 = nn.Sequential(
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6)
            )
        # 1/32 16 x 16
        self.block_6 = nn.Sequential(
            InvertedResidual(96, 160, 2, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6)
            )
        # 1/32 16 x 16
        self.block_7 = InvertedResidual(160, 320, 1, 1)

    def forward(self, x):
        x = self.head_conv(x)

        x = self.block_1(x)
        x = self.block_2(x)
        low_level_feat = x

        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.block_7(x)

        return x, low_level_feat


# -------------------------------------------------------------------------------------------------
# MobileNet_v2_os_8
# --------------------
class MobileNet_v2_os_8(nn.Module):
    def __init__(self, nInputChannels=3):
        super(MobileNet_v2_os_8, self).__init__()

        # 1/2 256 x 256
        self.head_conv = conv_bn(nInputChannels, 32, 2)
        # 1/2 256 x 256
        # head bock different form original mobilenetv2
        self.block_1 = nn.Sequential(
            # dw
            nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(32, 16, 1, 1, 0, bias=False),
            nn.BatchNorm2d(16),
        )
        # 1/4 128 x 128
        self.block_2 = nn.Sequential(
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6)
            )
        # 1/8 64 x 64
        self.block_3 = nn.Sequential(
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6)
            )
        # 1/8 64 x 64
        self.block_4 = nn.Sequential(
            InvertedResidual(32, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6)
            )
        # 1/8 64 x 64
        self.block_5 = nn.Sequential(
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6)
            )
        # 1/8 64 x 64
        self.block_6 = nn.Sequential(
            InvertedResidual(96, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6)
            )
        # 1/8 64 x 64
        self.block_7 = InvertedResidual(160, 320, 1, 6)

    def forward(self, x):
        x = self.head_conv(x)

        x = self.block_1(x)
        x = self.block_2(x)
        # 1/4 128 x 128
        low_level_feat = x

        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.block_7(x)

        return x, low_level_feat


# -------------------------------------------------------------------------------------------------
class MobileNet_v2(nn.Module):
    def __init__(self, nInputChannels=3, d=32):
        super(MobileNet_v2, self).__init__()
        # 1/2
        # 256 x 256
        self.head_conv = conv_bn(nInputChannels, 32, 2)

        # 1/2
        # 256 x 256
        self.block_1 = InvertedResidual(32, 16, 1, 1)
        # 1/4 128 x 128
        self.block_2 = nn.Sequential(
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6)
            )
        # 1/8 64 x 64
        self.block_3 = nn.Sequential(
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6)
            )

        dilation = 2 if d == 32 else 1  # 2---1/32, 1---1/8
        # 1/16 32 x 32
        self.block_4 = nn.Sequential(
            InvertedResidual(32, 64, dilation, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6)
            )
        # 1/16 32 x 32
        self.block_5 = nn.Sequential(
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6)
            )
        # 1/32 16 x 16
        dilation = 1
        self.block_6 = nn.Sequential(
            InvertedResidual(96, 160, dilation, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 1024, 1, 6)
            )
        # 1/32 16 x 16
        # self.block_7 = InvertedResidual(1024, 2048, 1, 1)
        self.block_7 = nn.Conv2d(1024, 2048, 1, 1)

    def forward(self, x):
        x = self.head_conv(x)

        x = self.block_1(x)
        x = self.block_2(x)
        low_level_feat = x

        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.block_7(x)

        return x, low_level_feat


# -------------------------------------------------------------------------------------------------
class MobileNet_base(nn.Module):
    def __init__(self, nInputChannels=3):
        super(MobileNet_base, self).__init__()
        # 1/2
        # 256 x 256
        self.head_conv = conv_bn(nInputChannels, 32, 2)

        # 1/2
        # 256 x 256
        self.block_1 = InvertedResidual(32, 16, 1, 1)
        # 1/4 128 x 128
        self.block_2 = nn.Sequential(
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6)
            )
        # 1/8 64 x 64
        self.block_3 = nn.Sequential(
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6)
            )

        dilation = 2
        # 1/16 32 x 32
        self.block_4 = nn.Sequential(
            InvertedResidual(32, 64, dilation, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6)
            )
        # 1/16 32 x 32
        self.block_5 = nn.Sequential(
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6)
            )

        # 1/32 16 x 16
        dilation = 1
        self.block_6 = nn.Sequential(
            InvertedResidual(96, 160, dilation, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 1024, 1, 6)
            )

    def forward(self, x):
        x = self.head_conv(x)

        x = self.block_1(x)
        x = self.block_2(x)
        low_level_feat = x

        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)

        return x, low_level_feat


# -------------------------------------------------------------------------------------------------
class mobilenet(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):  # TODO
    # self.model_path = 'data/pretrained_model/resnet101_caffe.pth'  # TODO
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

    self.dlb = True  # Used for mobilenet+deeplabv3

  def _init_modules(self):
    # mobilenet = MobileNet_v2(d=8)
    mobilenet = MobileNet_v2(d=32)

    if self.pretrained == True:
        print("Do Not have a pretrained model")
      # print("Loading pretrained weights from %s" %(self.model_path))
      # state_dict = torch.load(self.model_path)
      # resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

    # Build resnet.
    self.RCNN_base_low = nn.Sequential(mobilenet.head_conv,
                                       mobilenet.block_1,
                                       mobilenet.block_2)  # 1/4
    self.RCNN_base = nn.Sequential(mobilenet.block_3,
                                   mobilenet.block_4,
                                   mobilenet.block_5,
                                   mobilenet.block_6)  # 1/32 0r 1/8

    # self.RCNN_base = nn.Sequential(mobilenet.head_conv,
    #                                mobilenet.block_1,
    #                                mobilenet.block_2,
    #                                mobilenet.block_3,
    #                                mobilenet.block_4,
    #                                mobilenet.block_5,
    #                                mobilenet.block_6)  # 1/32 0r 1/8

    self.RCNN_top = nn.Sequential(mobilenet.block_7)

    self.SegDecoder = DeepLabv_v3_plus_decoder(n_classes=2, upsample=True)  # Drive Line Segmentation, mobilenetv2_32,  TODO: 1/16--->1/8
    # self.SegDecoder = DeepLabv_v3_plus_decoder(n_classes=2, upsample=False)  # Drive Line Segmentation, mobilenetv2_8

    self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(2048, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

    # --- Fix blocks # TODO
    # for p in self.RCNN_base[0].parameters(): p.requires_grad=False
    # for p in self.RCNN_base[1].parameters(): p.requires_grad=False

    # assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    # if cfg.RESNET.FIXED_BLOCKS >= 3:
    #   for p in self.RCNN_base[6].parameters(): p.requires_grad=False
    # if cfg.RESNET.FIXED_BLOCKS >= 2:
    #   for p in self.RCNN_base[5].parameters(): p.requires_grad=False
    # if cfg.RESNET.FIXED_BLOCKS >= 1:
    #   for p in self.RCNN_base[4].parameters(): p.requires_grad=False

    # --- TODO
    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  # def train(self, mode=True):
  #   # Override train so that the training mode is set as we want
  #   nn.Module.train(self, mode)
  #   if mode:
  #     # Set fixed blocks to be in eval mode
  #     self.RCNN_base.eval()
  #     self.RCNN_base[5].train()
  #     self.RCNN_base[6].train()
  #
  #     def set_bn_eval(m):
  #       classname = m.__class__.__name__
  #       if classname.find('BatchNorm') != -1:
  #         m.eval()
  #
  #     self.RCNN_base.apply(set_bn_eval)
  #     self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    # fc7 = self.RCNN_top(pool5)
    # print('fc7.size()-1', fc7.size())
    # fc7 = fc7.mean(3)
    # print('fc7.size()-2', fc7.size())
    # fc7 = fc7.mean(2)
    # print('fc7.size()-3', fc7.size())
    return fc7


