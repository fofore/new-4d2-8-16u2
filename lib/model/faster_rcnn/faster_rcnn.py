import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

        # Start add by Jie, set the flag to use mobilenetV2 as the backbone network for feature extraction.
        self.dlb = False
        self.neg_rate = 1  # For resample
        # End add

    def forward(self, im_data, im_info, gt_boxes, num_boxes, dl_data):
        # batch_size = im_data.size(0)
        batch_size, c, h, w = im_data.size()

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        start_tic = time.time()
        # Start add by Jie, use mobilenetV2 as the backbone network for feature extraction.
        if self.dlb:
            # padding

            if h % 32 != 0:  # 720*1280 -->736*1280
                m, n = divmod(h, 32)
                ph = int(((m+1)*32-h)/2)
                im_data = F.pad(im_data, (0, 0, ph, ph), "constant", 0)
            if w % 32 != 0:
                m, n = divmod(w, 32)
                pw = int(((m+1)*32-w)/2)
                im_data = F.pad(im_data, (pw, pw, 0, 0), "constant", 0)  # (padLeft, padRight, padTop, padBottom)
            # print('im_data', im_data.size())

            low_level_features = self.RCNN_low_base(im_data) #1/4
            # print('low_level_features', low_level_features.size())

            mid_level_features = self.RCNN_mid_base(low_level_features) #1/8

            base_feat = self.RCNN_base(mid_level_features) #1/16
            # print('base_feat', base_feat.size())

            base_toc = time.time()

            # ----- Do segmentation
            seg_feat = self.RCNN_top(base_feat)
            # print('seg_feat', seg_feat.size())

            # the previous implementation
            # drive_line = self.SegDecoder(seg_feat, low_level_features)

            # print('drive_line', drive_line.size())
            # TODO here we need to pass all the feature into the decoder
            drive_line = self.SegDecoder(low_level_features,
                                         mid_level_features,
                                         base_feat)

            # print("drive line size", drive_line.size())
            if h % 32 != 0:
                drive_line = drive_line[:, :, ph:h+ph, :]
            if w % 32 != 0:
                drive_line = drive_line[:, :, :, pw:h+pw]

            drive_toc = time.time()

        # End add
        else:
            low_level_features = self.RCNN_low_base(im_data)

            # feed image data to base model to obtain base feature map
            base_feat = self.RCNN_base(low_level_features)

            # print('base_feat.size()', base_feat.size())
            # print('drive_line = 0')
            # drive_line = 0

        # ------ No Detection
        """
        rois_label = None
        rois_target = None
        rois_inside_ws = None
        rois_outside_ws = None
        rpn_loss_cls = 0
        rpn_loss_bbox = 0
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        drive_line_loss = 0
        rois = 0
        cls_prob = 0
        bbox_pred = 0
        """
        # ------ End: No Detection

        # ------ With Detection
        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        det_toc = time.time()
        # print('base_time {:.3f}s  driveline {:.3f}s   detection {:.3f}s\r' \
        #                  .format( base_toc - start_tic, drive_toc - base_toc, det_toc - drive_toc))

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)


        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        drive_line_loss = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # Add by Jie, TODO: add resample
            # print('Calc drive line segmentation loss')
            # print('faster rcnn: forward, drive_line.shape, dl_data.shape', drive_line.shape, dl_data.shape)
            neg_rate = 5
            resample = True if neg_rate < 100 else False
            # if resample:  # TODO, use torch instead of numpy
            #     target = dl_data
            #     bs, h, w = target.shape
            #     y_true = target.reshape(-1)
            #     y_true_0_dix = torch.where(y_true == 0)  # ???
            #     num_neg = torch.sum(y_true == 0)
            #     num_pos = torch.sum(y_true == 1)
            #     num_ign = min(max(int(num_neg - neg_rate * num_pos), 0), int((num_neg + num_pos) * 0.95))
            #     inds = torch.multinomial(y_true_0_dix[0], num_ign, replacement=False, out=None)
            #     # inds = np.random.choice(y_true_0_dix[0], num_ign, replace=False)
            #     y_true[inds] = 255  # ignore
            #     y_true = y_true.reshape(bs, h, w)
            #     y_true = torch.from_numpy(y_true).long().cuda()
            # else:
            #     y_true = dl_data

            if resample:
                target = dl_data.cpu().numpy()
                # print('target.shape', target.shape, np.amax(target))
                bs, h, w = target.shape
                y_true = target.reshape(-1)
                y_true_0_dix = np.where(y_true == 0)
                # ---
                num_neg = np.sum(np.array(y_true == 0))
                num_pos = np.sum(np.array(y_true == 1))
                # count = np.bincount(y_true)
                # num_neg = count[0]
                # num_pos = count[1]  # when only have neg sample, count[1] outof index
                # ---
                num_ign = min(max(int(num_neg - neg_rate * num_pos), 0), int((num_neg + num_pos) * 0.95))
                inds = np.random.choice(y_true_0_dix[0], num_ign, replace=False)

                y_true[inds] = 255  # ignore
                y_true = y_true.reshape(bs, h, w)
                y_true = torch.from_numpy(y_true).long().cuda()
            else:
                y_true = dl_data

            drive_line_loss = F.cross_entropy(drive_line, dl_data)

            # drive_line_loss = F.cross_entropy(drive_line, y_true, ignore_index=255)  # TODO: Use Segmentation

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        if self.training:  # for python 2.7
            rpn_loss_cls = torch.unsqueeze(rpn_loss_cls, 0)
            rpn_loss_bbox = torch.unsqueeze(rpn_loss_bbox, 0)
            RCNN_loss_cls = torch.unsqueeze(RCNN_loss_cls, 0)
            RCNN_loss_bbox = torch.unsqueeze(RCNN_loss_bbox, 0)
            drive_line_loss = torch.unsqueeze(drive_line_loss, 0)

        # Drive Line Segmentation
        # print('torch.max(drive_line)', torch.max(drive_line), drive_line.size())
        # ------ END: With Detection

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, drive_line, drive_line_loss

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        #Modify by Minming to keep the parameters
        # self._init_weights()
