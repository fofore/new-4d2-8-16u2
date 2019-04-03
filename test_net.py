#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.mobilenet import mobilenet
from model.faster_rcnn.shufflenet import shufflenet

from scipy.misc import imread, imresize
from scipy import misc
import Metrics
from PIL import Image

import datetime

import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="models",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=10021, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    args = parser.parse_args()
    return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "vg":
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    #Start Add by Minming Qian, 27-09-2018
    elif args.dataset == "kiktech_20181001":
        args.imdb_name = "kiktech_20181001_trainval"
        args.imdbval_name = "kiktech_20181001_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    # kik000001 ignore also test on the original test dataset
    elif args.dataset == "kiktech_ignore":
        args.imdb_name = "kiktech_2018_trainval"
        args.imdbval_name = "kiktech_2018_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "kiktech_2018joint-480p-147":
        args.imdb_name = "kiktech_2018joint-480p-147_trainval"
        args.imdbval_name = "kiktech_2018joint-480p-147_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '4']
    elif args.dataset == "kiktech_2019jointd1":
        args.imdb_name = "kiktech_2018joint-480p-147_trainval"
        args.imdbval_name = "kiktech_2019jointd1_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '4']
    elif args.dataset == "kiktech_2019jointd2":
        args.imdb_name = "kiktech_2018joint-480p-147_trainval"
        args.imdbval_name = "kiktech_2019jointd2_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '4']
    elif args.dataset == "kiktech_2019jointd3":
        args.imdb_name = "kiktech_2018joint-480p-147_trainval"
        args.imdbval_name = "kiktech_2019jointd3_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '4']
    else:
        print(args.dataset)
        raise Exception('Dataset name error')
    #End Add by Minming Qian


    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'mobilenet':
        fasterRCNN = mobilenet(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'shufflenet':
        fasterRCNN = shufflenet(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']


    print('load model successfully!')
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    dl_data = torch.LongTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        dl_data = dl_data.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    dl_data = Variable(dl_data)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        fasterRCNN.cuda()


    max_per_image = 100

    vis = args.vis

    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

    save_name = 'faster_rcnn_10'
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                             imdb.num_classes, training=False, normalize = False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))

    total_scores = None
    start = time.time()
    for i in range(num_images):
        print(i)

        data = next(data_iter)
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])
        dl_data.data.resize_(data[4].size()).copy_(data[4])

        # Mod: by Jie, add evaluation of segmentation
        all_tic = time.time()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, drive_line, drive_line_loss = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, dl_data)
        all_toc = time.time()
        all_time = all_toc - all_tic


        # print('dl_data.size()', dl_data.size())
        # print('drive_line.shape', drive_line.shape)
        # print('im_data.size()', im_data.size())

        # ----------------- Evaluate Segmentation ------------------------------------------
        """
        #   Common the rest codes when testing  FPS
        """

        evaseg_tic = time.time()
        im = cv2.imread(imdb.image_path_at(i))

        # y_pred = y_pred.cpu().data.numpy()
        y_pred = drive_line.cpu().data.numpy()
        _idx = 0
        bs, c, h, w = drive_line.shape
        y_pred_idx = np.argmax(y_pred[_idx,], axis=0)  # one-hot: (C, H, W)-->  label: (H, W)
        hs, ws, cs = im.shape
        # print(im.shape)

        y_pred_idx = y_pred_idx.astype(np.uint8)  # This step is very important
        y_pred_idx = imresize(y_pred_idx, (hs, ws), interp='nearest')

        # seg_gt_index = os.path.basename(im_file).split(".")[0]
        # seg_gt_filename = os.path.join(args.data_path, 'PNGSegmentationMask', seg_gt_index + '.png')
        # gt_png = misc.imread(seg_gt_filename)
        # gt_png = gt_png.astype(np.uint8)
        y_true_idx = dl_data[0,0,].cpu().numpy()
        y_true_idx = y_true_idx.astype(np.uint8)
        y_true_idx = imresize(y_true_idx, (hs, ws), interp='nearest')  # BGR


        # print('y_pred_idx.shape, y_true_idx.shape', y_pred_idx.shape, y_true_idx.shape)

        # ---- get mask
        # vis_seg = True
        vis_seg = vis
        if vis_seg:
            mask_result = np.zeros((hs, ws, 3), dtype=np.uint8)
            tp = np.where(np.logical_and(y_true_idx == 1, y_pred_idx == 1))
            # print(tp.sum())
            # False Positive （假正, FP）被模型预测为正的负样本；可以称作误报率
            fp = np.where(np.logical_and(y_true_idx == 0, y_pred_idx == 1))
            # False Negative（假负 , FN）被模型预测为负的正样本；可以称作漏报率
            fn = np.where(np.logical_and(y_true_idx == 1, y_pred_idx == 0))
            # 颜色顺序为RGB
            # mask_result[tp[0], tp[1], :] = 0, 255, 0  # 正确，Green
            # mask_result[fp[0], fp[1], :] = 0, 0, 255  # 误报，Blue
            # mask_result[fn[0], fn[1], :] = 255, 0, 0  # 漏报率，Red
            # 颜色顺序为BGR
            mask_result[tp[0], tp[1], :] = 0, 255, 0  # 正确，Green
            mask_result[fp[0], fp[1], :] = 255, 0, 0  # 误报，Blue
            mask_result[fn[0], fn[1], :] = 0, 0, 255  # 漏报率，Red

            # ---- show evaluation mask
            # cv2.imwrite('result_mask_{}.jpg'.format(i), mask_result)

            # ---- show mix
            # im_mix = cv2.addWeighted(im, 1, mask_result, 0.4, 0)
            # cv2.imwrite('result_mix_{}.jpg'.format(i), im_mix)

            # ---- show perdict mask
            # mask_pred = np.zeros((hs, ws, 3), dtype=np.uint8)
            # id = np.where(y_pred_idx == 1)  # only lane marker
            # mask_pred[id[0], id[1], :] = 0, 255, 0  #
            # cv2.imwrite('result_pred_{}.jpg'.format(i), mask_pred)

            # exit(0)
        evaseg_toc = time.time()

        # ---- get score
        batch_scores = Metrics.get_score_binary(predict=y_pred_idx[np.newaxis, :, :], target=y_true_idx[np.newaxis, :, :],
                                                ignore=255)
        if total_scores is not None:
            total_scores = np.append(total_scores, batch_scores, axis=0)
        else:
            total_scores = batch_scores

        # ----------------- End: Evaluate Segmentation ------------------------------------------
        # End mod

        # ----------------- Evaluate Detection ------------------------------------------
        det_tic = time.time()
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data[1][0][2].item()


        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        if vis:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)
        for j in xrange(1, imdb.num_classes):
            inds = torch.nonzero(scores[:,j]>thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:,j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        det_toc = time.time()
        det_time = det_toc - det_tic

        evaseg_time = evaseg_toc - evaseg_tic

        sys.stdout.write('im_detect: {:d}/{:d} nntime {:.3f}s  visdet {:.3f}s  visseg {:.3f}s  \r' \
                         .format(i + 1, num_images, all_time, det_time,  evaseg_time))
        sys.stdout.flush()

        if vis:
            # cv2.imwrite('result.png', im2show)

            im_mix = cv2.addWeighted(im2show, 1, mask_result, 0.4, 0)
            cv2.imwrite('result_mix_{}.jpg'.format(i), im_mix)

            # pdb.set_trace()
            #cv2.imshow('test', im2show)
            #cv2.waitKey(0)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    deteva_tic = time.time()
    imdb.evaluate_detections(all_boxes, output_dir)
    deteva_toc = time.time()
    print("evaluation time: {:.3f}s".format(deteva_toc - deteva_tic))


    print('Evaluating segmentation')
    scores = np.sum(total_scores, axis=0) / total_scores.shape[0] * 100.0
    print('Scores: P {:.2f}, R {:.2f}, IoU {:.2f}, Acc {:.2f}'.format(scores[0], scores[1], scores[2], scores[3]))
    # ----------------- End: Evaluate Detection ------------------------------------------

    end = time.time()

    #
    print("test time: %0.4fs" % (end - start))
    print("FPS: %0.4f" % (num_images/(end - start)))
