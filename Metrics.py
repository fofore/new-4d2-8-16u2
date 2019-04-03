#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from sklearn.metrics import precision_score, recall_score


def get_score_binary(predict, target, ignore=None):
    _bs = predict.shape[0]  # batch size
    # _C = predict.shape[1]  # number of classes
    if len(predict.shape)==4: # (bs, c, h, w)
        # ---- one-hot to index
        predict = np.argmax(predict, axis=1)  # (bs, c, h, w) --> (bs, h, w)
    # # ---- ignore
    # if ignore:
    #     predict[target == ignore] = 0
    #     target[target == ignore] = 0
    # ---- flatten
    b_true = target.reshape(_bs, -1)    # (_bs, 129600)
    b_pred = predict.reshape(_bs, -1)  # (_bs, _C, 129600), 60*36*60=129600

    scores = np.zeros((_bs, 4), dtype=np.float)
    for idx in range(_bs):
        y_true = b_true[idx, :]  # GT
        y_pred = b_pred[idx, :]
        if ignore:
            y_pred = y_pred[y_true != ignore]
            y_true = y_true[y_true != ignore]

        # correct = (predict == target)
        # _acc = correct.sum() / correct.size  # pixel accuracy
        _acc = accuracy_score(y_true, y_pred)
        # ---- default pos_label=1, average='binary',   # labels=[0, 1] pos_label=1,
        # _p, _r, _, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        _p, _r, _, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='binary')
        _iou = 1.0 / (1.0 / _p + 1.0 / _r - 1) if _p else 0  # 1/iou = (tp+fp+fn)/tp = (tp+fp)/tp + (tp+fn)/tp - 1
        scores[idx, :] = _p, _r, _iou, _acc
        # arr_p[idx] = _p
        # arr_r[idx] = _r
        # arr_iou[idx] = _iou
        # arr_acc[idx] = _acc
    # return arr_p, arr_r, arr_iou, arr_acc
    return scores


def get_score(predict, target, ignore=None):
    _bs = predict.shape[0]  # batch size
    _C = predict.shape[1]  # _C = 12
    # ---- one-hot: _bs x _C x 60*36*60 -->  label: _bs x 60*36*60.
    predict = np.argmax(predict, axis=1)
    # ---- check empty
    # if ignore is not None:
    #     predict[ignore == 0] = 0     # 0 empty
    #     ignore = ignore.reshape(_bs, -1)
    # ---- ignore
    # predict[target == 255] = 0
    # target[target == 255] = 0
    # ---- flatten
    target = target.reshape(_bs, -1)    # (_bs, 129600)
    predict = predict.reshape(_bs, -1)  # (_bs, 129600), 60*36*60=129600

    cnt_class = np.zeros(_C, dtype=np.int32)  # count for each class
    iou_sum = np.zeros(_C, dtype=np.float32)  # sum of iou for each class
    tp_sum = np.zeros(_C, dtype=np.int32)  # tp
    fp_sum = np.zeros(_C, dtype=np.int32)  # fp
    fn_sum = np.zeros(_C, dtype=np.int32)  # fn

    acc = 0.0

    for idx in range(_bs):
        y_true = target[idx, :]  # GT
        y_pred = predict[idx, :]
        # print('y_true.shape, y_pred.shape', y_true.shape, y_pred.shape)
        # y_pred = y_pred[y_true != 255]  # ---- ignore
        # y_true = y_true[y_true != 255]
        # print('y_true.shape, y_pred.shape', y_true.shape, y_pred.shape)
        if ignore:
            y_pred = y_pred[y_true != ignore]
            y_true = y_true[y_true != ignore]
        # if ignore is not None:
        #     nonempty_idx = ignore[idx, :]
        #     # y_pred = y_pred[nonempty_idx == 1]
        #     # y_true = y_true[nonempty_idx == 1]
        #     y_pred = y_pred[np.where(np.logical_and(nonempty_idx == 1, y_true != 255))]  # 去掉需ignore的点
        #     y_true = y_true[np.where(np.logical_and(nonempty_idx == 1, y_true != 255))]
            # print('y_true.shape, y_pred.shape', y_true.shape, y_pred.shape)
        acc += accuracy_score(y_true, y_pred)  # pixel accuracy
        for j in range(_C):  # for each class
            tp = np.array(np.where(np.logical_and(y_true == j, y_pred == j))).size
            fp = np.array(np.where(np.logical_and(y_true != j, y_pred == j))).size
            fn = np.array(np.where(np.logical_and(y_true == j, y_pred != j))).size
            u_j = np.array(np.where(y_true == j)).size
            cnt_class[j] += 1 if u_j else 0
            # iou = 1.0 * tp/(tp+fp+fn) if u_j else 0
            # iou_sum[j] += iou
            iou_sum[j] += 1.0*tp/(tp+fp+fn) if u_j else 0  # iou = tp/(tp+fp+fn)

            tp_sum[j] = tp
            fp_sum[j] = fp
            fn_sum[j] = fn

    acc = acc / _bs
    # return acc, iou_sum, cnt_class
    return acc, iou_sum, cnt_class, tp_sum, fp_sum, fn_sum