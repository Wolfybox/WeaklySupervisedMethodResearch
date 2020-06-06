import math
import os

import cv2
import joblib
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from SpaAttenLSTM import SpaAttenLSTM
import torch.nn.utils.rnn as rnn_utils
from SpaAttenLSTM_V2 import SpaAttenLSTM_V2


def load_frame_nums(video_list):
    frame_nums = {}
    for vp in video_list:
        capture = cv2.VideoCapture(vp)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        vname = vp.split('/')[-1].split('.')[0]
        if 'Normal' in vname:
            frame_count = 5408 if frame_count > 5408 else frame_count
        frame_nums[vname] = frame_count
    return frame_nums


def load_temporal_annotations(anno_dir):
    annos = {}
    with open(anno_dir, 'r') as f:
        lines = f.readlines()
        lines = [line.strip().split() for line in lines]
    for line in lines:
        name = line[0].split('.')[0]
        start = int(line[2])
        end = int(line[3])
        annos[name] = (start, end)
    return annos


def get_video_paths(path_dir):
    with open(path_dir, 'r') as f:
        test_videos = f.readlines()
        test_videos = [line.strip() for line in test_videos]
    anomaly_list = []
    normal_list = []
    for videop in test_videos:
        if 'Normal' in videop:
            normal_list.append(videop)
        else:
            anomaly_list.append(videop)
    return anomaly_list, normal_list


def spa_predict(model_dir, video_paths, feat_dir, model=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    if model is None:
        net = SpaAttenLSTM_V2(pretrained=True, model_dir=model_dir).eval().cuda()
    else:
        net = model.eval()
    scorelist = {}
    for vp in video_paths:
        vname = vp.split('/')[-1].split('.')[0]
        classname = vp.split('/')[-2]
        split = vp.split('/')[-3]
        target_dir = os.path.join(feat_dir, split, classname, '{}.pt'.format(vname))
        features = [torch.load(target_dir)]
        feat_len = [len(feat) for feat in features]
        features = rnn_utils.pad_sequence(features, batch_first=True, padding_value=0).cuda()
        with torch.no_grad():
            scores = net.feat_base_predict(features, feat_len)
        scorelist[vname] = scores
    return scorelist


def prepare_spa_score(score_list, frame_num_list, anno_list):
    y_trues = []
    y_preds = []
    for name in score_list.keys():
        y_pred = []
        cur_score = score_list[name][0].cpu().squeeze().numpy().tolist()
        frame_n = frame_num_list[name]
        gnd_start, gnd_end = anno_list[name]
        for i in range(len(cur_score)):
            y_pred += [cur_score[i]] * 14
        y_pred = y_pred[:frame_n]
        y_true = [1 if (gnd_start < i < gnd_end) else 0 for i in range(frame_n)]
        y_preds += y_pred
        y_trues += y_true
    return y_preds, y_trues


def prepare_video_level_score(ano_score_list, nor_score_list):
    y_trues = []
    y_preds = []
    for name in ano_score_list.keys():
        cur_score = ano_score_list[name][0].squeeze().cpu().numpy().tolist()
        y_preds.append(max(cur_score))
        y_trues.append(1)
    for name in nor_score_list.keys():
        cur_score = nor_score_list[name][0].squeeze().cpu().numpy().tolist()
        y_preds.append(max(cur_score))
        y_trues.append(0)
    return y_preds, y_trues


def add_dict(d1, d2):
    result = d1.copy()
    result.update(d2)
    return result


def get_auc(model, include_normal, path_dir, anno_dir):
    anomaly_list, normal_list = get_video_paths(path_dir)
    annotations = load_temporal_annotations(anno_dir)
    frame_nums = load_frame_nums(anomaly_list)
    ano_score_list = spa_predict(model_dir='',
                                 video_paths=anomaly_list,
                                 feat_dir='/home/yangzehua/UCF_Crimes/VGG_Features_Trunc',
                                 model=model)
    nor_score_list = spa_predict(model_dir='',
                                 video_paths=normal_list,
                                 feat_dir='/home/yangzehua/UCF_Crimes/VGG_Features_Trunc',
                                 model=model)
    y_preds, y_trues = prepare_spa_score(score_list=ano_score_list, frame_num_list=frame_nums, anno_list=annotations)
    # y_preds, y_trues = prepare_video_level_score(ano_score_list, nor_score_list)
    fpr, tpr, threshold = roc_curve(y_trues, y_preds)
    roc_auc = auc(fpr, tpr)
    return roc_auc


if __name__ == '__main__':
    anomaly_list, normal_list = get_video_paths('/home/yangzehua/SpaAttenRADetector/data_info/URAD_Test.txt')
    annotations = load_temporal_annotations('/home/yangzehua/SpaAttenRADetector/data_info/URAD_Annotations.txt')
    frame_nums = load_frame_nums(anomaly_list + normal_list)



    ano_score_list = spa_predict(model_dir='/home/yangzehua/SpaAttenRADetector/model/train_v2.pt',
                                 video_paths=anomaly_list,
                                 feat_dir='/home/yangzehua/UCF_Crimes/VGG_Features_Trunc')
    nor_score_list = spa_predict(model_dir='/home/yangzehua/SpaAttenRADetector/model/train_v2.pt',
                                 video_paths=normal_list,
                                 feat_dir='/home/yangzehua/UCF_Crimes/VGG_Features_Trunc')

    y_preds, y_trues = prepare_video_level_score(ano_score_list, nor_score_list)
    fpr, tpr, _ = roc_curve(y_trues, y_preds)
    roc_auc = auc(fpr, tpr)
    # print('VL:{}'.format(roc_auc))

    # y_preds, y_trues = prepare_spa_score(score_list=ano_score_list, frame_num_list=frame_nums, anno_list=annotations)
    # fpr, tpr, _ = roc_curve(y_trues, y_preds)
    # roc_auc = auc(fpr, tpr)
    # print('OC:{}'.format(roc_auc))
    # all_scores_list = spa_predict(model_dir='/home/yangzehua/SpaAttenRADetector/model/train_v2.pt',
    #                              video_paths=anomaly_list + normal_list,
    #                              feat_dir='/home/yangzehua/UCF_Crimes/VGG_Features_Trunc')
    # y_preds, y_trues = prepare_spa_score(score_list=all_scores_list, frame_num_list=frame_nums, anno_list=annotations)
    # fpr, tpr, _ = roc_curve(y_trues, y_preds)
    # roc_auc = auc(fpr, tpr)
    # print('BC:{}'.format(roc_auc))
