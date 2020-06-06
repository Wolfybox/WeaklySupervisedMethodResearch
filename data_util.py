import math
import os

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
from loss_function import ranking_loss_max, sparse_loss, continuity_loss


def mean_std_norm(seg):
    mean = seg.mean(dim=0).unsqueeze(dim=0)
    sub = seg - mean
    std = torch.sqrt((sub * sub).sum(dim=0) / 32).unsqueeze(dim=0)
    return sub / std


def mean_norm(seg):
    mean = seg.mean(dim=0).unsqueeze(dim=0)
    sub = seg - mean
    return sub


def min_max_normalize(scores, mode='zero'):
    if not isinstance(scores, list):
        scores = scores.reshape(32).tolist()
    if mode == 'none':
        return scores
    max_score = max(scores)
    min_score = min(scores) if mode == 'min' else 0
    try:
        norm_scores = [(s - min_score) / (max_score - min_score) for s in scores]
    except Exception:
        print(scores)
        norm_scores = scores
        print('Min Max Error')
    return norm_scores


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
        if 'RoadAccident' in name:
            start = int(line[2])
            end = int(line[3])
            annos[name] = (start, end)
        elif 'Normal' in name:
            annos[name] = (-1, -1)
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


def get_valid_segandscores(frame_num, score, seg_num):
    seg_frame_indices = []
    valid_scores = []
    if frame_num < 16 * seg_num:
        seg_len = 16
        valid_seg_num = math.ceil(frame_num / seg_len)
        score_indices = np.linspace(0, valid_seg_num, seg_num, endpoint=False, dtype=np.int)
        for seg_i in range(valid_seg_num):
            seg_score_indices = np.argwhere(score_indices == seg_i)
            valid_scores.append(score[seg_score_indices[0][0]])
            start = seg_i * seg_len
            end = seg_i * seg_len + seg_len if seg_i * seg_len + seg_len <= frame_num else frame_num
            seg_frame_indices.append((start, end))
    else:
        valid_seg_num = seg_num
        feat_len = 16
        feat_num = math.ceil(frame_num / feat_len)
        feat_indices = np.linspace(0, feat_num, valid_seg_num + 1, endpoint=True, dtype=np.int)
        valid_scores = score.tolist()
        for j in range(valid_seg_num):
            start = feat_indices[j] * feat_len
            end = feat_indices[j + 1] * feat_len if feat_indices[j + 1] * feat_len < frame_num else frame_num
            seg_frame_indices.append((start, end))
    return valid_scores, seg_frame_indices


def prepare_true_pred(anno, valid_score, seg_frame_indice):
    y_pred = []
    valid_seg_num = len(seg_frame_indice)
    total_frame_num = seg_frame_indice[valid_seg_num - 1][1]
    gnd_start, gnd_end = anno
    for i in range(valid_seg_num):
        f_start, f_end = seg_frame_indice[i]
        y_pred += [valid_score[i]] * (f_end - f_start)
    y_true = [(1 if gnd_start <= f_index <= gnd_end else 0) for f_index in range(total_frame_num)]
    return y_true, y_pred


def get_roc_metric(score_list, include_normal, path_dir, anno_dir):
    anomaly_list, normal_list = get_video_paths(path_dir)
    annotations = load_temporal_annotations(anno_dir)
    frame_nums = load_frame_nums(anomaly_list + normal_list)
    y_trues = []
    y_preds = []
    for name in annotations.keys():
        if not include_normal and 'Normal' in name:
            continue
        cur_anno = annotations[name]
        cur_f_num = frame_nums[name]
        cur_score = score_list[name].squeeze()
        valid_scores, seg_frame_indices = get_valid_segandscores(cur_f_num, cur_score, seg_num=32)
        y_true, y_pred = prepare_true_pred(cur_anno, valid_scores, seg_frame_indices)
        y_trues += y_true
        y_preds += y_pred
    fpr, tpr, threshold = roc_curve(y_trues, y_preds)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, threshold, roc_auc


def get_false_alarm_rate(score_list, path_dir):
    anomaly_list, normal_list = get_video_paths(path_dir)
    frame_nums = load_frame_nums(anomaly_list + normal_list)
    y_preds = []
    for name in score_list.keys():
        if 'RoadAccid' in name:
            continue
        cur_anno = (-1, -1)
        cur_f_num = frame_nums[name]
        cur_score = score_list[name].squeeze()
        valid_scores, seg_frame_indices = get_valid_segandscores(cur_f_num, cur_score, seg_num=32)
        _, y_pred = prepare_true_pred(cur_anno, valid_scores, seg_frame_indices)
        y_pred = ((np.array(y_pred) > 0.5) + 0).tolist()
        y_preds += y_pred
    fa_rate = sum(y_preds) / len(y_preds)
    return fa_rate


def get_f1_score(score_list, path_dir, anno_dir):
    anomaly_list, normal_list = get_video_paths(path_dir)
    annos = load_temporal_annotations(anno_dir)
    frame_nums = load_frame_nums(anomaly_list + normal_list)
    y_preds = []
    y_trues = []
    for name in score_list.keys():
        if 'Normal' in name:
            continue
        cur_anno = annos[name]
        cur_f_num = frame_nums[name]
        cur_score = score_list[name].squeeze()
        valid_scores, seg_frame_indices = get_valid_segandscores(cur_f_num, cur_score, seg_num=32)
        y_true, y_pred = prepare_true_pred(cur_anno, valid_scores, seg_frame_indices)
        y_pred = ((np.array(y_pred) > 0.5) + 0).tolist()
        y_preds += y_pred
        y_trues += y_true
    f1 = f1_score(y_true=y_trues, y_pred=y_preds)
    return f1


def rn_predict(seg_dir, input_dim, net, mean_sub=False):
    score_list = {}
    seg_list = []
    name_list = []
    for segname in os.listdir(seg_dir):
        vname = segname.split('.')[0]
        if input_dim in [30, 298, 500, 1024, 2048]:
            namecomponents = vname.split('_')[:-1]
            vname = '_'.join(namecomponents)
        cur_seg = torch.load(os.path.join(seg_dir, segname), map_location=torch.device('cpu')).cuda()
        cur_seg = torch.nn.functional.normalize(cur_seg, p=2, dim=1)
        if mean_sub:
            cur_seg = mean_norm(cur_seg)
        seg_list.append(cur_seg)
        name_list.append(vname)
    seg_list = torch.stack(seg_list, dim=0).cuda()
    with torch.no_grad():
        seg_scores = net.predict(seg_list.float()).cpu().numpy()
    for i in range(seg_scores.shape[0]):
        score_list[name_list[i]] = seg_scores[i]
    return score_list


def pack_test_pred_true(score_list):
    y_pred = []
    y_true = []
    for name in score_list.keys():
        score = torch.tensor(score_list[name])
        label = 0 if 'Normal' in name else 1
        y_pred.append(score)
        y_true.append(label)
    y_pred = torch.stack(y_pred).cuda()
    y_true = torch.tensor(y_true).cuda()
    return y_pred, y_true


def get_test_loss(y_pred, y_true, n, cl_weight, sl_weight):
    rl = ranking_loss_max(y_pred, y_true, n)
    cl = cl_weight * continuity_loss(y_pred, y_true, n)
    sl = sl_weight * sparse_loss(y_pred, y_true, n)
    return rl, cl, sl
