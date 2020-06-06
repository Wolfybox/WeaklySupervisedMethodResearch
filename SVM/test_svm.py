import math
import os

import joblib
import torch
import numpy as np
from SVM.load_data import load_video_level_segments
from data_util import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def predict_segments_scores(seg_dir, model_dir, feat_mode):
    svm = joblib.load(model_dir)
    score_list = {}
    for segname in os.listdir(seg_dir):
        vname = segname.split('.')[0]
        if feat_mode != 'C3D':
            namecomponents = vname.split('_')[:-1]
            vname = '_'.join(namecomponents)
        cur_seg = torch.load(os.path.join(seg_dir, segname))
        cur_seg = torch.nn.functional.normalize(cur_seg, p=2, dim=1)
        score = [svm.predict(cur_seg[i].view(1, 4096 if feat_mode == 'C3D' else 1024).numpy()) for i in
                 range(cur_seg.size()[0])]
        score_list[vname] = np.array(score)
    return score_list


def test_frame_level(anno_list, frame_num_list, score_list, include_normal, save_dir):
    y_trues = []
    y_preds = []
    for name in anno_list.keys():
        if not include_normal and 'Normal' in name:
            continue
        cur_anno = anno_list[name]
        cur_f_num = frame_num_list[name]
        cur_score = score_list[name]
        valid_scores, seg_frame_indices = get_valid_segandscores(cur_f_num, cur_score, seg_num=32)
        y_true, y_pred = prepare_true_pred(cur_anno, valid_scores, seg_frame_indices)
        y_trues += y_true
        y_preds += y_pred
    fpr, tpr, threshold = roc_curve(y_trues, y_preds)
    roc_auc = auc(fpr, tpr)
    test_res = {
        'fpr': fpr,
        'tpr': tpr,
        'th': threshold,
        'auc': roc_auc
    }
    print('Frame Level:{}'.format(roc_auc))
    np.save(save_dir, test_res)


def test_video_level(test_data_dir, model_dir, save_dir, feat_mode):
    test_X, test_y = load_video_level_segments(test_data_dir, seg_len=4096 if feat_mode == 'C3D' else 1024)
    svm = joblib.load(model_dir)
    scores = svm.predict(test_X)
    fpr, tpr, threshold = roc_curve(test_y, scores)
    roc_auc = auc(fpr, tpr)
    test_res = {
        'fpr': fpr,
        'tpr': tpr,
        'th': threshold,
        'auc': roc_auc
    }
    print('Video Level:{}'.format(roc_auc))
    np.save(save_dir, test_res)


if __name__ == '__main__':
    test_data_dir = '/home/yangzehua/UCF_Crimes/FLOW_Segments/test'
    mode = 'FLOW'
    model_dir = 'SVM_{}'.format(mode)

    test_video_level(test_data_dir=test_data_dir, model_dir=model_dir, save_dir='{}_Video_Level.npy'.format(mode),
                     feat_mode=mode)

    ano_list, nor_list = get_video_paths(
        path_dir='/home/yangzehua/RoadAccidentsDetector/ucf_train_test_info/URAD_Test.txt')
    annos = load_temporal_annotations(
        anno_dir='/home/yangzehua/RoadAccidentsDetector/ucf_train_test_info/URAD_Annotations.txt')
    fnums = load_frame_nums(ano_list + nor_list)
    rgb_scores = predict_segments_scores(seg_dir=test_data_dir, model_dir=model_dir, feat_mode=mode)
    test_frame_level(anno_list=annos, frame_num_list=fnums, score_list=rgb_scores, include_normal=False,
                     save_dir='{}_Frame_Level_OC.npy'.format(mode))
    test_frame_level(anno_list=annos, frame_num_list=fnums, score_list=rgb_scores, include_normal=True,
                     save_dir='{}_Frame_Level_BC.npy'.format(mode))

    # test_data_dir = '/home/yangzehua/UCF_Crimes/C3D_Segments/test'
    # path_dir = '/home/yangzehua/RoadAccidentsDetector/ucf_train_test_info/URAD_Test.txt'
    # anno_dir = '/home/yangzehua/RoadAccidentsDetector/ucf_train_test_info/URAD_Annotations.txt'
    # mode = 'C3D'
    # model_dir = 'SVM_{}'.format(mode)
    #
    # test_video_level(test_data_dir=test_data_dir, model_dir=model_dir, save_dir='{}_Video_Level.npy'.format(mode),
    #                  feat_mode=mode)

    # ano_list, nor_list = get_video_paths(path_dir=path_dir)
    # annos = load_temporal_annotations(anno_dir=anno_dir)
    # fnums = load_frame_nums(ano_list + nor_list)
    # c3d_scores = predict_segments_scores(seg_dir=test_data_dir, model_dir=model_dir, feat_mode=mode)
    # test_frame_level(anno_list=annos, frame_num_list=fnums, score_list=c3d_scores, include_normal=True,
    #                  save_dir='{}_Frame_Level_BC.npy'.format(mode))
    # test_frame_level(anno_list=annos, frame_num_list=fnums, score_list=c3d_scores, include_normal=False,
    #                  save_dir='{}_Frame_Level_OC.npy'.format(mode))

    # test_frame_level(anno_list=annos, frame_num_list=fnums, score_list=rgb_scores, include_normal=True,
    #                  save_dir='{}_Frame_Level_BC.npy'.format(mode))

    # test_data_dir = '/home/yangzehua/UCF_Crimes/FLOW_Segments/test'
    # mode = 'FLOW'
    # model_dir = 'SVM_{}'.format(mode)
    #
    # test_video_level(test_data_dir=test_data_dir, model_dir=model_dir, save_dir='{}_Video_Level.npy'.format(mode),
    #                  feat_mode=mode)
    #
    # ano_list, nor_list = get_video_paths()
    # annos = load_temporal_annotations()
    # fnums = load_frame_nums(ano_list + nor_list)
    # flow_scores = predict_segments_scores(seg_dir=test_data_dir, model_dir=model_dir, feat_mode=mode)
    # test_frame_level(anno_list=annos, frame_num_list=fnums, score_list=flow_scores, include_normal=False,
    #                  save_dir='{}_Frame_Level_OC.npy'.format(mode))
    # test_frame_level(anno_list=annos, frame_num_list=fnums, score_list=flow_scores, include_normal=True,
    #                  save_dir='{}_Frame_Level_BC.npy'.format(mode))
    #
    # joint_scores = {}
    # for vname in rgb_scores.keys():
    #     rgb = rgb_scores[vname]
    #     flow = flow_scores[vname]
    #     joint = rgb * flow
    #     joint_scores[vname] = joint
    # test_frame_level(anno_list=annos, frame_num_list=fnums, score_list=joint_scores, include_normal=False,
    #                  save_dir='JOINT_Frame_Level_OC.npy'.format(mode))
    # test_frame_level(anno_list=annos, frame_num_list=fnums, score_list=joint_scores, include_normal=True,
    #                  save_dir='JOINT_Frame_Level_BC.npy'.format(mode))
