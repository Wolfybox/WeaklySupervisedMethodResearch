import os

import cv2
import torch
from sklearn.metrics import roc_curve, auc
from AE.TemporalRegularityDetector import TemporalRegularityDetector
from data_util import *


def ae_predict(model, model_dir, seg_dir, input_dim=1024):
    net = TemporalRegularityDetector(pretrained=True, input_dim=input_dim,
                                     model_dir=model_dir).eval().cuda() if model is None else model.eval().cuda()
    score_list = {}
    seg_list = []
    name_list = []
    for segname in os.listdir(seg_dir):
        vname = segname.split('.')[0]
        if input_dim in [298, 1024, 2048]:
            namecomponents = vname.split('_')[:-1]
            vname = '_'.join(namecomponents)
        cur_seg = torch.load(os.path.join(seg_dir, segname), map_location=torch.device('cpu')).cuda()
        seg_list.append(torch.nn.functional.normalize(cur_seg, p=2, dim=1))
        name_list.append(vname)
    seg_list = torch.stack(seg_list, dim=0)
    with torch.no_grad():
        recon_seg_list = net(seg_list)
    recon_err = ((seg_list - recon_seg_list) ** 2).sum(dim=-1).sqrt()
    # recon_err_max = recon_err.max(dim=-1)[0]
    # recon_err_min = recon_err.min(dim=-1)[0]
    recon_err_max = recon_err.max(dim=-1)[0].max(dim=0)[0]
    recon_err_min = recon_err.min(dim=-1)[0].min(dim=0)[0]
    for i in range(recon_err.size()[0]):
        cur_recon_err = recon_err[i]
        # cur_max_err = recon_err_max[i]
        # cur_min_err = recon_err_min[i]
        # cur_score = (cur_recon_err - cur_min_err) / cur_max_err
        cur_score = (cur_recon_err - recon_err_min) / (recon_err_max - recon_err_min)
        cur_score = cur_score.cpu().numpy()
        score_list[name_list[i]] = cur_score
    return score_list


def ae_predict_video_level(model, model_dir, seg_dir, input_dim=1024):
    net = TemporalRegularityDetector(pretrained=True, input_dim=input_dim,
                                     model_dir=model_dir).eval().cuda() if model is None else model.eval().cuda()
    score_list = {}
    seg_list = []
    name_list = []
    for segname in os.listdir(seg_dir):
        vname = segname.split('.')[0]
        if input_dim in [298, 1024, 2048]:
            namecomponents = vname.split('_')[:-1]
            vname = '_'.join(namecomponents)
        cur_seg = torch.load(os.path.join(seg_dir, segname), map_location=torch.device('cpu')).cuda()
        seg_list.append(torch.nn.functional.normalize(cur_seg, p=2, dim=1))
        name_list.append(vname)
    seg_list = torch.stack(seg_list, dim=0)
    with torch.no_grad():
        recon_seg_list = net(seg_list)
    recon_err = ((seg_list - recon_seg_list) ** 2).sum(dim=-1).sqrt()
    # recon_err_max = recon_err.max(dim=-1)[0]
    # recon_err_min = recon_err.min(dim=-1)[0]
    recon_err_max = recon_err.max(dim=-1)[0].max(dim=0)[0]
    recon_err_min = recon_err.min(dim=-1)[0].min(dim=0)[0]
    for i in range(recon_err.size()[0]):
        cur_recon_err = recon_err[i]
        # cur_max_err = recon_err_max[i]
        # cur_min_err = recon_err_min[i]
        cur_score = (cur_recon_err - recon_err_min) / (recon_err_max - recon_err_min)
        cur_score = cur_score.cpu().numpy()
        score_list[name_list[i]] = np.max(cur_score)
    return score_list


def get_vl_auc(score_list):
    y_preds = []
    y_trues = []
    for name in score_list.keys():
        y_trues.append(0 if 'Normal' in name else 1)
        y_preds.append(score_list[name])
    fpr, tpr, threshold = roc_curve(y_trues, y_preds)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, threshold, roc_auc


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    input_dim = 1024
    net = TemporalRegularityDetector(input_dim=input_dim, model_dir='AE_FLOW.pt', pretrained=True).eval().cuda()
    test_seg_dir = '/home/yangzehua/UCF_Crimes/FLOW_Segments/test'
    test_path_dir = '/home/yangzehua/RoadAccidentsDetector/ucf_train_test_info/URAD_Test.txt'
    anno_dir = '/home/yangzehua/RoadAccidentsDetector/ucf_train_test_info/URAD_Annotations.txt'
    score_list = ae_predict(model=net, model_dir='', seg_dir=test_seg_dir, input_dim=input_dim)
    vl_scores = ae_predict_video_level(model=net, model_dir='', seg_dir=test_seg_dir, input_dim=input_dim)
    _, _, _, vl_auc = get_vl_auc(vl_scores)
    _, _, _, bc_auc = get_roc_metric(score_list=score_list, include_normal=True, path_dir=test_path_dir,
                                     anno_dir=anno_dir)
    _, _, _, oc_auc = get_roc_metric(score_list=score_list, include_normal=False, path_dir=test_path_dir,
                                     anno_dir=anno_dir)
    print('VL:{} BC:{} OC:{}'.format(vl_auc, bc_auc, oc_auc))
