import os

import cv2
import numpy as np

import torch


def load_video_level_features(feat_dir, feat_len):
    data = []
    labels = []
    for featname in os.listdir(feat_dir):
        feat = torch.load(os.path.join(feat_dir, featname), map_location=torch.device('cpu'))
        feat = torch.mean(feat, dim=0).view(feat_len).numpy()
        data.append(feat)
        labels.append(0 if 'Normal' in featname else 1)
    return np.array(data), np.array(labels)


def load_video_level_segments(seg_dir, seg_len):
    data = []
    labels = []
    for segname in os.listdir(seg_dir):
        seg = torch.load(os.path.join(seg_dir, segname), map_location=torch.device('cpu'))
        seg = torch.nn.functional.normalize(seg, p=2, dim=1)
        seg = torch.mean(seg, dim=0).view(seg_len).numpy()
        data.append(seg)
        labels.append(0 if 'Normal' in segname else 1)
    return np.array(data), np.array(labels)

