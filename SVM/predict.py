import os

import joblib
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def predict_segments_scores(seg_dir, model, feat_mode):
    cur_seg = torch.load(seg_dir)
    cur_seg = torch.nn.functional.normalize(cur_seg, p=2, dim=1)
    score = [model.predict(cur_seg[i].view(1, 4096 if feat_mode == 'C3D' else 1024).numpy()) for i in
             range(cur_seg.size()[0])]
    return score


if __name__ == '__main__':
    svm = joblib.load('SVM_C3D')
    seg_dir = '/home/yangzehua/UCF_Crimes/C3D_Segments/test'
    save_dir = '/home/yangzehua/UCF_Crimes/Scores/SVM'
    for name in tqdm(os.listdir(seg_dir)):
        score = predict_segments_scores(seg_dir=os.path.join(seg_dir, name),
                                        model=svm,
                                        feat_mode='C3D')
        x = np.linspace(1, 32, 32)
        y = np.array(score).squeeze()
        plt.cla()
        plt.ylim(-0.1, 1.1)
        plt.plot(x, y)
        plt.savefig(os.path.join(save_dir, name.split('.')[0]))

