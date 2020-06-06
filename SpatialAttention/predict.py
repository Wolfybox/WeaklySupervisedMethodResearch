import os

from tqdm import tqdm

from UCF_Frame_Base_ROC import get_video_paths, spa_predict, load_temporal_annotations, load_frame_nums
import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':
    anomaly_list, normal_list = get_video_paths('/home/yangzehua/SpaAttenRADetector/data_info/URAD_Test.txt')
    annotations = load_temporal_annotations('/home/yangzehua/SpaAttenRADetector/data_info/URAD_Annotations.txt')
    all_video_paths = anomaly_list + normal_list
    frame_nums = load_frame_nums(all_video_paths)
    all_score_list = spa_predict(model_dir='/home/yangzehua/SpaAttenRADetector/model/train_v2.pt',
                                 video_paths=all_video_paths,
                                 feat_dir='/home/yangzehua/UCF_Crimes/VGG_Features_Trunc',
                                 model=None)
    save_dir = '/home/yangzehua/UCF_Crimes/Scores/SpaAtten'
    for name in tqdm(all_score_list.keys()):
        score = all_score_list[name].squeeze().cpu()
        score_len = score.size()[0]
        x = np.linspace(1, score_len, score_len)
        y = score.numpy()
        plt.cla()
        plt.ylim(-0.1, 1.1)
        plt.plot(x, y)
        plt.savefig(os.path.join(save_dir, name.split('.')[0]))


