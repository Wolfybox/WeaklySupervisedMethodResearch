import os

from tqdm import tqdm

from AE.test_ae import ae_predict
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    save_dir = '/home/yangzehua/UCF_Crimes/Scores/AE/test'
    model_dir = '/home/yangzehua/RoadAccidentsDetector/AE/AE_C3D.pt'
    seg_dir = '/home/yangzehua/UCF_Crimes/C3D_Segments/test'
    score_list = ae_predict(model=None,
                            model_dir=model_dir,
                            seg_dir=seg_dir,
                            input_dim=4096)
    for name in tqdm(score_list.keys()):
        score = score_list[name]
        x = np.linspace(1, 32, 32)
        y = np.array(score).squeeze()
        plt.cla()
        plt.ylim(-0.1, 1.1)
        plt.plot(x, y)
        plt.savefig(os.path.join(save_dir, name.split('.')[0]))
