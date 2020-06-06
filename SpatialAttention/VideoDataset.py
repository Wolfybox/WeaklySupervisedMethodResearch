from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import torch
import cv2
import numpy as np


class VideoDataset(Dataset):
    def __init__(self, data_dir, split):
        print('Initialize VideoDataset...')
        self.data_dir = os.path.join(data_dir, split)
        self.ano_video_paths, self.nor_video_paths = self.__get_video_path_list()
        self.ano_num = len(self.ano_video_paths)
        self.nor_num = len(self.nor_video_paths)
        self.get_anomaly = True
        print('Initialization Done.')

    def __len__(self):
        return self.ano_num + self.nor_num

    def __getitem__(self, index):
        if self.get_anomaly:
            idx = np.random.randint(self.ano_num)
            video_path = self.ano_video_paths[idx]
        else:
            idx = np.random.randint(self.nor_num)
            video_path = self.nor_video_paths[idx]
        label = 1 if self.get_anomaly else 0
        self.get_anomaly = not self.get_anomaly
        features = torch.load(video_path)
        return features, torch.tensor(label)

    def __get_video_path_list(self):
        video_paths = []
        for classname in os.listdir(self.data_dir):
            classdir = os.path.join(self.data_dir, classname)
            for videoname in os.listdir(classdir):
                video_paths.append(os.path.join(classdir, videoname))
        ano_video_paths = []
        nor_video_paths = []
        for pth in video_paths:
            if 'Normal' in pth:
                nor_video_paths.append(pth)
            else:
                ano_video_paths.append(pth)
        return ano_video_paths, nor_video_paths


if __name__ == '__main__':
    split = 'train'
    feat_dir = '/home/yangzehua/RADetection/features'
    dataset = VideoDataset(data_dir=feat_dir, split=split)
    video_loader = DataLoader(dataset=dataset, batch_size=60, num_workers=1, shuffle=False)
    for segs, label in tqdm(video_loader):
        print(segs.size(), label)
