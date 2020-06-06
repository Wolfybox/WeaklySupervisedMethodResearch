import os

import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, data_dir, split):
        print('Initialize VideoDataset...')
        self.data_dir = os.path.join(data_dir, split)
        self.nor_video_paths = self.__get_video_path_list()
        self.nor_num = len(self.nor_video_paths)
        print('Initialization Done.')

    def __len__(self):
        return self.nor_num

    def __getitem__(self, index):
        video_path = self.nor_video_paths[index]
        seg = torch.load(video_path, map_location=torch.device('cpu'))
        seg = torch.nn.functional.normalize(seg, p=2, dim=1)
        return seg

    def __get_video_path_list(self):
        video_paths = []
        for videoname in os.listdir(self.data_dir):
            video_paths.append(os.path.join(self.data_dir, videoname))
        nor_video_paths = []
        for pth in video_paths:
            if 'Normal' in pth:
                nor_video_paths.append(pth)
        return nor_video_paths


if __name__ == '__main__':
    pass
