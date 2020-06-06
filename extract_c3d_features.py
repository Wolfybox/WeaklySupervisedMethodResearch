import math

import cv2
import torchvision.transforms as tf
from C3D.C3D_Network import C3D
# from C3D_Network import C3D
import torch
from tqdm import tqdm
import os
from PIL import Image

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

video_tf = tf.Compose([
    tf.Resize((112, 112), Image.CUBIC),
    tf.ToTensor(),
    # tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

clip_len = 16
batch_size = 128
clip_per_batch = math.floor(batch_size / clip_len)
device = torch.device('cuda:2')
net = C3D(pretrained=True, mode='eval').eval().to(device)


def extract_feat_by_batch(partial_frames):
    # partial frames shape : batch_size x 112 x 112 x 3
    partial_clips = torch.stack(partial_frames)
    partial_clips = partial_clips.view(-1, clip_len, 112, 112, 3).permute(0, 4, 1, 2, 3).float().to(device)
    # batch features shape: clip_num(batch_size / clip_len) x 4096
    with torch.no_grad():
        batch_feats = net(partial_clips)
    return batch_feats


def load_videos(videopath):
    capture = cv2.VideoCapture(videopath)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if 'Normal' in videopath:
        frame_count = 5408 if frame_count > 5408 else frame_count
    count = 0
    retaining = True
    frames = []
    while count < frame_count and retaining:
        retaining, frame = capture.read()
        if frame is not None:
            frame = video_tf(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            frames.append(frame)
        count += 1
    capture.release()
    return frames


def extract_c3d_features(videopath, mode='save'):
    video_name = videopath.split('/')[-1].split('.')[0]
    split = videopath.split('/')[-3]
    print('Loading Frames For {}'.format(video_name))
    frames = load_videos(videopath)
    print('Extract Features For {}'.format(video_name))
    frame_num = len(frames)
    clip_num = math.ceil(frame_num / clip_len)
    total_frame = clip_len * clip_num
    remain = total_frame - frame_num
    frames += [frames[frame_num - 1]] * remain
    batch_num = math.floor(total_frame / batch_size)
    feats = []
    # Calculate features by batch
    for batch_index in range(batch_num):
        start = batch_index * batch_size
        end = batch_index * batch_size + batch_size
        feats.append(extract_feat_by_batch(partial_frames=frames[start:end]))
    # Calculate last batch
    if batch_size * batch_num != total_frame:
        start = batch_num * batch_size
        end = total_frame
        feats.append(extract_feat_by_batch(partial_frames=frames[start: end]))
    feats = torch.cat(feats, dim=0)
    if mode == 'save':
        torch.save(feats, os.path.join(feat_dir, split, '{}.pt'.format(video_name)))
    elif mode == 'output':
        return feats, frame_num


if __name__ == '__main__':
    feat_dir = '/home/yangzehua/UCF_Crimes/C3D_Features'
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    net = C3D(pretrained=True, mode='eval').eval().cuda()

    # feat_dir = '/home/yangzehua/UCF_Crimes/UCF_RoadAccident_C3D_Features_Clean'

    # with open('ucf_train_test_info/Train_Clean.txt', 'r') as f:
    #     train_videos = f.readlines()
    #     train_videos = [line.strip() for line in train_videos]
    # with open('ucf_train_test_info/Test_Clean.txt', 'r') as f:
    #     test_videos = f.readlines()
    #     test_videos = [line.strip() for line in test_videos]
    # videolist = train_videos + test_videos
    videolist = []
    root_dir = '/home/yangzehua/UCF_Crimes/URAD'
    for split in os.listdir(root_dir):
        split_dir = os.path.join(root_dir, split)
        for classname in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, classname)
            for vname in os.listdir(class_dir):
                videolist.append(os.path.join(class_dir, vname))
    # for video in os.listdir(root_dir):
    #     videolist.append(os.path.join(root_dir, video))
    finished = []
    for vp in tqdm(videolist):
        try:
            extract_c3d_features(vp, net=net)
            finished.append(vp)
        except Exception:
            with open('c3d_features.txt', 'w') as f:
                for fin in finished:
                    f.write('{}\n'.format(fin))
