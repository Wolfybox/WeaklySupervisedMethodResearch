import math
import os

import cv2
import torch
import torchvision
import torchvision.transforms as tf
from PIL import Image


def load_videos(videopath):
    capture = cv2.VideoCapture(videopath)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if 'nor' in videopath:
        frame_count = 5408 if frame_count > 5408 else frame_count
    count = 0
    retaining = True
    frames = []
    while count < frame_count and retaining:
        retaining, frame = capture.read()
        if frame is not None and count % 14 == 0:
            frame = video_tf(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            frames.append(frame)
        count += 1
    capture.release()
    frames = torch.stack(frames)
    return frames


def extract_features(frames, video_name, video_class, video_split):
    frame_num = frames.size()[0]
    batch_num = math.ceil(frame_num / batch_size)
    features = []
    for batch_index in range(batch_num):
        start = batch_index * batch_size
        end = (batch_index + 1) * batch_size
        end = end if end < frame_num else frame_num
        batch_frames = frames[start:end]
        with torch.no_grad():
            batch_feats = vgg_model(batch_frames.cuda()).cpu()
        features.append(batch_feats)
    features = torch.cat(features, dim=0).permute(0, 2, 3, 1)
    feat_dir = '/home/yangzehua/UCF_Crimes/VGG_Features_Trunc'
    feat_dir = os.path.join(feat_dir, video_split, video_class, '{}.pt'.format(video_name))
    torch.save(features, feat_dir)


def get_video_paths(filename):
    with open('data_info/{}.txt'.format(filename), 'r') as f:
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


if __name__ == '__main__':
    batch_size = 16
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    vgg_model = torchvision.models.vgg19(pretrained=True).features[:-1].eval().cuda()
    # print(vgg_model)

    video_tf = tf.Compose([
        tf.Resize((224, 224), Image.CUBIC),
        tf.ToTensor(),
        tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    ano_list, nor_list = get_video_paths(filename='Train_Clean')
    all_list = ano_list + nor_list
    ano_list, nor_list = get_video_paths(filename='Test_Clean')
    all_list += ano_list
    all_list += nor_list

    for vp in all_list:
        vname = vp.split('/')[-1].split('.')[0]
        classname = vp.split('/')[-2]
        videosplit = vp.split('/')[-3]
        v_frames = load_videos(vp)
        extract_features(v_frames, vname, classname, videosplit)
