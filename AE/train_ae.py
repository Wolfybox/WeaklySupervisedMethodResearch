import math
import os

import numpy as np
import torch
import torch.nn.modules as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.adam import Adam
from AE.dataset_normal import VideoDataset
from AE.TemporalRegularityDetector import TemporalRegularityDetector
from data_util import get_roc_metric
from AE.test_ae import ae_predict
from AE.test_ae import ae_predict_video_level, get_vl_auc

SEED = 1

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    regularization_weight = 1e-5
    learning_rate = 1e-2
    epsilon = 1e-8
    epoch_num = 200
    batch_size = 120
    n = math.floor(batch_size / 2)
    input_dim = 1024
    seg_num = 32

    test_path_dir = '/home/yangzehua/RoadAccidentsDetector/ucf_train_test_info/URAD_Test.txt'
    anno_dir = '/home/yangzehua/RoadAccidentsDetector/ucf_train_test_info/URAD_Annotations.txt'

    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    net = TemporalRegularityDetector(pretrained=False, input_dim=input_dim).train().cuda()
    reg_optimizer = Adam(net.parameters(), lr=learning_rate, eps=epsilon, weight_decay=regularization_weight)
    loss_func = nn.MSELoss().cuda()
    # scheduler = ReduceLROnPlateau(optimizer=reg_optimizer, mode='min', factor=0.1, patience=20, min_lr=1e-5,
    #                               verbose=True)
    scheduler = CosineAnnealingWarmRestarts(reg_optimizer, T_0=10, T_mult=2, eta_min=1e-5)

    split = 'train'
    seg_dir = '/home/yangzehua/UCF_Crimes/FLOW_Segments'
    test_seg_dir = os.path.join(seg_dir, 'test')
    model_save_dir = 'AE_FLOW.pt'

    dataset = VideoDataset(data_dir=seg_dir, split=split)
    video_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)

    loss_list = []
    auc = 0.0
    for epoch in tqdm(range(epoch_num)):
        epoch_loss = 0
        for batchX in video_loader:
            batchX = batchX.cuda()
            reconX = net(batchX).cuda()
            batch_loss = loss_func(reconX, batchX)
            epoch_loss += batch_loss
            reg_optimizer.zero_grad()
            batch_loss.backward()
            reg_optimizer.step()
        print('Epoch:{}/{} Loss:{}'.format(epoch + 1, epoch_num, epoch_loss))
        scheduler.step(epoch_loss)
        loss_list.append(epoch_loss)

        if (epoch + 1) % 3 == 0:
            score_list = ae_predict(model=net, model_dir='', seg_dir=test_seg_dir, input_dim=input_dim)
            vl_scores = ae_predict_video_level(model=net, model_dir='', seg_dir=test_seg_dir, input_dim=input_dim)
            _, _, _, vl_auc = get_vl_auc(vl_scores)
            _, _, _, bc_auc = get_roc_metric(score_list=score_list, include_normal=True, path_dir=test_path_dir,
                                             anno_dir=anno_dir)
            _, _, _, oc_auc = get_roc_metric(score_list=score_list, include_normal=False, path_dir=test_path_dir,
                                             anno_dir=anno_dir)
            print('VL:{} BC:{} OC:{}'.format(vl_auc, bc_auc, oc_auc))
            net.train()
            torch.save(net.state_dict(), model_save_dir)
