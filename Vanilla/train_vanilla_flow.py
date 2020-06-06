import math
import os

import numpy as np
import torch
import torch.nn.modules as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
from util import draw_loss_graph
from dataset import VideoDataset
from Vanilla.RN_Vanilla import RNVanilla
from data_util import *
from loss_function import *
from Vanilla.RN_Vanilla_Flow import RNVanillaFlow

SEED = 1

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, y_true):
        return ranking_loss_max(y_pred, y_true, n) + \
               sparse_loss_weight * sparse_loss(y_pred, y_true, n) + \
               continuity_loss_weight * continuity_loss(y_pred, y_true, n)


if __name__ == '__main__':
    continuity_loss_weight = 8e-5
    sparse_loss_weight = 8e-5
    reg_weight = 1e-3

    learning_rate = 1e-2
    epsilon = 1e-8
    epoch_num = 1000
    batch_size = 120
    n = math.floor(batch_size / 2)
    input_dim = 1024
    seg_num = 32

    os.environ['CUDA_VISIBLE_DEVICES7'] = '5'
    net = RNVanillaFlow(pretrained=False, input_dim=input_dim).train().cuda()
    reg_optimizer = Adam(net.parameters(), lr=learning_rate, eps=epsilon, weight_decay=reg_weight)
    loss_func = CustomLoss().cuda()
    scheduler = CosineAnnealingWarmRestarts(reg_optimizer, T_0=10, T_mult=2, eta_min=1e-5)
    split = 'train'
    seg_dir = '/home/yangzehua/UCF_Crimes/FLOW_Segments'
    anno_dir = '/home/yangzehua/RoadAccidentsDetector/ucf_train_test_info/CADP_Annotations.txt'
    path_dir = '/home/yangzehua/RoadAccidentsDetector/ucf_train_test_info/CADP_Test.txt'
    # test_seg_dir = os.path.join(seg_dir, 'test')
    test_seg_dir = '/home/yangzehua/UCF_Crimes/CADP_FLOW_Segments/test'
    model_save_dir = 'Vanilla_FLOW_CADP.pt'
    graph_save_dir = 'Vanilla_FLOW_CADP.png'

    dataset = VideoDataset(data_dir=seg_dir, split=split)
    video_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)

    loss_list = []
    auc = 0.0
    auc_list = [0]
    for epoch in tqdm(range(epoch_num)):
        epoch_loss = 0
        for batchX, batchY in video_loader:
            batchX = batchX.cuda()
            batchY = batchY.cuda()
            score_pred = net(batchX).cuda()
            batch_loss = loss_func(score_pred, batchY)
            epoch_loss += batch_loss
            reg_optimizer.zero_grad()
            batch_loss.backward()
            reg_optimizer.step()
        print('Epoch:{}/{} Loss:{}'.format(epoch + 1, epoch_num, epoch_loss))
        scheduler.step(epoch_loss)
        loss_list.append(epoch_loss)

        if (epoch + 1) % 10 == 0:
            net.eval()
            score_list = rn_predict(seg_dir=test_seg_dir, input_dim=input_dim, net=net)
            _, _, _, cur_auc = get_roc_metric(score_list=score_list, include_normal=True, anno_dir=anno_dir,
                                              path_dir=path_dir)
            _, _, _, oc_auc = get_roc_metric(score_list=score_list, include_normal=False, anno_dir=anno_dir,
                                             path_dir=path_dir)
            fa_rate = get_false_alarm_rate(score_list=score_list, path_dir=path_dir)
            f1 = get_f1_score(score_list=score_list, anno_dir=anno_dir, path_dir=path_dir)
            test_pred, test_true = pack_test_pred_true(score_list)
            test_rl, test_cl, test_sl = get_test_loss(y_pred=test_pred, y_true=test_true, n=23,
                                                      cl_weight=continuity_loss_weight,
                                                      sl_weight=sparse_loss_weight)
            auc_list.append(cur_auc)
            net.train()
            print('\n BC_AUC:{} OC_AUC:{} FA Rate:{} F1 Score:{}\n'.format(cur_auc, oc_auc, fa_rate, f1))
            print('\n RL:{} CL:{} SL:{} \n'.format(test_rl, test_cl, test_sl))
            torch.save(net.state_dict(), model_save_dir)
            auc = cur_auc

    draw_loss_graph(loss_list, auc_list, save_dir=graph_save_dir)
