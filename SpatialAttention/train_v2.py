import os

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from SpaAttenLSTM_V2 import SpaAttenLSTM_V2
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn

from VideoDataset import VideoDataset
from UCF_Frame_Base_ROC import get_auc


def collate_fn(batch_data):
    batch_data.sort(key=lambda x: len(x[0]), reverse=True)
    batch_X = [data[0] for data in batch_data]
    batch_Y = torch.stack([data[1] for data in batch_data]).squeeze()
    batch_X_len = [len(sq) for sq in batch_X]
    batch_X = rnn_utils.pad_sequence(batch_X, batch_first=True, padding_value=0)
    return batch_X, batch_X_len, batch_Y


def custom_optimizer(network, lr):
    weight_p, bias_p = [], []
    for name, p in network.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    # print(network.named_parameters)
    optimizer = torch.optim.Adam([
        {'params': weight_p, 'weight_decay': regularization_weight},
        {'params': bias_p, 'weight_decay': 0}
    ], lr=lr, eps=epsilon)
    return optimizer


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    batch_size = 10
    epoch_num = 200
    regularization_weight = 1e-4
    epsilon = 1e-8
    learning_rate = 1e-3
    model_save_dir = os.path.join('/home/yangzehua/SpaAttenRADetector', 'model', 'train_trunc.pt')
    video_dataset = VideoDataset(data_dir=os.path.join('/home/yangzehua/UCF_Crimes', 'VGG_Features_Trunc'),
                                 split='train')
    video_loader = DataLoader(dataset=video_dataset,
                              batch_size=batch_size,
                              num_workers=8,
                              shuffle=True,
                              drop_last=True,
                              collate_fn=collate_fn)
    net = SpaAttenLSTM_V2(pretrained=False, model_dir='').train().cuda()
    loss_func = nn.MSELoss().cuda()
    reg_optimizer = custom_optimizer(net, learning_rate)
    scheduler = ReduceLROnPlateau(optimizer=reg_optimizer, mode='min', factor=0.1, patience=40, min_lr=1e-5,
                                  verbose=True)

    loss_list = []
    auc = 0.0
    for epoch in tqdm(range(epoch_num)):
        epoch_loss = 0
        for batchX, batchXlen, batchY in video_loader:
            batchX = batchX.cuda()
            # batchXlen = batchXlen.cuda()
            batchY = batchY.cuda().float()
            pred = net(batchX, batchXlen).cuda().squeeze()
            batch_loss = loss_func(pred, batchY).cuda()
            epoch_loss += batch_loss
            reg_optimizer.zero_grad()
            batch_loss.backward()
            reg_optimizer.step()
        print('Epoch:{}/{} Loss:{}'.format(epoch + 1, epoch_num, epoch_loss))
        scheduler.step(epoch_loss)
        loss_list.append(epoch_loss)
        # if (epoch + 1) % 10 == 0:
            #     auc = get_auc(model=net)
            #     net = net.train()
            # print('AUC:{}'.format(auc))
        #     torch.save(net.state_dict(), model_save_dir)
