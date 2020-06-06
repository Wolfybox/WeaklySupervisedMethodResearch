import math

import torch
import numpy as np


def ranking_loss_max(y_pred, y_true, n):
    ano_indices = (y_true == 1).nonzero().flatten()
    nor_indices = (y_true == 0).nonzero().flatten()
    ano_pred = y_pred[ano_indices]
    nor_pred = y_pred[nor_indices]
    ano_max_pred = ano_pred.max(1)[0]
    nor_max_pred = nor_pred.max(1)[0]
    rank_loss = 0
    # constant_zero_array = torch.from_numpy(np.array([0] * n)).float().cuda()
    for i in range(n):
        sub = 1 + nor_max_pred - ano_max_pred[i]
        rank_loss += sub.squeeze().sum(0)
        # rank_loss += torch.stack([sub.squeeze(), constant_zero_array], dim=1).max(1)[0].sum(0)
    return rank_loss / (n * n)


def ranking_loss_max_inter_outer(y_pred, y_true, n):
    ano_indices = (y_true == 1).nonzero().flatten()
    nor_indices = (y_true == 0).nonzero().flatten()
    ano_pred = y_pred[ano_indices]
    nor_pred = y_pred[nor_indices]
    ano_max_pred = ano_pred.max(1)[0]
    ano_min_pred = ano_pred.min(1)[0]
    nor_max_pred = nor_pred.max(1)[0]
    rank_loss = 0
    inter_loss = 0
    # constant_zero_array = torch.from_numpy(np.array([0] * n)).float().cuda()
    inter_loss += (1 - ano_max_pred + ano_min_pred).sum(0)
    for i in range(n):
        sub = 1 + nor_max_pred - ano_max_pred[i]
        rank_loss += sub.squeeze().sum(0)
        # rank_loss += torch.stack([sub.squeeze(), constant_zero_array], dim=1).max(1)[0].sum(0)
    return rank_loss / (n * n) + inter_loss / n


def margin_ranking(y_pred, y_true, n, margin_low, margin_high):
    ano_indices = (y_true == 1).nonzero().flatten()
    nor_indices = (y_true == 0).nonzero().flatten()
    ano_pred = y_pred[ano_indices]
    nor_pred = y_pred[nor_indices]
    ano_max_pred = ano_pred.max(1)[0]
    nor_max_pred = nor_pred.max(1)[0]
    ano_loss = margin_high - ano_max_pred
    nor_loss = nor_max_pred - margin_low
    ano_loss[ano_loss < 0] = 0
    nor_loss[nor_loss < 0] = 0
    rank_loss = (ano_loss + nor_loss).sum()
    return rank_loss / (n * 2)


def continuity_loss(y_pred, y_true, n):
    ano_indices = (y_true == 1).nonzero().flatten()
    ano_pred = y_pred[ano_indices]
    cont_loss = 0
    constant_zeros = torch.from_numpy(np.array([0])).float().cuda()
    for i in range(n):
        cur_ano = ano_pred[i].squeeze()
        fast = torch.cat([constant_zeros, cur_ano], dim=0)
        slow = torch.cat([cur_ano, constant_zeros], dim=0)
        sub = (fast - slow)[1:n * 2 - 1]
        cont_loss += (sub * sub).sum(0)
    return cont_loss / n


def sparse_loss(y_pred, y_true, n):
    ano_indices = (y_true == 1).nonzero().flatten()
    ano_pred = y_pred[ano_indices]
    return ano_pred.sum(1).sum(0) / n


def ranking_loss_sum(y_pred, y_true, n):
    ano_indices = (y_true == 1).nonzero().flatten()
    nor_indices = (y_true == 0).nonzero().flatten()
    ano_pred = y_pred[ano_indices]
    nor_pred = y_pred[nor_indices]
    ano_sum_pred = ano_pred.sum(1)
    nor_sum_pred = nor_pred.sum(1)
    rank_loss = 0
    constant_zero_array = torch.from_numpy(np.array([0] * n)).float().cuda()
    for i in range(n):
        sub = 1 + nor_sum_pred - ano_sum_pred[i]
        rank_loss += torch.stack([sub.squeeze(), constant_zero_array], dim=1).max(1)[0].sum(0)
    return rank_loss / (n * n)


def peak_loss(y_pred, y_true, n):
    ano_indices = (y_true == 1).nonzero().flatten()
    ano_pred = y_pred[ano_indices]
    p_loss = 0
    for i in range(n):
        cur_ano = ano_pred[i].squeeze()
        fast = torch.cat([cur_ano, torch.tensor([0]).float().cuda()], dim=0)
        slow = torch.cat([torch.tensor([0]).float().cuda(), cur_ano], dim=0)
        sub = (fast - slow) > 0
        peak = []
        for j in range(1, len(sub) - 1):
            prev = sub[j - 1]
            after = sub[j + 1]
            if prev and not after:
                peak.append(cur_ano[j - 1].item())
        max_indices = peak.index(max(peak))
        del peak[max_indices]
        p_loss += sum([-math.log(1.1 - score) for score in peak])
    return p_loss / n
