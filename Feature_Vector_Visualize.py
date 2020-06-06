import os
import torchvision.transforms as tf

import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from Deep.RN_Deep import RNDeep
import seaborn as sns
from TCA import TCA
from sklearn.decomposition import PCA
from Vanilla.RN_Vanilla import RNVanilla
from data_util import get_valid_segandscores
from data_util import mean_std_norm

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


def scatter(x, colors, color_range):
    palette = np.array(sns.color_palette("hls", color_range))

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=20,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    txts = []
    for i in range(0):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=34)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def load_features(feat_dir, count=1):
    features = []
    for featname in os.listdir(feat_dir):
        features.append(torch.load(os.path.join(feat_dir, featname)).numpy())
    return features[:count]


def load_segments(seg_dir, count):
    segments = []
    labels = []
    for segname in os.listdir(seg_dir):
        cur_seg = torch.load(os.path.join(seg_dir, segname))
        cur_seg = torch.nn.functional.normalize(cur_seg, p=2, dim=1)
        # cur_mean = cur_seg.mean(dim=0).unsqueeze(dim=0)
        # cur_seg = cur_seg - cur_mean
        # cur_seg = net.get_fc_layer_output(x=cur_seg, layer_no=3)
        segments.append(cur_seg)
        labels.append(0 if 'Normal' in segname else 1)
    segments = torch.cat(segments[:count], dim=0)
    return segments.cpu().numpy(), labels


def mean_dev_norm(segments):
    segments = segments.reshape(-1, 1024)
    mean = np.mean(segments, axis=0)
    dev = 0
    for i in range(segments.shape[0]):
        cur_seg = segments[i]
        sub = cur_seg - mean
        dev += sub * sub
    std = np.sqrt(dev / segments.shape[0])
    for i in range(segments.shape[0]):
        cur_seg = segments[i]
        segments[i] = (cur_seg - mean) / std
    return segments


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    # net = RNVanilla(pretrained=True, model_dir=os.path.join('Vanilla', 'Vanilla_RGB.pt'),
    #                 input_dim=1024).eval().cuda()
    train_video_num = 254
    test_video_num = 46
    total_video_num = train_video_num + test_video_num
    train_X, train_labels = load_segments('/home/yangzehua/UCF_Crimes/FLOW_LDA/train',
                                          count=train_video_num)
    X = train_X
    # test_X, test_labels = load_segments('/home/yangzehua/UCF_Crimes/FLOW_LDA/test',
    #                                     count=test_video_num)
    # X = test_X
    # train_X = torch.from_numpy(train_X)
    # test_X = torch.from_numpy(test_X)
    # X = []
    # for i in range(test_video_num):
    #     cur_seg = mean_std_norm(test_X[i * 32:(i + 1) * 32])
    #     X.append(cur_seg)
    # X = torch.stack(X, dim=0).view(-1, 1024).numpy()
    # with torch.no_grad():
    #     X = net.get_layer_output(torch.from_numpy(test_X).cuda(), 0).cpu().numpy()
    # X = None
    # tca = TCA(kernel_type='rbf', lamb=1, gamma=1, dim=-1)
    # X = test_X[:32]
    # for i in range(1, test_video_num):
    #     temp = test_X[i * 32:(i + 1) * 32]
    #     newX, new_temp = tca.fit(X, temp)
    #     X = np.concatenate([X, temp], axis=0)
    # test1 = test_X[:32]
    # test2 = test_X[32:64]
    # pca = PCA(n_components=64)
    # test_new1, test_new2 = tca.fit(test1, test2)
    # X = np.concatenate([test1, test2], axis=0)
    # X = np.concatenate([test_new1, test_new2], axis=0)
    # X = pca.fit_transform(X)
    # X = np.concatenate([train_X, test_X], axis=0)
    # y = [0] * 32 + [1] * 32
    y = []
    for k in range(train_video_num):
        cur_labels = [train_labels[k]] * 32
        y += cur_labels
    # # for k in range(test_video_num):
    # #     cur_labels = [k] * 32
    # #     y += cur_labels
    y = np.array(y)
    digits_proj = TSNE(random_state=20200401).fit_transform(X)

    scatter(digits_proj, y, 2)
    foo_fig = plt.gcf()
    # foo_fig.savefig('demo.eps', format='eps', dpi=1000)
    plt.show()
