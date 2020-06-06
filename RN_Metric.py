import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm

from Vanilla.RN_Vanilla import RNVanilla
from data_util import *


def merge_score(score1, score2):
    mlist = {}
    for name in score1.keys():
        cs1 = score1[name].squeeze()
        cs2 = score2[name].squeeze()
        ms = cs1 * cs2
        mlist[name] = ms
        # mlist[name] = ms if 'Normal' in name else cs2
    return mlist


def svr_predict(seg_dir, model_dir):
    svr_rgb = joblib.load(model_dir)
    svr_score_list = {}
    seg_list = []
    name_list = []
    for segname in os.listdir(seg_dir):
        vname = segname.split('.')[0]
        namecomponents = vname.split('_')[:-1]
        vname = '_'.join(namecomponents)
        seg = torch.load(os.path.join(seg_dir, segname), map_location=torch.device('cpu'))
        seg = torch.nn.functional.normalize(seg, p=2, dim=1)
        # seg = torch.mean(seg, dim=0).view(1024).numpy()
        seg_list.append(seg)
        name_list.append(vname)
    seg_list = np.array(seg_list)
    svr_score = svr_rgb.predict(seg_list)
    for i in range(len(svr_score)):
        svr_score_list[name_list[i]] = svr_score[i]


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    path_dir = '/home/yangzehua/RoadAccidentsDetector/ucf_train_test_info/URAD_Test.txt'
    anno_dir = '/home/yangzehua/RoadAccidentsDetector/ucf_train_test_info/URAD_Annotations.txt'

    # vanilla_c3d = RNVanilla(pretrained=True, model_dir='Vanilla/Vanilla_C3D.pt', input_dim=4096).eval().cuda()
    # vanilla_rgb = RNVanilla(pretrained=True, model_dir='Vanilla/Vanilla_RGB.pt', input_dim=1024).eval().cuda()
    vanilla_flow = RNVanilla(pretrained=True, model_dir='Vanilla/Vanilla_FLOW.pt', input_dim=1024).eval().cuda()
    # test_rgb_dir = '/home/yangzehua/UCF_Crimes/RGB_Segments/test'
    test_flow_dir = '/home/yangzehua/UCF_Crimes/FLOW_Segments/test'
    # test_c3d_dir = '/home/yangzehua/UCF_Crimes/CADP_C3D_Segments/test'
    # vanilla_c3d_score_list = rn_predict(seg_dir=test_c3d_dir, input_dim=4096, net=vanilla_c3d)
    # vanilla_rgb_score_list = rn_predict(seg_dir=test_rgb_dir, input_dim=1024, net=vanilla_rgb)
    # for name in vanilla_rgb_score_list.keys():
    #     score = vanilla_rgb_score_list[name]
    #     vanilla_rgb_score_list[name] = np.array(min_max_normalize(score, mode='min'))
    #
    vanilla_flow_score_list = rn_predict(seg_dir=test_flow_dir, input_dim=1024, net=vanilla_flow)
    for name in vanilla_flow_score_list.keys():
        score = vanilla_flow_score_list[name]
        vanilla_flow_score_list[name] = np.array(min_max_normalize(score, mode='min'))
    # _, _, _, oc_auc = get_roc_metric(score_list=vanilla_c3d_score_list, include_normal=False, path_dir=path_dir,
    #                                  anno_dir=anno_dir)
    # _, _, _, bc_auc = get_roc_metric(score_list=vanilla_c3d_score_list, include_normal=True, path_dir=path_dir,
    #                                  anno_dir=anno_dir)
    # print(oc_auc, bc_auc)

    # _, _, _, oc_auc = get_roc_metric(score_list=vanilla_rgb_score_list, include_normal=False, path_dir=path_dir,
    #                                  anno_dir=anno_dir)
    # _, _, _, bc_auc = get_roc_metric(score_list=vanilla_rgb_score_list, include_normal=True, path_dir=path_dir,
    #                                  anno_dir=anno_dir)
    # print(oc_auc, bc_auc)
    #
    # _, _, _, oc_auc = get_roc_metric(score_list=vanilla_flow_score_list, include_normal=False, path_dir=path_dir,
    #                                  anno_dir=anno_dir)
    # _, _, _, bc_auc = get_roc_metric(score_list=vanilla_flow_score_list, include_normal=True, path_dir=path_dir,
    #                                  anno_dir=anno_dir)
    # print(oc_auc, bc_auc)
    #
    # merge_score_list = merge_score(vanilla_rgb_score_list, vanilla_flow_score_list)
    save_dir = '/home/yangzehua/UCF_Crimes/Scores/FLOW_MINMAX'
    for name in tqdm(vanilla_flow_score_list.keys()):
        x = np.linspace(1, 32, 32)
        y = vanilla_flow_score_list[name]
        plt.cla()
        plt.ylim(-0.1, 1.1)
        plt.plot(x, y)
        plt.savefig(os.path.join(save_dir, name.split('.')[0]))

    # _, _, _, oc_auc = get_roc_metric(score_list=merge_score_list, include_normal=False, path_dir=path_dir,
    #                                  anno_dir=anno_dir)
    # _, _, _, bc_auc = get_roc_metric(score_list=merge_score_list, include_normal=True, path_dir=path_dir,
    #                                  anno_dir=anno_dir)
    # print(oc_auc, bc_auc)
