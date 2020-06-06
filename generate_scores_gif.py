import matplotlib.animation as animation

from RN_Metric import *
from tqdm import tqdm
import matplotlib.pyplot as plt

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
            frames.append(frame)
        count += 1
    capture.release()
    return frames


def map_frames2scores(videopath, scores):
    scores = min_max_normalize(scores)
    frames = load_videos(videopath)
    frame_num = len(frames)
    frames_scores_map = []
    if frame_num < 16 * 32:
        seg_len = 16
        seg_num = math.ceil(frame_num / seg_len)
        score_indices = np.linspace(0, seg_num, 32, endpoint=False, dtype=np.int)
        for seg_i in range(seg_num):
            seg_score_indices = np.argwhere(score_indices == seg_i)
            seg_score = scores[seg_score_indices[0][0]]
            start = seg_i * seg_len
            end = seg_i * seg_len + seg_len if seg_i * seg_len + seg_len <= frame_num else frame_num
            frames_scores_map += [(frames[k], seg_score) for k in range(start, end)]
    else:
        seg_num = 32
        feat_len = 16
        feat_num = math.ceil(frame_num / feat_len)
        feat_indices = np.linspace(0, feat_num, seg_num + 1, endpoint=True, dtype=np.int)
        for j in range(seg_num):
            start = feat_indices[j] * feat_len
            end = feat_indices[j + 1] * feat_len if feat_indices[j + 1] * feat_len < frame_num else frame_num
            frames_scores_map += [(frames[k], scores[j]) for k in range(start, end)]
    return frames_scores_map


def init():
    show_video = ax1.imshow(f_s[0][0], animated=True)
    ax2.set_xlim(1, len(f_s))
    ax2.set_ylim(-0.1, 1.1)
    show_scores, = ax2.plot([], [], animated=True)
    return [show_video, show_scores]  # , show_scores


def update(frame_index):
    ax1.cla()
    show_video = ax1.imshow(f_s[frame_index][0], animated=True)
    axis_frame_indices.append(frame_index + 1)
    axis_scores.append(f_s[frame_index][1])
    show_scores, = ax2.plot(axis_frame_indices, axis_scores, animated=True)
    return [show_video, show_scores]


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    path_dir = '/home/yangzehua/RoadAccidentsDetector/ucf_train_test_info/URAD_Test.txt'
    gif_dir = '/home/yangzehua/UCF_Crimes/UCF_GIF/JOINT'
    test_rgb_dir = '/home/yangzehua/UCF_Crimes/RGB_Segments/test'
    test_flow_dir = '/home/yangzehua/UCF_Crimes/FLOW_Segments/test'

    vanilla_rgb = RNVanilla(pretrained=True, model_dir='Vanilla/Vanilla_RGB.pt', input_dim=1024).eval().cuda()
    vanilla_flow = RNVanilla(pretrained=True, model_dir='Vanilla/Vanilla_FLOW.pt', input_dim=1024).eval().cuda()
    vanilla_rgb_score_list = rn_predict(seg_dir=test_rgb_dir, input_dim=1024, net=vanilla_rgb)
    for name in vanilla_rgb_score_list.keys():
        score = vanilla_rgb_score_list[name]
        vanilla_rgb_score_list[name] = np.array(min_max_normalize(score, mode='min'))
    vanilla_flow_score_list = rn_predict(seg_dir=test_flow_dir, input_dim=1024, net=vanilla_flow)
    for name in vanilla_flow_score_list.keys():
        score = vanilla_flow_score_list[name]
        vanilla_flow_score_list[name] = np.array(min_max_normalize(score, mode='min'))
    merge_score_list = merge_score(vanilla_rgb_score_list, vanilla_flow_score_list)
    score_list = {}
    for name in merge_score_list.keys():
        new_name = '_'.join(name.split('.')[0].split('_')[:-1])
        score_list[new_name] = merge_score_list[name]
    anomaly_list, normal_list = get_video_paths(path_dir)
    for vp in tqdm(anomaly_list):
        vname = vp.split('/')[-1].split('.')[0]
        f_s = map_frames2scores(videopath=vp, scores=merge_score_list[vname].reshape(32).tolist())
        indices = np.arange(0, len(f_s))
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        axis_frame_indices = []
        axis_scores = []

        anim = animation.FuncAnimation(fig, update, frames=indices, interval=33, init_func=init, blit=True)
        anim.save(os.path.join(gif_dir, '{}.gif'.format(vname)), writer='imagemagick')
