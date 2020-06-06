import joblib
from sklearn.svm import SVR, SVC

from SVM.load_data import load_video_level_segments


def train_svm(train_seg_dir, save_dir, seg_len):
    svm = SVC(kernel='linear',
              gamma='scale',
              tol=1e-3,
              C=1)
    # train_X, train_y = load_video_level_features(train_data_dir, feat_len=1024)
    train_X, train_y = load_video_level_segments(seg_dir=train_seg_dir, seg_len=seg_len)
    svm.fit(train_X, train_y)
    joblib.dump(svm, save_dir)


if __name__ == '__main__':
    train_svm(train_seg_dir='/home/yangzehua/UCF_Crimes/RGB_Picked/train',
              save_dir='SVM_RGB_PICK',
              seg_len=1024)
