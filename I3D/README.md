【代码说明】  
I3D_Network.py： I3D网络结构  
InceptionModule.py： I3D网络中的Inception模块  
MaxPool3dSamePadding.py： I3D网络中的3D池化模块  
Unit3D.py： I3D网络中的3D卷积单元  
extract_flow_features.py： 抽取I3D的FLOW模式下特征  
extract_rgb_features.py：抽取I3D的RGB模式下特征  
注：I3D_Network.py需要ImageNet上的预训练模型文件rgb_imagenet.pt和flow_imagenet.pt，同样由于模型过大无法上传。  
