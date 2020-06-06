【代码说明】  
RN_Vanilla.py：基于多实例学习的深度排序回归网络（MIL Ranking Model，下简称MRM）  
RN_Vanilla_CAT.py：MRM的拼接特征融合版本  
RN_Vanilla_Joint.py：MRM的隐层特征融合版本  
dataset.py：训练MRM时加载数据的视频段特征  
train_vanilla_c3d.py：基于C3D特征训练MRM  
train_vanilla_cat.py：基于拼接融合后的特征训练MRM  
train_vanilla_flow.py：基于I3D的FLOW模式特征训练MRM  
train_vanilla_joint.py：基于隐层融合后的特征训练MRM  
train_vanilla_rgb.py：基于I3D的RGB模式特征训练MRM  
