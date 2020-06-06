# WeaklySupervisedMethodResearch
【杨泽华-毕业设计】  

以下是文件说明：  
AE：基于自编码器的正常行为重构的检测方法  
C3D：C3D特征抽取代码  
I3D：I3D特征抽取代码  
SVM：基于SVM和视频级特征学习的检测方法  
SpatialAttention：基于空间自注意力机制和LSTM的帧序列学习方法  
Vanilla：基于多实例学习的深度排序检测算法及其各种特征融合的改进  
auc_result：各个实验得到的AUC结果  
gif：最终端到端检测系统的动态事故打分的GIF示例  
Feature_Vector_Visualize.py：特征向量可视化  
data_util.py：数据加载、段分数到帧分数的转换等辅助功能  
generate_scores_gif.py：产生动态打分的GIF图  
RN_Metric.py：RGB和FLOW模式下模型的AUC测试，同时包含Min-Max后以及外积融合改进后的AUC测试  

注：实验室服务器又又又挂掉了，另外特征文件和训练好的模型文件也比较大，目前没法上传。  

【最终系统动态打分效果】  
![image](https://github.com/Wolfybox/WeaklySupervisedMethodResearch/blob/master/gif/RoadAccidents002_x264.gif)   
![image](https://github.com/Wolfybox/WeaklySupervisedMethodResearch/blob/master/gif/RoadAccidents017_x264.gif)   
![image](https://github.com/Wolfybox/WeaklySupervisedMethodResearch/blob/master/gif/RoadAccidents021_x264.gif)   
![image](https://github.com/Wolfybox/WeaklySupervisedMethodResearch/blob/master/gif/RoadAccidents133_x264.gif)   

