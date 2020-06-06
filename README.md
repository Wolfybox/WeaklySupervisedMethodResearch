# WeaklySupervisedMethodResearch
【杨泽华-毕业设计】  
=========
【简介】
----
本课题为哈工大（深圳）毕业设计，主要基于弱标签视频数据实现监控视频中的交通事故检测，从三种思路实现弱监督交通事故检测，完成算法的定性和定量分析，并基于多实例学习的深度排序回归网络实现最终的交通事故检测，完成端到端的检测系统。

【主要研究成果】
---
（1）本文实现了基于三种不同特征学习思路的弱监督交通事故检测方法—— 基于SVM 和视频级整体特征的学习方法，基于唯正常视频与自编码器的特征重构方法以及基于空间自注意力机制和LSTM 网络的帧序列学习方法。接着我们验证了它们在URAD 上的性能，并从定性和定量的角度分析了这些算法的利弊。对于其中基于空间注意力机制的检测算法，我们提出使用隔帧采样和Ranking的方式进行改进，提高了其双类上的检测性能。  
（2）通过定性分析和定量实验，本文指出在弱监督交通视频事故检测任务中使用AUC 进行评估时双类到单类所存在的数值落差，并由此提出单类AUC 的评测方法，来更好地刻画算法对于视频中事故位置的定位能力。  
（3）本文分别基于C3D 特征与I3D 的RGB 和FLOW 模式下的特征实现了基于多实例学习的深度排序算法，接着验证了几种常见的特征融合手段；然后，我们通过结合Min-Max 归一化和外积的方式成功融合了RGB 和FLOW 特征下的预测分数，并得到了更高的单类AUC 性能。  
（4）本文基于外积融合后的检测方法实现了一个可用于视频交通事故检测的可视化的端到端系统，其检测速度达43 FPS。  


【项目结构】
--- 
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

注：特征文件和训练好的模型文件比较大，目前无法上传。  

【最终系统动态打分效果】 
--
![image](https://github.com/Wolfybox/WeaklySupervisedMethodResearch/blob/master/gif/RoadAccidents002_x264.gif)   
![image](https://github.com/Wolfybox/WeaklySupervisedMethodResearch/blob/master/gif/RoadAccidents017_x264.gif)   
![image](https://github.com/Wolfybox/WeaklySupervisedMethodResearch/blob/master/gif/RoadAccidents021_x264.gif)   
![image](https://github.com/Wolfybox/WeaklySupervisedMethodResearch/blob/master/gif/RoadAccidents133_x264.gif)   

