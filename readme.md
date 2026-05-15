### 一、数据增强
1、运行unzip.sh得到2万帧鱼眼数据
1、运行offline_fisheye_camera_only_aug.py得到增强后的鱼眼数据。


### 二、训练前的数据预处理
1、将百度数据集转成kitti形式的相机系的真值
运行24CPU_gt_4fisheye_camera_cord.py
2、将kitti形式的相机系的真值转成柱面形式的真值
运行24CPU_fisheye2cyl.py
3、将柱面形式的真值转成模型需要的ann形式
运行24CPU_cyl_kitti2ann.py


运行fisheye2ann.py，输入是5个场景的数据，输出是模型需要的真值ann，1个脚本搞定数据预处理。



### 三、训练命令
在tmux.txt

### 四、测试
1、在.vscode/launch.json将"name": "test.py ddd kitti"的权重路径"--load_model",更换为要测试的权重路径，得到result_lidar，这是lidar系的预测结果
2、运行脚本/指标/gt_lidar_cord_baidu2kitti.py，这是真值的kitti形式
3、运行脚本/指标/recall.py得到测试指标。运行前更改真值和预测值的路径
