### 一、数据增强
1、运行offline_fisheye_camera_only_aug.py得到增强后的鱼眼数据。
2、visualize_fisheye_gt.py可视化出来增强的鱼眼数据，按ctrl c打断，只要看一部分就行。


### 二、训练前的数据预处理
1、将百度数据集转成kitti形式的相机系的真值
运行脚本\柱面相关的\40CPU_gt_4fisheye_camera_cord.py
2、将kitti形式的相机系的真值转成柱面形式的真值
运行脚本/柱面相关的/40CPU_fisheye2cyl.py
3、将柱面形式的真值转成模型需要的ann形式
运行脚本/柱面相关的/40CPU_cyl_kitti2ann.py

### 三、训练命令
在tmux.txt

### 四、测试
1、在.vscode/launch.json将"name": "test.py ddd kitti"的权重路径"--load_model",更换为要测试的权重路径，得到result_lidar，这是lidar系的预测结果
2、运行脚本/指标/gt_lidar_cord_baidu2kitti.py，这是真值的kitti形式
3、运行脚本/指标/recall.py得到测试指标。运行前更改真值和预测值的路径
