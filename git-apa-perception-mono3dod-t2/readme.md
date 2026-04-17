# 高速目标检测项目 (High-Speed Object Detection)

## 项目概述

这是一个用于高速行车场景下的目标检测项目，专门用于实时检测场景中的物体。该项目实现了先进的目标检测算法，支持多种检测任务，包括常规目标检测(ctdet)、3D目标检测(ddd)、和极值点检测(exdet)。

项目主要特点：
- 支持多种神经网络架构（ResNet、DLA、Hourglass等）
- 高性能实时检测能力
- 支持双目摄像头系统
- 集成了COCO API用于评估和数据处理
- 针对高速场景优化的检测算法

## 技术栈

- **深度学习框架**: PyTorch 1.10.1+cu111
- **计算机视觉**: OpenCV 4.2.0.34
- **数据处理**: NumPy, SciPy
- **API集成**: COCO API
- **辅助工具**: Cython, Numba, Progress, Matplotlib, EasyDict

## 项目结构

```
highspeed_OD/
├── src/                   # 源代码目录
│   ├── object_det/        # 核心实现
│   │   ├── detectors/     # 检测器实现
│   │   ├── models/        # 神经网络模型
│   │   ├── datasets/      # 数据集处理
│   │   ├── trains/        # 训练相关代码
│   │   └── utils/         # 工具函数
│   ├── tools/             # 工具脚本
│   ├── main.py            # 主训练入口
│   └── demo.py            # 演示脚本
├── data/                  # 数据目录
│   ├── demo_data/         # 演示数据
│   └── ip42/              # IP42数据集
├── cocoapi/               # COCO API实现
├── exp/                   # 实验输出目录
└── models/                # 预训练模型存储
```

## 安装与配置

### 环境要求

- Python 3.6+
- CUDA 11.1 (用于GPU加速)
- PyTorch 1.10.1+cu111

### 安装步骤

1. 安装依赖包：
```bash
pip install -r requirements.txt
```

2. 编译COCO API：
```bash
cd cocoapi/PythonAPI
make
python setup.py install --user
```


## 使用方法

### 训练模型

```bash
# 基本训练命令
python main.py ddd --exp_id kitti_resnet18 --batch_size 1 --lr 1.25e-4  --gpus 0
```

### 运行演示

```bash
# 对单张图片进行检测
python src/demo.py ddd --demo path/to/image.jpg --load_model path/to/model.pth

# 对视频进行检测
python src/demo.py ddd --demo path/to/video.mp4 --load_model path/to/model.pth

# 使用摄像头实时检测
python src/demo.py ddd --demo webcam --load_model path/to/model.pth
```

### 测试评估

```bash
# 在测试集上评估模型
python src/test.py ddd --exp_id default --dataset kitti --test
```

### 数据准备

```bash
# json数据标注转kitti格式
python src/tools/ip42data2kitti.py

# kitti格式转为模型可载入数据
python src/tools/kitti2anno.py
```

## 配置参数

主要配置参数说明：

- `task`: 任务类型 (ctdet|ddd|exdet)
- `dataset`: 数据集类型 (coco|kitti|coco_hp|pascal)
- `arch`: 网络架构 (res_18|res_101|resdcn_18|resdcn_101|dlav0_34|dla_34|hourglass)
- `gpus`: GPU设备设置
- `batch_size`: 批处理大小
- `lr`: 学习率
- `num_epochs`: 训练轮数
- `input_res`: 输入分辨率
- `load_model`: 预训练模型路径

## 数据格式

项目支持自定义数据集，数据格式需遵循以下规范：

- 图像数据：支持常见格式（jpg、png等）
- 标注数据：JSON格式，包含边界框、类别等信息
- 相机参数：支持双目标定参数（内参矩阵、畸变参数等）

示例标注格式：
```json
{
  "image_path.jpg": {
    "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "box2d": [[x1, y1, x2, y2], ...],
    "box3d": [[x, y, z, w, h, l, yaw], ...]
  }
}
```

## 模型架构

项目支持多种检测模型：

1. **ResNet系列**: ResNet-18, ResNet-101
2. **DLA系列**: DLA-34, DLAV0-34
3. **Hourglass**: 多级沙漏网络

每种模型都针对不同场景进行了优化，可根据精度和速度需求选择。

## 性能优化

- **多GPU训练**: 支持数据并行训练
- **混合精度**: 使用FP16加速训练
- **模型量化**: 支持INT8量化推理
- **批处理优化**: 高效的数据加载和预处理

## 常见问题

1. **CUDA内存不足**: 减小batch_size或使用更小的模型
2. **编译错误**: 确保安装了正确版本的CUDA和PyTorch
3. **检测精度低**: 尝试使用更大的模型或调整训练参数

## 开发指南

添加新功能的基本步骤：

1. 在`object_det/detectors/`中添加新的检测器
2. 在`object_det/models/`中添加新的网络架构
3. 在`object_det/datasets/`中添加数据集处理逻辑
4. 更新`opts.py`中的配置选项

## 贡献

欢迎提交问题和改进建议。在提交代码前，请确保：

1. 代码符合项目风格
2. 添加必要的测试
3. 更新相关文档