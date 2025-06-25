# 实验报告 REPORT.md

## 1. 项目简介

本项目基于 PyTorch 实现手势数字图片的分类，支持自定义 NormalCNN 和迁移学习 ResNet18 两种模型，包含数据处理、训练、测试等完整流程。

## 2. 数据集说明

- 数据来源：Kaggle 手势数字图片数据集
- 预处理：自动下载、解压、去除无关类别，统一尺寸 128x128
- 划分比例：训练集 70%，验证集 15%，测试集 15%

## 3. 实验设置

- 主要依赖：torch、torchvision、scikit-learn、matplotlib、seaborn、opencv-python
- 运行环境：macOS，Python 3.8+
- 主要超参数：
    - batch_size: 64
    - learning_rate: 0.001
    - epochs: 20
    - early_stop_patience: 5
- 数据增强：随机水平翻转、随机旋转、亮度/对比度扰动

## 4. 实验过程

- 数据准备：
    ```bash
    bash scripts/get_data.sh
    ```
- 训练模型：
    ```bash
    bash scripts/train_normal_cnn.sh
    # 或
    bash scripts/train_resnet.sh
    ```
- 测试模型：
    ```bash
    bash scripts/test_normal_cnn.sh
    # 或
    bash scripts/test_resnet.sh
    ```
- 训练过程自动保存验证集最优模型，支持 early stopping。
- 训练和验证损失/准确率曲线自动绘制。

## 5. 实验结果

- NormalCNN 测试集准确率：95.33%
- ResNet18 测试集准确率：98.60%
- 混淆矩阵如下（以 test.py 输出为准）：

## 6. 结果分析

- 大部分数字分类准确率高，部分手势（如 2/3、6/9）易混淆。
- ResNet18 迁移学习模型在小样本下表现更稳健。
- 数据增强和 early stopping 有效防止过拟合。

## 7. 结论与展望

- 本项目实现了高精度的手势数字识别。

