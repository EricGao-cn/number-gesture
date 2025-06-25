# 基于卷积神经网络的手势数字识别

本项目是一个基于 PyTorch 框架的卷积神经网络（CNN）手势数字识别项目。旨在通过深度学习技术，实现对 0-9 十个数字手势图像的准确分类。

## 项目背景

随着人机交互需求的不断提高，手势识别作为一种自然、直观的交互方式，在智能家居、虚拟现实、医疗康复等领域展现出巨大的应用潜力。本项目旨在通过搭建、训练和评估一个卷积神经网络，掌握深度学习在图像分类任务中的应用，并达到 80% 以上的分类精度。

## 技术栈

*   **深度学习框架**: PyTorch
*   **主要依赖库**:
    *   `torchvision`: 用于数据预处理和加载。
    *   `scikit-learn`: 用于生成混淆矩阵，评估模型性能。
    *   `matplotlib` & `seaborn`: 用于数据可视化，如损失曲线、准确率曲线和混淆矩阵。
    *   `numpy`: 用于数值计算。
    *   `opencv-python`: 用于图像处理。
    *   `kagglehub`: 用于从 Kaggle 下载数据集。

## 项目结构

```
.
├── data/               # 数据处理与原始数据
│   ├── data_info.py    # 数据分析与划分脚本
│   ├── get_data.py     # 下载与整理数据
│   └── dataset/        # 原始图片
├── dataset_split/      # 按比例划分的训练/验证/测试集
├── model/              # 模型结构定义
│   ├── normalCNN.py
│   └── resnet.py
├── saved_models/       # 训练好的模型权重（normal/、resnet/）
├── scripts/            # 一键运行脚本
├── train.py            # 训练主程序
├── test.py             # 测试主程序
├── README.md           # 项目说明文档
└── REPORT.md           # 实验报告
```

## 环境配置

建议使用虚拟环境。

### 使用 uv (推荐)

安装依赖：
```bash
uv pip install -r requirements.txt
```
或直接根据 `pyproject.toml` 安装：
```bash
uv pip install .
```

### 使用 Conda 

> [!warning]注意
> 如果使用 conda 而不使用 uv，执行脚本之前需要对脚本进行修改, 将 `uv run` 均改成 `python` 或 `python3`

1.  **创建并激活 Conda 环境**:
    ```bash
    conda create -n gesture python=3.12 -y
    conda activate gesture
    ```

2.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```


## 数据准备

1. **下载数据并整理**：
    ```bash
    bash scripts/get_data.sh
    ```
    数据会被自动整理到 `data/dataset/`，并划分为 `dataset_split/train/`、`valid/`、`test/`。

## 训练模型

- 训练普通 CNN：
    ```bash
    bash scripts/train_normal_cnn.sh
    ```
- 训练 ResNet18 迁移学习模型：
    ```bash
    bash scripts/train_resnet.sh
    ```

可自定义参数，例如：
```bash
python train.py --model-name NormalCNN --epochs 30 --batch-size 32 --learning-rate 0.0005 --early-stop-patience 7
```

## 测试模型

- 测试普通 CNN：
    ```bash
    bash scripts/test_normal_cnn.sh
    ```
- 测试 ResNet18：
    ```bash
    bash scripts/test_resnet.sh
    ```

## 主要参数说明

- `--model-name`：选择模型（NormalCNN 或 Advanced_ResNet18）
- `--epochs`：训练轮数
- `--batch-size`：批量大小
- `--learning-rate`：学习率
- `--early-stop-patience`：early stopping 容忍轮数
- `--model-save-dir`：模型保存目录

## 结果与可视化

训练结束后会自动保存最优模型权重，并绘制损失与准确率曲线。测试脚本会输出准确率和混淆矩阵。

## 模型简介

本项目支持两种模型结构：

1.  **NormalCNN**（自定义卷积神经网络）
    *   三个卷积层（conv1/2/3），每层后接 ReLU 和最大池化
    *   两个全连接层（fc1, fc2），fc1 后有 Dropout
2.  **Advanced_ResNet18**（迁移学习）
    *   基于 torchvision 的 ResNet18，最后一层输出数目根据类别自动调整

## 实验结果

经过训练，NormalCNN 在测试集上可达 **95.33%** 分类精度，ResNet18 通常表现更优。具体分类情况可参考 `test.py` 输出的混淆矩阵。

---

如需详细实验过程、超参数对比等，请参见 [REPORT.md](REPORT.md)。
