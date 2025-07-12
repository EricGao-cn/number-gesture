import torch.nn as nn
from torchvision import models

def get_advanced_model(num_classes=10):
    """
    加载预训练的ResNet18模型，并替换其分类头以适应我们的任务。
    """
    # 加载使用ImageNet预训练的ResNet18模型
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 获取全连接层的输入特征数
    num_ftrs = model.fc.in_features
    
    # 替换最后一个全连接层以匹配我们的10个类别
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

if __name__ == '__main__':
    model = get_advanced_model()
    print("成功创建基于ResNet18的迁移学习模型：")
    print(model)
