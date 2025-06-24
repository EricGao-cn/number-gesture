import torch.nn as nn
from torchvision import models

def get_advanced_model(num_classes=10):
    """
    加载预训练的ResNet18模型，并替换其分类头以适应我们的任务。
    """
    # 加载使用ImageNet预训练的ResNet18模型
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 预训练模型中的参数通常已经被很好地训练过
    # 在迁移学习中，可以选择冻结部分或全部卷积层，只训练我们修改的分类层
    # 这里我们选择微调整个网络，所以不对参数进行冻结
    # for param in model.parameters():
    #     param.requires_grad = False

    # 获取全连接层的输入特征数
    num_ftrs = model.fc.in_features
    
    # 替换最后一个全连接层以匹配我们的10个类别
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

if __name__ == '__main__':
    model = get_advanced_model()
    print("成功创建基于ResNet18的迁移学习模型：")
    print(model)
