import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

from model.normalCNN import SimpleCNN # 从 model.py 导入模型
from model.resnet import get_advanced_model # 导入高级模型

# --- 模型选择 ---
# 设置为 True 来加载和测试ResNet18模型
# 设置为 False 来加载和测试SimpleCNN模型
USE_ADVANCED_MODEL = True

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    此函数打印并绘制混淆矩阵。
    """
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def test_model(model, loader, device, num_classes):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    return cm, loader.dataset.classes

if __name__ == '__main__':
    # 1. 设置设备
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    model_name = "Advanced_ResNet18" if USE_ADVANCED_MODEL else "SimpleCNN"
    print(f"Using device: {DEVICE}")
    print(f"Testing model: {model_name}")

    # 2. 路径和参数
    data_dir = 'dataset_split'
    test_dir = os.path.join(data_dir, 'test')
    model_path = 'best_model.pth'
    BATCH_SIZE = 64

    # 3. 数据预处理
    data_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. 加载测试数据集
    test_dataset = ImageFolder(root=test_dir, transform=data_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    num_classes = len(test_dataset.classes)
    print(f"Found {num_classes} classes: {test_dataset.classes}")

    # 5. 加载模型
    if USE_ADVANCED_MODEL:
        model = get_advanced_model(num_classes=num_classes).to(DEVICE)
    else:
        model = SimpleCNN(num_classes=num_classes).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print("Model loaded from best_model.pth")

    # 6. 测试模型并生成混淆矩阵
    cm, class_names = test_model(model, test_loader, DEVICE, num_classes)
    
    # 7. 绘制混淆矩阵
    plot_confusion_matrix(cm, classes=class_names)
