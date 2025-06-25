import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from model.normalCNN import NormalCNN
from model.resnet import get_advanced_model # 导入高级模型
import matplotlib.pyplot as plt
import os
import argparse

def main(args):
    # 1. 设置超参数和设备
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    model_name = args.model_name
    print(f"Using device: {DEVICE}")
    print(f"Using model: {model_name}")

    # 数据集路径
    data_dir = args.data_dir
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    # 2. 数据预处理和加载
    # 为训练集增加数据增强
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.RandomRotation(10),     # 随机旋转-10到10度
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # 随机调整亮度和对比度
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证集和测试集不需要数据增强
    valid_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    valid_dataset = ImageFolder(root=valid_dir, transform=valid_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 模型、损失函数和优化器
    if model_name == 'Advanced_ResNet18':
        model = get_advanced_model(num_classes=len(train_dataset.classes)).to(DEVICE)
    else:
        model = NormalCNN(num_classes=len(train_dataset.classes)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. 训练和验证函数
    def train_one_epoch(model, loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    # 5. 训练主循环
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    patience = args.early_stop_patience
    patience_counter = 0

    # 创建保存模型的目录
    model_save_dir = args.model_save_dir
    os.makedirs(model_save_dir, exist_ok=True)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, valid_loader, criterion, DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} -> "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 根据不同模型保存不同的权重文件
            model_save_path = os.path.join(model_save_dir, f'best_model_{model_name}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path} with validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Early stop patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    print("Training finished.")

    # 6. 绘制结果
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def get_args():
    parser = argparse.ArgumentParser(description='Train a model for number gesture classification.')
    parser.add_argument('--model-name', type=str, default='Advanced_ResNet18',
                        choices=['NormalCNN', 'Advanced_ResNet18'],
                        help='The model to use for training.')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train for.')
    parser.add_argument('--data-dir', type=str, default='dataset_split', help='Directory containing the dataset.')
    parser.add_argument('--model-save-dir', type=str, default='saved_models', help='Directory to save the best model.')
    parser.add_argument('--early-stop-patience', type=int, default=5, help='Early stopping patience (epochs without improvement).')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
