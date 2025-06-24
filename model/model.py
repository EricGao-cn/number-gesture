import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 动态计算全连接层输入大小
        # 假设输入图像大小为 128x128
        # 经过3次池化后，大小变为 128 -> 64 -> 32 -> 16
        self.fc1_input_dim = 64 * 16 * 16 
        
        self.fc1 = nn.Linear(self.fc1_input_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, self.fc1_input_dim) # 展平
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == '__main__':
    # 创建一个模型实例
    model = SimpleCNN(num_classes=10)
    
    # 创建一个虚拟的输入张量 (batch_size, channels, height, width)
    # 假设输入图像大小为 128x128
    dummy_input = torch.randn(64, 3, 128, 128)
    
    # 前向传播
    output = model(dummy_input)
    
    # 打印模型结构和输出形状
    print(model)
    print(f"\nOutput shape: {output.shape}") # 应该是 (64, 10)
