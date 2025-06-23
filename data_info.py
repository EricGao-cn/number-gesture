import os
import cv2
import numpy as np
from collections import Counter
import shutil
import random
from sklearn.model_selection import train_test_split

class DatasetInfo:
    def __init__(self, data_path='data'):
        self.data_path = data_path
        self.classes = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
        
    def analyze_images(self):
        """分析图像基本信息"""
        print("=" * 50)
        print("数据集图像信息分析")
        print("=" * 50)
        
        all_sizes = []
        all_channels = []
        class_counts = {}
        
        for class_name in self.classes:
            class_path = os.path.join(self.data_path, class_name)
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            class_counts[class_name] = len(images)
            
            print(f"\n类别 {class_name}: {len(images)} 张图像")
            
            # 分析前几张图像的信息
            for i, img_name in enumerate(images[:5]):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    height, width, channels = img.shape
                    all_sizes.append((height, width))
                    all_channels.append(channels)
                    
                    if i == 0:  # 只打印第一张图像的详细信息
                        print(f"  示例图像: {img_name}")
                        print(f"  尺寸: {height} x {width}")
                        print(f"  通道数: {channels}")
                        print(f"  数据类型: {img.dtype}")
        
        # 统计信息
        print("\n" + "=" * 50)
        print("整体统计信息")
        print("=" * 50)
        print(f"总类别数: {len(self.classes)}")
        print(f"总图像数: {sum(class_counts.values())}")
        print(f"各类别图像数量: {class_counts}")
        
        if all_sizes:
            size_counter = Counter(all_sizes)
            print(f"图像尺寸分布: {dict(size_counter.most_common(5))}")
            
        if all_channels:
            channel_counter = Counter(all_channels)
            print(f"通道数分布: {dict(channel_counter)}")
    
    def split_dataset(self, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, output_dir='dataset_split'):
        """将数据集分割为训练集、验证集和测试集"""
        assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须等于1"
        
        print("\n" + "=" * 50)
        print("数据集分割")
        print("=" * 50)
        print(f"训练集比例: {train_ratio}")
        print(f"验证集比例: {valid_ratio}")
        print(f"测试集比例: {test_ratio}")
        
        # 创建输出目录
        for split in ['train', 'valid', 'test']:
            for class_name in self.classes:
                os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
        
        total_train, total_valid, total_test = 0, 0, 0
        
        for class_name in self.classes:
            class_path = os.path.join(self.data_path, class_name)
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # 随机打乱
            random.shuffle(images)
            
            # 计算分割点
            n_total = len(images)
            n_train = int(n_total * train_ratio)
            n_valid = int(n_total * valid_ratio)
            n_test = n_total - n_train - n_valid
            
            # 分割数据
            train_images = images[:n_train]
            valid_images = images[n_train:n_train + n_valid]
            test_images = images[n_train + n_valid:]
            
            # 复制文件
            for img_list, split_name in [(train_images, 'train'), (valid_images, 'valid'), (test_images, 'test')]:
                for img_name in img_list:
                    src = os.path.join(class_path, img_name)
                    dst = os.path.join(output_dir, split_name, class_name, img_name)
                    shutil.copy2(src, dst)
            
            total_train += len(train_images)
            total_valid += len(valid_images)
            total_test += len(test_images)
            
            print(f"类别 {class_name}: 训练集{len(train_images)}, 验证集{len(valid_images)}, 测试集{len(test_images)}")
        
        print(f"\n总计: 训练集{total_train}, 验证集{total_valid}, 测试集{total_test}")
        print(f"数据已保存到: {output_dir}")

if __name__ == "__main__":
    # 设置随机种子
    random.seed(42)
    
    # 创建数据集信息对象
    dataset_info = DatasetInfo('data')
    
    # 分析图像信息
    dataset_info.analyze_images()
    
    # 分割数据集
    dataset_info.split_dataset(train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15)