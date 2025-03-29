#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
在Google Colab TPU上训练模型的示例脚本
"""

import os
import sys
import torch
import numpy as np
import logging
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tpu_train')

# TPU依赖导入(使用try-except确保在非TPU环境也能运行)
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    HAS_TPU = True
    logger.info("TPU支持已启用")
except ImportError:
    HAS_TPU = False
    logger.warning("找不到TPU支持，将使用CPU/GPU")

def get_device():
    """获取可用的计算设备，优先使用TPU"""
    if HAS_TPU:
        try:
            device = xm.xla_device()
            logger.info(f"使用TPU设备: {device}")
            return device, 'tpu'
        except:
            logger.warning("TPU初始化失败")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"使用GPU设备: {device}")
        return device, 'cuda'
    
    device = torch.device('cpu')
    logger.info("使用CPU设备")
    return device, 'cpu'

# 创建一个简单的模型
class SimpleModel(torch.nn.Module):
    def __init__(self, input_size=1000):
        super(SimpleModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.conv2 = torch.nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(64, 1)
        
    def forward(self, x):
        # 确保输入有正确的维度 [batch, channels, length]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# 创建合成数据集
class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=1000, seq_length=1000):
        self.num_samples = num_samples
        self.seq_length = seq_length
        
        # 生成随机数据
        np.random.seed(42)
        self.data = np.random.randn(num_samples, seq_length).astype(np.float32)
        self.targets = np.random.randn(num_samples, 1).astype(np.float32)
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def train_model_on_tpu():
    """在TPU上训练模型的主函数"""
    # 获取设备
    device, device_type = get_device()
    
    # 创建模型并移动到设备
    model = SimpleModel()
    model = model.to(device)
    
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 创建数据集和数据加载器
    train_dataset = SyntheticDataset(num_samples=1000)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True
    )
    
    # 如果是TPU，使用TPU特定的数据加载器
    if device_type == 'tpu':
        train_loader = pl.MpDeviceLoader(train_loader, device)
    
    # 训练参数
    num_epochs = 5
    criterion = torch.nn.MSELoss()
    
    # 训练循环
    logger.info(f"开始在{device_type}上训练...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for data, targets in train_loader:
            # 将数据移动到设备
            if device_type != 'tpu':  # TPU数据加载器已经处理这一步
                data = data.to(device)
                targets = targets.to(device)
            
            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 如果使用TPU，需要特殊处理
            if device_type == 'tpu':
                xm.optimizer_step(optimizer, barrier=True)
            else:
                optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            # 每10个批次打印一次进度
            if batch_count % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_count}, Loss: {loss.item():.6f}")
        
        # 计算平均损失
        avg_loss = epoch_loss / batch_count
        logger.info(f"Epoch {epoch+1}/{num_epochs} 完成, 平均损失: {avg_loss:.6f}")
    
    # 训练结束
    total_time = time.time() - start_time
    logger.info(f"训练完成! 总时间: {total_time:.2f} 秒")
    
    # 如果在TPU上，保存模型需要特殊处理
    if device_type == 'tpu':
        logger.info("将模型从TPU移回CPU...")
        model = model.cpu()
    
    # 保存模型
    torch.save(model.state_dict(), "tpu_trained_model.pt")
    logger.info("模型已保存到 tpu_trained_model.pt")

if __name__ == "__main__":
    if not HAS_TPU:
        print("\n" + "="*50)
        print("警告: 未检测到TPU支持!")
        print("请确保您在Google Colab TPU环境中运行此脚本")
        print("或先运行setup_tpu_colab.py安装必要依赖")
        print("脚本将继续在CPU/GPU上运行以作示范")
        print("="*50 + "\n")
    
    train_model_on_tpu() 