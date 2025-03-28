import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from config import Config
from preprocessdata7 import LAMOSTDataset

class LAMOSTNet(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(LAMOSTNet, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        return self.features(x)

def train_model(model, train_loader, val_loader, test_loader, device, 
                epochs=100, learning_rate=0.001):
    """训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        device: 设备
        epochs: 训练轮数
        learning_rate: 学习率
    
    Returns:
        history: 训练历史
    """
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': []
    }
    
    # 训练模型
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_idx, (wavelength, flux, labels) in enumerate(train_loader):
            wavelength, flux, labels = wavelength.to(device), flux.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(flux)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for wavelength, flux, labels in val_loader:
                wavelength, flux, labels = wavelength.to(device), flux.to(device), labels.to(device)
                outputs = model(flux)
                val_loss += criterion(outputs, labels).item()
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)
        
        # 测试阶段
        test_loss = 0
        with torch.no_grad():
            for wavelength, flux, labels in test_loader:
                wavelength, flux, labels = wavelength.to(device), flux.to(device), labels.to(device)
                outputs = model(flux)
                test_loss += criterion(outputs, labels).item()
            test_loss /= len(test_loader)
            history['test_loss'].append(test_loss)
        
        # 打印训练进度
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Test Loss: {test_loss:.4f}')
        print('-' * 50)
    
    return history 

def create_data_loaders(train_data, val_data, test_data, batch_size=Config.TRAINING_BATCH_SIZE, num_workers=4):
    """创建数据加载器"""
    train_dataset = LAMOSTDataset(train_data)
    val_dataset = LAMOSTDataset(val_data)
    test_dataset = LAMOSTDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader 