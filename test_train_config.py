#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试train函数是否能正确处理不同的配置结构
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_train_config')

# 导入模型和train函数
try:
    from model import SpectralResCNN, SpectralResCNN_GCN, train
    logger.info("成功导入model模块")
except ImportError as e:
    logger.error(f"导入model模块失败: {e}")
    raise

# 创建一个简单的模型和数据集
def create_test_data(batch_size=32, input_size=100):
    """创建测试数据"""
    # 创建随机数据
    X = np.random.randn(100, 1, input_size).astype(np.float32)
    y = np.random.randn(100, 1).astype(np.float32)
    
    # 划分训练集和验证集
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]
    
    # 创建PyTorch数据集
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size
    )
    
    return train_loader, val_loader

def test_train_with_training_key():
    """测试使用'training'键的配置"""
    logger.info("测试使用'training'键的配置")
    
    # 创建测试数据
    train_loader, val_loader = create_test_data()
    
    # 创建模型
    input_size = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpectralResCNN(input_size=input_size).to(device)
    
    # 创建配置（使用'training'键）
    config = {
        'training': {
            'lr': 0.001,
            'weight_decay': 1e-4,
            'num_epochs': 2,  # 使用较少的轮次以加快测试
            'early_stopping_patience': 5
        },
        'model_config': {
            'model_dir': 'models',
            'use_gcn': False
        }
    }
    
    # 测试训练函数
    try:
        train_losses, val_losses = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            element="test_training_key"
        )
        logger.info("使用'training'键的配置测试成功")
        return True
    except Exception as e:
        logger.error(f"使用'training'键的配置测试失败: {e}")
        return False

def test_train_with_training_config_key():
    """测试使用'training_config'键的配置"""
    logger.info("测试使用'training_config'键的配置")
    
    # 创建测试数据
    train_loader, val_loader = create_test_data()
    
    # 创建模型
    input_size = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpectralResCNN(input_size=input_size).to(device)
    
    # 创建配置（使用'training_config'键）
    config = {
        'training_config': {
            'lr': 0.001,
            'weight_decay': 1e-4,
            'num_epochs': 2,  # 使用较少的轮次以加快测试
            'early_stopping_patience': 5
        },
        'model_config': {
            'model_dir': 'models',
            'use_gcn': False
        }
    }
    
    # 测试训练函数
    try:
        train_losses, val_losses = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            element="test_training_config_key"
        )
        logger.info("使用'training_config'键的配置测试成功")
        return True
    except Exception as e:
        logger.error(f"使用'training_config'键的配置测试失败: {e}")
        return False

if __name__ == "__main__":
    logger.info("开始测试train函数对不同配置结构的处理")
    
    # 测试使用'training'键的配置
    result1 = test_train_with_training_key()
    
    # 测试使用'training_config'键的配置
    result2 = test_train_with_training_config_key()
    
    # 输出总结
    if result1 and result2:
        logger.info("所有测试都成功通过！train函数现在可以处理两种不同的配置结构。")
    else:
        logger.error("测试失败！请检查train函数的修改。") 