import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from datetime import datetime
from dataset import LAMOSTDataset
import wandb

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_dir, element):
    """加载训练、验证和测试数据"""
    # 加载训练集
    train_data = np.load(os.path.join(data_dir, f'train_dataset_{element}.npz'))
    X_train = train_data['X']
    y_train = train_data['y']
    
    # 加载验证集
    val_data = np.load(os.path.join(data_dir, f'val_dataset_{element}.npz'))
    X_val = val_data['X']
    y_val = val_data['y']
    
    # 加载测试集
    test_data = np.load(os.path.join(data_dir, f'test_dataset_{element}.npz'))
    X_test = test_data['X']
    y_test = test_data['y']
    
    # 创建数据集对象
    train_dataset = LAMOSTDataset([{'data': x, 'label': y} for x, y in zip(X_train, y_train)])
    val_dataset = LAMOSTDataset([{'data': x, 'label': y} for x, y in zip(X_val, y_val)])
    test_dataset = LAMOSTDataset([{'data': x, 'label': y} for x, y in zip(X_test, y_test)])
    
    return train_dataset, val_dataset, test_dataset

def train_model(model, train_loader, val_loader, config, device):
    """训练模型"""
    # 初始化优化器和损失函数
    optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss()
    
    # 记录最佳模型
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 训练循环
    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_idx, (spectra, labels) in enumerate(train_loader):
            spectra, labels = spectra.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(spectra)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 记录每个批次的损失
            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
                wandb.log({
                    'batch': epoch * len(train_loader) + batch_idx,
                    'train_batch_loss': loss.item()
                })
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for spectra, labels in val_loader:
                spectra, labels = spectra.to(device), labels.to(device)
                outputs = model(spectra)
                val_loss += criterion(outputs, labels).item()
        
        val_loss /= len(val_loader)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录每个epoch的损失
        logger.info(f'Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 早停
        if patience_counter >= config['patience']:
            logger.info(f'Early stopping at epoch {epoch}')
            break
    
    return best_model_state, best_val_loss

def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for spectra, labels in test_loader:
            spectra, labels = spectra.to(device), labels.to(device)
            outputs = model(spectra)
            test_loss += criterion(outputs, labels).item()
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    
    test_loss /= len(test_loader)
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # 计算评估指标
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    r2 = 1 - np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2)
    
    return {
        'test_loss': test_loss,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions,
        'actuals': actuals
    }

def main():
    # 配置参数
    config = {
        'data_dir': 'processed_data',
        'element': 'CA_FE',
        'batch_size': 32,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'epochs': 100,
        'patience': 15,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # 初始化wandb
    wandb.init(
        project="lamost-abundance",
        config=config,
        name=f"{config['element']}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # 加载数据
    train_dataset, val_dataset, test_dataset = load_data(config['data_dir'], config['element'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # 创建模型
    input_size = next(iter(train_loader))[0].shape[1]  # 获取输入维度
    model = CNNModel(input_size).to(config['device'])
    
    # 训练模型
    best_model_state, best_val_loss = train_model(
        model, train_loader, val_loader, config, config['device']
    )
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    # 评估模型
    results = evaluate_model(model, test_loader, config['device'])
    
    # 记录最终结果
    logger.info("\n=== 最终评估结果 ===")
    logger.info(f"测试集损失: {results['test_loss']:.6f}")
    logger.info(f"MSE: {results['mse']:.6f}")
    logger.info(f"MAE: {results['mae']:.6f}")
    logger.info(f"R2分数: {results['r2']:.6f}")
    
    wandb.log({
        'final_test_loss': results['test_loss'],
        'final_mse': results['mse'],
        'final_mae': results['mae'],
        'final_r2': results['r2']
    })
    
    # 保存模型和结果
    save_dir = os.path.join('models', config['element'])
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
    torch.save({
        'model_state_dict': best_model_state,
        'config': config,
        'normalization_params': train_dataset.get_normalization_params(),
        'results': results
    }, model_path)
    
    logger.info(f"\n模型已保存到: {model_path}")
    wandb.save(model_path)
    wandb.finish()

if __name__ == '__main__':
    main() 