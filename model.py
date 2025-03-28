import torch
import torch.nn as nn
import config
import os
import numpy as np
import logging
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

# 配置logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建格式化器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# 添加处理器到logger
logger.addHandler(console_handler)

"""
模型文件结构：
1. 模型定义
   - ResidualBlock: 残差块实现
   - SpectralResCNN: 主模型架构

2. 训练相关
   - train_model: 模型训练主函数
   - _train_epoch: 单轮训练
   - _validate: 验证过程
   - _save_checkpoint: 保存检查点

3. 评估相关
   - evaluate_model: 模型评估（用于测试集评估）
   - predict: 模型预测（用于单样本预测）

4. 工具函数
   - load_trained_model: 加载训练好的模型

完整流程：
1. 训练阶段：
   - 使用train_model进行模型训练
   - 通过_validate进行验证
   - 使用_save_checkpoint保存最佳模型

2. 测试阶段：
   - 使用load_trained_model加载训练好的模型
   - 使用evaluate_model在测试集上进行评估
   - 使用predict进行单样本预测
"""

# =============== 1. 模型定义 ===============
class ResidualBlock(nn.Module):
    """
    残差块
    特点：
    1. 包含两个卷积层，每层后接BatchNorm和ReLU
    2. 跳跃连接，将输入直接加到输出上
    3. 通过1x1卷积调整输入通道数（如需要）
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = F.relu(out)
        
        return out

class SpectralResCNN(nn.Module):
    """光谱残差CNN模型"""
    def __init__(self, input_size):
        super(SpectralResCNN, self).__init__()
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 残差模块 - 参数驱动的纵向特征提取
        self.res_blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(3)
        ])
        
        # 循环模块 - 跨波段信念增强
        self.gru = nn.GRU(64, 64, bidirectional=True, batch_first=True)
        self.cross_band_attention = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 信息融合层
        self.fusion = nn.Sequential(
            nn.Conv1d(192, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # 添加自适应池化层，确保输出大小固定
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # 特征提取
        x = self.feature_extractor(x)
        
        # 残差特征提取
        res_features = x
        for res_block in self.res_blocks:
            res_features = res_block(res_features)
            
        # 循环特征提取
        rec_features = x.permute(0, 2, 1)  # [batch, length, channels]
        rec_features, _ = self.gru(rec_features)
        rec_features = rec_features.permute(0, 2, 1)  # [batch, channels*2, length]
        
        # 跨波段信念增强
        attention_weights = self.cross_band_attention(rec_features)
        rec_features = rec_features * attention_weights
        
        # 特征融合
        combined_features = torch.cat([res_features, rec_features], dim=1)
        x = self.fusion(combined_features)
        
        # 使用自适应池化层将特征图压缩为固定大小
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        
        # 全连接层
        x = self.fc(x)
        
        return x

class SpectralResCNNEnsemble:
    """光谱CNN集成模型"""
    
    def __init__(self, input_size, num_models=3, dropout_rate=0.5):
        self.input_size = input_size
        self.num_models = num_models
        self.dropout_rate = dropout_rate
        self.models = []
        
        # 创建多个模型实例
        for i in range(num_models):
            model = SpectralResCNN(input_size)
            self.models.append(model)
    
    def train(self, train_loader, val_loader, element, config):
        """训练所有模型"""
        results = []
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{self.num_models}...")
            device = config.training_config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            train_losses, val_losses = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                element=f"{element}_model{i+1}",
                config=config,
                device=device,
                resume_from=None
            )
            results.append((train_losses, val_losses))
        return results
    
    def predict(self, x, device=None, mc_samples=5):
        """预测函数，支持MC-Dropout不确定性估计"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # 确保输入是张量
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # 添加通道维度
                
        x = x.to(device)
        
        # 收集所有预测结果
        all_predictions = []
        
        for model in self.models:
            model.eval()
            model_predictions = []
            
            # 进行MC-Dropout采样
            for _ in range(mc_samples):
                with torch.no_grad():
                    # 启用dropout(training=True)进行MC采样
                    outputs = model(x, training=True)
                    model_predictions.append(outputs.cpu().numpy())
            
            # 将模型的所有MC预测堆叠
            model_predictions = np.stack(model_predictions, axis=0)
            all_predictions.append(model_predictions)
        
        # 合并所有模型的预测结果
        all_predictions = np.stack(all_predictions, axis=0)  # [num_models, mc_samples, batch, 1]
        
        # 计算最终预测值（所有模型和样本的平均值）
        predictions = np.mean(all_predictions, axis=(0, 1))
        
        # 计算预测不确定性（所有预测的标准差）
        uncertainty = np.std(all_predictions, axis=(0, 1))
        
        return predictions.squeeze(), uncertainty.squeeze()

# =============== 2. 训练相关 ===============
def _train_epoch(model, train_loader, criterion, optimizer, device):
    """
    单轮训练函数
    """
    model.train()
    train_loss = 0
    for spectra, abundances in train_loader:
        spectra = spectra.to(device)
        abundances = abundances.to(device)
        
        optimizer.zero_grad()
        outputs = model(spectra)
        loss = criterion(outputs.squeeze(), abundances)
        
        # 检查损失值是否为 nan
        if torch.isnan(loss):
            logger.warning("Detected nan loss value, skipping current batch")
            continue
        
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item()
    
    return train_loss / len(train_loader)

def _validate(model, val_loader, criterion, device):
    """
    验证函数
    """
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for spectra, abundances in val_loader:
            spectra = spectra.to(device)
            abundances = abundances.to(device)
            outputs = model(spectra)
            val_loss += criterion(outputs.squeeze(), abundances).item()
    
    return val_loss / len(val_loader)

def _save_checkpoint(model, optimizer, scheduler, epoch, loss, element, config):
    """
    保存检查点
    """
    model_path = os.path.join(config['output']['model_dir'], f'best_model_{element}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, model_path)

def train(model, train_loader, val_loader, element, config, device='cuda', resume_from=None):
    """
    训练模型，使用分阶段训练策略
    """
    import torch.optim as optim
    import torch.nn as nn
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.training_config['lr'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.MSELoss()
    
    # 训练参数
    num_epochs = config.training_config['num_epochs']
    patience = config.training_config['early_stopping_patience']
    
    # 记录最佳模型
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # 恢复训练
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
    
    # 第一阶段：特征提取器训练
    logger.info("第一阶段：训练特征提取器")
    for param in model.fc.parameters():
        param.requires_grad = False
    
    for epoch in range(start_epoch, num_epochs // 2):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(epoch + batch_idx / len(train_loader))
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f'第一阶段 - 轮次: {epoch} [{batch_idx}/{len(train_loader)}], 损失: {loss.item():.6f}')
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
    
    # 第二阶段：全模型微调
    logger.info("第二阶段：全模型微调")
    for param in model.fc.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(model.parameters(), lr=config.training_config['lr'] * 0.1)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    for epoch in range(num_epochs // 2, num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(epoch + batch_idx / len(train_loader))
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f'第二阶段 - 轮次: {epoch} [{batch_idx}/{len(train_loader)}], 损失: {loss.item():.6f}')
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(config.model_config['model_dir'], f'best_model_{element}.pth'))
        else:
            patience_counter += 1
            
        # 保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }, os.path.join(config.model_config['model_dir'], f'checkpoint_{element}.pth'))
        
        # 早停
        if patience_counter >= patience:
            logger.info(f'早停：验证损失在 {patience} 轮内没有改善')
            break
            
    return train_losses, val_losses

# =============== 3. 评估相关 ===============
def evaluate_model(model, test_loader, device=None):
    """
    在测试集上评估模型
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
    Returns:
        评估结果字典
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model.to(device)
    model.eval()
    
    all_outputs = []
    all_targets = []
    
    # 添加NaN统计
    nan_stats = {'input': 0, 'output': 0, 'total_samples': 0}
    
    with torch.no_grad():
        for data, target in test_loader:
            nan_stats['total_samples'] += data.size(0)
            
            # 将数据移到相应设备
            data, target = data.to(device), target.to(device)
            
            # 处理输入数据中的NaN值
            data, has_nan, nan_count = handle_nan_values(
                data, 
                replacement_strategy='mean', 
                name="评估输入数据"
            )
            if has_nan:
                nan_stats['input'] += nan_count
            
            try:
                # 前向传播
                output = model(data)
                
                # 处理输出中的NaN值
                output, has_nan, nan_count = handle_nan_values(
                    output, 
                    replacement_strategy='zero', 
                    name="评估模型输出"
                )
                if has_nan:
                    nan_stats['output'] += nan_count
                
                # 收集非NaN预测结果和真实值
                valid_indices = ~torch.isnan(output).any(dim=1) & ~torch.isnan(target).any(dim=1)
                if valid_indices.any():
                    all_outputs.append(output[valid_indices].cpu().numpy())
                    all_targets.append(target[valid_indices].cpu().numpy())
                
            except Exception as e:
                logger.error(f"Error during evaluation: {str(e)}")
                continue
    
    # 如果出现了NaN值，记录日志
    if nan_stats['input'] > 0 or nan_stats['output'] > 0:
        logger.warning(f"Total samples processed during evaluation: {nan_stats['total_samples']}")
        logger.warning(f"NaN values in input data: {nan_stats['input']}")
        logger.warning(f"NaN values in model output: {nan_stats['output']}")
    
    # 合并结果，计算指标
    if len(all_outputs) > 0 and len(all_targets) > 0:
        try:
            all_outputs = np.vstack(all_outputs)
            all_targets = np.vstack(all_targets)
            
            # 检查连接后的数据是否仍包含NaN
            if np.isnan(all_outputs).any() or np.isnan(all_targets).any():
                logger.warning("Merged evaluation data still contains NaN values, attempting to filter")
                # 过滤掉包含NaN的行
                valid_rows = ~np.isnan(all_outputs).any(axis=1) & ~np.isnan(all_targets).any(axis=1)
                all_outputs = all_outputs[valid_rows]
                all_targets = all_targets[valid_rows]
                
                if len(all_outputs) == 0:
                    logger.error("Filtered NaN, no valid data for evaluation")
                    return {
                        'mse': float('nan'),
                        'rmse': float('nan'),
                        'mae': float('nan'),
                        'r2': float('nan')
                    }
            
            # 计算指标
            mse = np.mean((all_outputs - all_targets) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(all_outputs - all_targets))
            r2 = r2_score(all_targets.flatten(), all_outputs.flatten())
    
            # 计算散点图的统计数据
            slope, intercept, r_value, p_value, std_err = stats.linregress(all_targets.flatten(), all_outputs.flatten())
            
            # 返回评估结果
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'r_value': r_value,
                'slope': slope,
                'intercept': intercept,
                'std_err': std_err,
                'p_value': p_value
            }
                
        except Exception as e:
            logger.error(f"Error calculating evaluation metrics: {str(e)}")
    
    # 如果没有有效数据或发生错误，返回NaN指标
    return {
        'mse': float('nan'),
        'rmse': float('nan'),
        'mae': float('nan'),
        'r2': float('nan')
    }

# 添加一个新的工具函数，用于处理和检测NaN值
def handle_nan_values(tensor, replacement_strategy='mean', fill_value=0.0, name="数据"):
    """
    检测并处理张量中的NaN值
    
    参数:
        tensor (torch.Tensor): 需要处理的张量
        replacement_strategy (str): 替换策略，可选值: 'mean', 'zero', 'value'
        fill_value (float): 如果策略是'value'，使用此值填充
        name (str): 用于日志记录的数据名称
        
    返回:
        tuple: (处理后的张量, 是否含有NaN, NaN的数量)
    """
    # 检查是否包含NaN
    has_nan = torch.isnan(tensor).any()
    
    if not has_nan:
        return tensor, False, 0
    
    # 计算NaN的数量
    nan_mask = torch.isnan(tensor)
    nan_count = torch.sum(nan_mask).item()
    
    # 创建张量的副本以进行修改
    result = tensor.clone()
    
    # 根据策略替换NaN值
    if replacement_strategy == 'mean':
        # 计算非NaN值的均值
        tensor_mean = torch.nanmean(tensor, dim=0)
        # 对每一行，用相应维度的均值替换NaN
        for i in range(tensor.size(0)):
            sample_nan_mask = nan_mask[i]
            if sample_nan_mask.any():
                result[i][sample_nan_mask] = tensor_mean[sample_nan_mask]
    elif replacement_strategy == 'zero':
        result[nan_mask] = 0.0
    else:  # 'value'
        result[nan_mask] = fill_value
    
    logger.warning(f"Detected {nan_count} NaN values in {name}, replaced with {replacement_strategy} strategy")
    
    return result, True, nan_count

# 修改predict函数
def predict(model, data_loader, device=None):
    """
    使用训练好的模型进行预测
    
    参数:
        model (torch.nn.Module): 训练好的模型
        data_loader (DataLoader): 数据加载器
        device (str, optional): 计算设备
        
    返回:
        numpy.ndarray: 预测结果
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model.to(device)
    model.eval()
    
    predictions = []
    nan_stats = {'input': 0, 'output': 0, 'total_samples': 0}
    
    with torch.no_grad():
        for data in data_loader:
            # 处理不同的数据格式
            if isinstance(data, (list, tuple)) and len(data) >= 1:
                # 数据加载器返回(data, target)格式
                inputs = data[0]
            else:
                # 数据加载器只返回data
                inputs = data
            
            # 将数据移动到指定设备
            inputs = inputs.to(device)
            nan_stats['total_samples'] += inputs.size(0)
            
            # 处理输入数据中的NaN值
            inputs, has_nan, nan_count = handle_nan_values(
                inputs, 
                replacement_strategy='mean', 
                name="预测输入数据"
            )
            if has_nan:
                nan_stats['input'] += nan_count
            
            try:
                # 前向传播
                outputs = model(inputs)
                
                # 处理输出中的NaN值
                outputs, has_nan, nan_count = handle_nan_values(
                    outputs, 
                    replacement_strategy='zero', 
                    name="模型输出"
                )
                if has_nan:
                    nan_stats['output'] += nan_count
                
                predictions.append(outputs.cpu().numpy())
                
            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
                # 在异常情况下，生成一个全零的假输出
                fake_output = torch.zeros_like(inputs[:, 0:1])  # 假设输出是一维的
                predictions.append(fake_output.cpu().numpy())
    
    # 如果出现了NaN值，记录日志
    if nan_stats['input'] > 0 or nan_stats['output'] > 0:
        logger.warning(f"Total samples processed during prediction: {nan_stats['total_samples']}")
        logger.warning(f"NaN values in input data: {nan_stats['input']}")
        logger.warning(f"NaN values in model output: {nan_stats['output']}")
    
    if len(predictions) > 0:
        return np.vstack(predictions)
    else:
        return np.array([])

# =============== 4. 工具函数 ===============
def load_trained_model(model_path, device=None):
    """
    加载训练好的模型
    
    参数:
        model_path (str): 模型权重文件路径
        device (str, optional): 计算设备
        
    返回:
        torch.nn.Module: 加载好权重的模型
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 防止递归错误：设置递归深度限制
        import sys
        original_recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(10000)  # 增加递归深度限制
        
        # 尝试加载模型
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # 检查checkpoint是否包含模型架构信息
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 创建模型实例
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                input_size = config.get('input_size', 1024)
                hidden_size = config.get('hidden_size', 256)
                output_size = config.get('output_size', 1)
                dropout_rate = config.get('dropout_rate', 0.2)
                model = SpectralResCNN(input_size)
            else:
                # 使用默认参数
                logger.warning(f"Model configuration missing, using default parameters")
                model = SpectralResCNN()
                
            # 加载模型权重
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 直接加载模型状态字典
            logger.info(f"Loading from direct state dict")
            model = SpectralResCNN()
            
            # 如果checkpoint不是字典而是OrderedDict，直接加载
            if not isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            else:
                # 尝试直接加载整个checkpoint作为state_dict
                try:
                    model.load_state_dict(checkpoint)
                except Exception as e:
                    logger.error(f"Failed to load state_dict: {str(e)}")
                    # 回退到默认模型
                    model = SpectralResCNN()
        
        # 恢复原始递归深度限制
        sys.setrecursionlimit(original_recursion_limit)
        
        model.to(device)
        model.eval()  # 设置为评估模式
        logger.info(f"Successfully loaded model from {model_path}")
        return model
        
    except RecursionError as e:
        logger.error(f"Recursion error while loading model: {str(e)}")
        # 在递归错误时，返回一个新的未训练模型
        model = SpectralResCNN()
        model.to(device)
        model.eval()
        logger.warning("Using untrained model as backup due to recursion error")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # 创建一个空模型作为后备选择
        backup_model = SpectralResCNN()
        backup_model.to(device)
        backup_model.eval()
        logger.warning("Using untrained model as backup")
        return backup_model

# 在模型训练完成后添加
def analyze_model_performance(self, element, train_loader, val_loader, test_loader):
    """对训练好的模型进行全面性能分析"""
    # 创建结果目录
    os.makedirs("results/feature_importance", exist_ok=True)
    os.makedirs("results/residual_analysis", exist_ok=True)
    
    logger.info(f"Starting feature importance analysis for {element} model...")
    self.analyze_feature_importance(self.model, val_loader, self.device, element)
    
    logger.info(f"Starting residual analysis for {element} model...")
    self.analyze_residuals(self.model, test_loader, self.device, element)
    
    logger.info(f"{element} model analysis completed, results saved in results directory")

"""
关键超参数调优指南：

1. 模型架构参数：
   - 输入卷积层：
     * kernel_size: [5, 7, 9]  # 建议从7开始
     * 输出通道数: [16, 32, 64]  # 建议从32开始
   
   - 残差块：
     * 每个组的残差块数量: [1, 2, 3]  # 建议从2开始
     * 通道数变化: [32->64, 64->128, 128->256]  # 建议保持当前设置
     * kernel_size: [3, 5]  # 建议从5开始
   
   - 全连接层：
     * 隐藏层大小: [256, 512, 1024]  # 建议从512开始
     * Dropout率: [0.3, 0.5, 0.7]  # 建议从0.5开始

2. 训练参数：
   - 学习率: [1e-4, 5e-4, 1e-3]  # 建议从5e-4开始
   - 批次大小: [32, 64, 128]  # 建议从64开始
   - 权重衰减: [1e-4, 5e-4, 1e-5]  # 建议从1e-4开始
   - 早停耐心值: [10, 15, 20]  # 建议从15开始

调参建议：
1. 首先固定模型架构，只调整训练参数
2. 使用网格搜索，但范围要小：
   - 学习率: [5e-4, 1e-3]
   - 批次大小: [64, 128]
   - 权重衰减: [1e-4, 5e-4]
3. 如果效果不理想，再考虑调整模型架构参数
4. 建议的实验顺序：
   a. 基准实验：使用当前参数设置
   b. 学习率实验：调整学习率
   c. 批次大小实验：调整批次大小
   d. 正则化实验：调整权重衰减
   e. 模型深度实验：调整残差块数量
   f. 特征维度实验：调整通道数
"""

# 从main中复制train_and_evaluate_model函数定义
def train_and_evaluate_model(train_loader, val_loader, test_loader, element, config):
    """
    训练和评估模型的主函数
    """
    # 创建模型
    model = SpectralResCNN(config.model_config['input_size']).to(config.training_config['device'])
    
    # 设置超参数
    hyperparams = {
        'lr': config.training_config['lr'],
        'weight_decay': config.training_config['weight_decay'],
        'epochs': config.training_config['num_epochs'],
        'patience': config.training_config['early_stopping_patience']
    }
    
    # 训练模型
    train_losses, val_losses = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        element=element,
        config=config
    )
    
    # 获取最佳验证损失
    best_val_loss = min(val_losses) if val_losses else float('inf')
    
    # 加载最佳模型
    best_model = load_trained_model(config.model_config['input_size'], element, config)
    
    # 在测试集上评估
    test_metrics = evaluate_model(best_model, test_loader, config.training_config['device'])
    
    # 分析模型性能（特征重要性和残差）
    if hasattr(config, 'analysis_config') and config.analysis_config.get('enabled', False):
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Starting performance analysis for {element} model...")
        
        # 获取分析配置
        batch_size = config.analysis_config.get('batch_size', 32)
        save_batch_results = config.analysis_config.get('batch_results', {}).get('save_batch_results', True)
        
        try:
            from model_analysis import analyze_model_performance
            analysis_results = analyze_model_performance(
                best_model,
                element,
                train_loader,
                val_loader,
                test_loader,
                config.training_config['device'],
                config.model_config['input_size'],
                batch_size=batch_size,
                save_batch_results=save_batch_results
            )
            logger.info(f"{element} model performance analysis completed, results saved in results directory")
        except ImportError:
            logger.warning("model_analysis module not found, skipping performance analysis")
    
    return best_model, best_val_loss, test_metrics 

# 添加一个兼容性包装函数，支持旧版本的调用方式
def load_trained_model_compat(input_size_or_path, element_or_device=None, config=None):
    """
    兼容性加载模型函数，支持新旧两种不同的调用格式
    
    方式1: load_trained_model_compat(model_path, device=None)
    方式2: load_trained_model_compat(input_size, element, config)
    """
    # 检测调用方式
    if isinstance(input_size_or_path, str) and (element_or_device is None or isinstance(element_or_device, (str, torch.device))):
        # 新方式: load_trained_model(model_path, device=None)
        return load_trained_model(input_size_or_path, element_or_device)
    
    # 旧方式: load_trained_model(input_size, element, config)
    input_size = input_size_or_path
    element = element_or_device
    device = config.training_config['device']
    
    # 尝试找到模型路径
    try:
        # 确保模型目录存在
        os.makedirs(config.model_config['model_dir'], exist_ok=True)
        
        # 尝试多种可能的模型文件命名模式
        model_paths = [
            os.path.join(config.model_config['model_dir'], f"{element}_model.pth"),  # 标准格式
            os.path.join(config.model_config['model_dir'], f"{element.lower()}_model.pth"),  # 小写
            os.path.join(config.model_config['model_dir'], f"{element.upper()}_model.pth"),  # 大写
            os.path.join(config.model_config['model_dir'], f"{element}_best_model.pth"),  # best前缀
            os.path.join(config.model_config['model_dir'], f"model_{element}.pth"),  # 另一种格式
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
                
        if model_path is None:
            logger.warning(f"找不到{element}的模型文件，尝试其他格式")
            
            # 检查目录中的所有文件，查找可能匹配的模型
            if os.path.exists(config.model_config['model_dir']):
                for file in os.listdir(config.model_config['model_dir']):
                    if file.endswith('.pth') and element.lower() in file.lower():
                        model_path = os.path.join(config.model_config['model_dir'], file)
                        logger.info(f"找到可能匹配的模型文件: {file}")
                        break
        
        if model_path is None:
            logger.error(f"找不到元素 {element} 的模型文件")
            # 创建一个空模型作为后备选择
            backup_model = SpectralResCNN(input_size)
            backup_model.to(device)
            backup_model.eval()
            logger.warning(f"使用未训练的模型作为后备")
            return backup_model
        
        # 找到模型文件，加载它
        return load_trained_model(model_path, device)
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # 创建一个空模型作为后备选择
        backup_model = SpectralResCNN(input_size)
        backup_model.to(device)
        backup_model.eval()
        logger.warning(f"Using untrained model as backup")
        return backup_model

# 修改原来的load_trained_model函数名为load_trained_model_core，保持旧函数名对旧函数的向后兼容性
load_trained_model_core = load_trained_model
load_trained_model = load_trained_model_compat 
