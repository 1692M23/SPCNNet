import torch
import torch.nn as nn
import torch.optim as optim
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

def train(model, train_loader, val_loader, config, device, element):
    """训练模型"""
    logger = logging.getLogger('model')
    
    # 配置兼容性处理：确保config有正确的结构
    if 'training' not in config:
        if 'training_config' in config:
            logger.warning("配置结构使用了'training_config'键而非'training'键，正在转换...")
            config['training'] = config['training_config']
        else:
            logger.error("配置缺少训练参数！尝试创建默认配置。")
            config['training'] = {
                'lr': 0.001,
                'weight_decay': 1e-4,
                'num_epochs': 50,
                'early_stopping_patience': 10
            }
    
    # 确保model_config路径存在
    if 'model_config' not in config or 'model_dir' not in config['model_config']:
        if 'model_dir' not in config.get('model_config', {}):
            logger.warning("配置缺少model_dir路径，使用默认路径。")
            if 'model_config' not in config:
                config['model_config'] = {}
            config['model_config']['model_dir'] = 'models'
    
    # 设置优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), 
                          lr=config['training']['lr'],
                          weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 设置损失函数
    criterion = nn.MSELoss()
    
    # 设置早停
    best_val_loss = float('inf')
    patience = config['training']['early_stopping_patience']
    patience_counter = 0
    
    # 训练记录
    train_losses = []
    val_losses = []
    
    # 第一阶段：只训练特征提取器的前几层
    logger.info("开始第一阶段训练 - 特征提取器")
    
    # 获取模型的所有参数
    all_params = list(model.parameters())
    # 冻结最后一层的参数
    for param in all_params[-2:]:  # 最后一个线性层的权重和偏置
        param.requires_grad = False
    
    for epoch in range(config['training']['num_epochs']):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 检查数据是否包含NaN
            if torch.isnan(data).any() or torch.isnan(target).any():
                logger.warning("检测到输入数据包含NaN值，跳过该批次")
                continue
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"第一阶段 - 轮次: {epoch} [{batch_idx}/{len(train_loader)}], 损失: {loss.item():.6f}")
        
        # 更新学习率
        scheduler.step()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        
        val_loss /= len(val_loader)
        train_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        logger.info(f"第一阶段 - 轮次: {epoch}, 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            save_model(model, optimizer, scheduler, epoch, val_loss, config['model_config']['model_dir'], element)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"第一阶段 - 早停触发，最佳验证损失: {best_val_loss:.6f}")
                break
    
    # 第二阶段：微调全模型
    logger.info("开始第二阶段训练 - 全模型微调")
    # 解冻所有层
    for param in model.parameters():
        param.requires_grad = True
    
    # 重置优化器和学习率
    optimizer = optim.Adam(model.parameters(), 
                          lr=config['training']['lr'] * 0.1,
                          weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    # 重置早停
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['training']['num_epochs']):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            if torch.isnan(data).any() or torch.isnan(target).any():
                logger.warning("检测到输入数据包含NaN值，跳过该批次")
                continue
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"第二阶段 - 轮次: {epoch} [{batch_idx}/{len(train_loader)}], 损失: {loss.item():.6f}")
        
        # 更新学习率
        scheduler.step()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        
        val_loss /= len(val_loader)
        train_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        logger.info(f"第二阶段 - 轮次: {epoch}, 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            save_model(model, optimizer, scheduler, epoch, val_loss, config['model_config']['model_dir'], element)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"第二阶段 - 早停触发，最佳验证损失: {best_val_loss:.6f}")
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
    
    logger.info(f"在设备 {device} 上评估模型")
    
    # 确保模型在正确的设备上
    if next(model.parameters()).device != device:
        logger.info(f"将模型从 {next(model.parameters()).device} 移动到 {device}")
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
                logger.error(f"评估过程中出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                continue
    
    # 如果出现了NaN值，记录日志
    if nan_stats['input'] > 0 or nan_stats['output'] > 0:
        logger.warning(f"评估过程中处理的样本总数: {nan_stats['total_samples']}")
        logger.warning(f"输入数据中的NaN值数量: {nan_stats['input']}")
        logger.warning(f"模型输出中的NaN值数量: {nan_stats['output']}")
    
    # 合并结果，计算指标
    if len(all_outputs) > 0 and len(all_targets) > 0:
        try:
            all_outputs = np.vstack(all_outputs)
            all_targets = np.vstack(all_targets)
            
            # 检查连接后的数据是否仍包含NaN
            if np.isnan(all_outputs).any() or np.isnan(all_targets).any():
                logger.warning("合并后的评估数据仍包含NaN值，尝试过滤")
                # 过滤掉包含NaN的行
                valid_rows = ~np.isnan(all_outputs).any(axis=1) & ~np.isnan(all_targets).any(axis=1)
                all_outputs = all_outputs[valid_rows]
                all_targets = all_targets[valid_rows]
                
                if len(all_outputs) == 0:
                    logger.error("过滤NaN后，没有有效数据进行评估")
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
            
            # 安全计算R2分数
            try:
                r2 = r2_score(all_targets.flatten(), all_outputs.flatten())
            except Exception:
                # 如果r2_score函数失败，手动计算
                y_mean = np.mean(all_targets)
                ss_total = np.sum((all_targets - y_mean) ** 2)
                ss_residual = np.sum((all_targets - all_outputs) ** 2)
                r2 = 1 - (ss_residual / ss_total if ss_total > 0 else 0)
            
            # 计算散点图的统计数据
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(all_targets.flatten(), all_outputs.flatten())
            except Exception as e:
                logger.warning(f"无法计算线性回归统计: {str(e)}")
                slope, intercept, r_value, p_value, std_err = float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
            
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
                'p_value': p_value,
                'num_samples': len(all_outputs)
            }
                
        except Exception as e:
            logger.error(f"计算评估指标时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    # 如果没有有效数据或发生错误，返回NaN指标
    return {
        'mse': float('nan'),
        'rmse': float('nan'),
        'mae': float('nan'),
        'r2': float('nan'),
        'num_samples': 0
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
    
    logger.info(f"在设备 {device} 上进行预测")
    
    # 确保模型在正确的设备上
    if next(model.parameters()).device != device:
        logger.info(f"将模型从 {next(model.parameters()).device} 移动到 {device}")
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
                logger.error(f"预测过程中出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
                # 在异常情况下，生成一个全零的假输出
                fake_output = torch.zeros((inputs.size(0), 1), device=device)
                predictions.append(fake_output.cpu().numpy())
    
    # 如果出现了NaN值，记录日志
    if nan_stats['input'] > 0 or nan_stats['output'] > 0:
        logger.warning(f"预测过程中处理的样本总数: {nan_stats['total_samples']}")
        logger.warning(f"输入数据中的NaN值数量: {nan_stats['input']}")
        logger.warning(f"模型输出中的NaN值数量: {nan_stats['output']}")
    
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
    # 设置默认设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    logger.info(f"从 {model_path} 加载模型到设备 {device}")
    
    try:
        # 防止递归错误：设置递归深度限制
        import sys
        original_recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(10000)  # 增加递归深度限制
        
        # 首先检查文件是否存在
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        
        # 尝试加载模型
        try:
            checkpoint = torch.load(model_path, map_location=device)
            logger.info(f"成功加载checkpoint: {type(checkpoint)}")
        except Exception as e:
            logger.error(f"加载模型文件失败: {str(e)}")
            raise RuntimeError(f"无法加载模型权重: {str(e)}")
        
        # 确定输入大小，用于创建模型
        input_size = None
        
        # 检查checkpoint是否包含模型架构信息
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                logger.info("检测到model_state_dict格式")
                state_dict = checkpoint['model_state_dict']
                
                # 尝试从模型配置提取输入大小
                if 'model_config' in checkpoint:
                    config = checkpoint['model_config']
                    input_size = config.get('input_size', None)
                    logger.info(f"从模型配置中提取输入大小: {input_size}")
            else:
                # 尝试直接将整个checkpoint作为state_dict
                logger.info("尝试将checkpoint直接视为state_dict")
                state_dict = checkpoint
        else:
            # 如果checkpoint不是字典，可能直接是state_dict
            logger.info("checkpoint不是字典，直接用作state_dict")
            state_dict = checkpoint
        
        # 创建模型
        model = SpectralResCNN_GCN(input_size=input_size, device=device)
        
        # 尝试加载权重
        try:
            # 处理键名不匹配的情况（常见于旧模型与新代码）
            if isinstance(state_dict, dict):
                # 创建新的state_dict，修正键名问题
                new_state_dict = {}
                for k, v in state_dict.items():
                    # 处理可能的前缀问题
                    if k.startswith('module.'):
                        # 有时多GPU训练会添加'module.'前缀
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                
                # 尝试加载修正后的state_dict
                missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
                
                if missing_keys:
                    logger.warning(f"加载权重时缺少键: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"加载权重时有意外的键: {unexpected_keys}")
            else:
                logger.error(f"无法加载state_dict，格式错误: {type(state_dict)}")
                raise TypeError(f"state_dict类型错误: {type(state_dict)}")
        except Exception as e:
            logger.error(f"加载模型权重失败: {str(e)}")
            raise RuntimeError(f"加载模型权重失败: {str(e)}")
        
        # 恢复原始递归深度限制
        sys.setrecursionlimit(original_recursion_limit)
        
        # 将模型设置为评估模式
        model.eval()
        logger.info(f"成功从 {model_path} 加载模型")
        return model
        
    except RecursionError as e:
        logger.error(f"加载模型时递归错误: {str(e)}")
        # 在递归错误时，返回一个新的未训练模型
        model = SpectralResCNN_GCN(device=device)
        model.eval()
        logger.warning("由于递归错误，使用未训练的模型作为备份")
        return model
    except Exception as e:
        logger.error(f"加载模型时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # 创建一个空模型作为后备选择
        backup_model = SpectralResCNN_GCN(device=device)
        backup_model.eval()
        logger.warning("使用未训练的模型作为备份")
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
    # 获取设备
    device = config.training_config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"使用设备: {device}")
    
    # 创建模型 - 根据配置选择模型类型
    use_gcn = config.model_config.get('use_gcn', True)
    input_size = config.model_config.get('input_size')
    
    if use_gcn:
        logger.info(f"使用SpectralResCNN_GCN模型，输入大小: {input_size}")
        model = SpectralResCNN_GCN(
            input_size=input_size, 
            device=device
        )
    else:
        logger.info(f"使用SpectralResCNN模型，输入大小: {input_size}")
        model = SpectralResCNN(
            input_size=input_size
        ).to(device)
    
    # 设置超参数
    hyperparams = {
        'lr': config.training_config.get('lr', 0.001),
        'weight_decay': config.training_config.get('weight_decay', 1e-5),
        'epochs': config.training_config.get('num_epochs', 100),
        'patience': config.training_config.get('early_stopping_patience', 15)
    }
    
    logger.info(f"训练 {element} 模型，超参数: {hyperparams}")
    
    # 训练模型
    train_losses, val_losses = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config={
            'training': {
                'lr': hyperparams['lr'],
                'weight_decay': hyperparams['weight_decay'],
                'num_epochs': hyperparams['epochs'],
                'early_stopping_patience': hyperparams['patience'],
                'device': device
            },
            'model_config': {
                'model_dir': config.model_config.get('model_dir', 'models'),
                'use_gcn': use_gcn  # 传递模型类型
            }
        },
        device=device,
        element=element
    )
    
    # 获取最佳验证损失
    best_val_loss = min(val_losses) if val_losses else float('inf')
    
    # 加载最佳模型
    model_path = os.path.join(config.model_config.get('model_dir', 'models'), f'best_model_{element}.pth')
    if os.path.exists(model_path):
        best_model = load_trained_model(model_path, device)
    else:
        logger.warning(f"找不到最佳模型文件: {model_path}，使用当前模型代替")
        best_model = model
    
    # 在测试集上评估
    logger.info(f"在测试集上评估 {element} 模型")
    test_metrics = evaluate_model(best_model, test_loader, device)
    
    # 分析模型性能（特征重要性和残差）
    if hasattr(config, 'analysis_config') and config.analysis_config.get('enabled', False):
        logger.info(f"开始 {element} 模型性能分析...")
        
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
                device,
                batch_size=batch_size,
                save_batch_results=save_batch_results
            )
            logger.info(f"{element} 模型性能分析完成，结果已保存到results目录")
        except ImportError:
            logger.warning("找不到model_analysis模块，跳过性能分析")
    
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

def save_model(model, optimizer, scheduler, epoch, val_loss, model_dir, element):
    """保存模型检查点"""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'best_model_{element}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
    }, model_path) 

class CNNModel(nn.Module):
    def __init__(self, input_size):
        """
        初始化CNN模型
        
        参数:
            input_size: 输入光谱的长度
        """
        super(CNNModel, self).__init__()
        
        # 计算卷积层输出大小
        def conv_output_size(input_size, kernel_size, stride=1, padding=0):
            return (input_size + 2 * padding - kernel_size) // stride + 1
        
        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # 第三个卷积块
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # 计算展平后的特征维度
        self.feature_size = input_size
        self.feature_size = conv_output_size(self.feature_size, 7, 1, 3)  # conv1
        self.feature_size = conv_output_size(self.feature_size, 2, 2, 0)  # pool1
        self.feature_size = conv_output_size(self.feature_size, 5, 1, 2)  # conv2
        self.feature_size = conv_output_size(self.feature_size, 2, 2, 0)  # pool2
        self.feature_size = conv_output_size(self.feature_size, 3, 1, 1)  # conv3
        self.feature_size = conv_output_size(self.feature_size, 2, 2, 0)  # pool3
        self.feature_size *= 256  # 通道数
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, input_size)
        返回:
            输出张量，形状为 (batch_size, 1)
        """
        # 添加通道维度
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        
        # 卷积层
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc(x)
        
        return x 

class GraphConvLayer(nn.Module):
    """光谱图卷积层
    
    将光谱数据视为图结构，相邻波长点之间有连接关系，
    同时添加一些"跳跃连接"捕获远距离波长之间的关系。
    """
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.adj_matrix = None
        self.adj_cache = {}  # 缓存不同长度的邻接矩阵
        
    def forward(self, x):
        """
        输入 x: [batch_size, seq_len, in_features]
        输出: [batch_size, seq_len, out_features]
        """
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        # 为当前序列长度获取或构建邻接矩阵
        adj_key = f"{seq_len}"
        if adj_key not in self.adj_cache:
            # 创建默认邻接矩阵（相邻点连接 + 固定距离跳跃连接）
            adj = self._build_spectral_adjacency(seq_len).to(device)
            # 归一化邻接矩阵
            adj = self._normalize_adj(adj)
            self.adj_cache[adj_key] = adj
        
        # 使用缓存的邻接矩阵
        adj_matrix = self.adj_cache[adj_key]
        # 确保邻接矩阵在正确的设备上
        if adj_matrix.device != device:
            adj_matrix = adj_matrix.to(device)
            self.adj_cache[adj_key] = adj_matrix
        
        # 图卷积: X' = AXW
        # 首先应用线性变换
        support = self.linear(x)  # [batch, seq_len, out_features]
        
        # 然后应用邻接矩阵
        output = torch.matmul(adj_matrix, support)  # [batch, seq_len, out_features]
        
        return output
    
    def _build_spectral_adjacency(self, seq_len):
        """构建光谱数据的邻接矩阵"""
        # 初始化邻接矩阵为单位矩阵（每个节点与自身相连）
        adj = torch.eye(seq_len)
        
        # 添加相邻点的连接（一阶邻居）
        for i in range(seq_len-1):
            adj[i, i+1] = 1.0
            adj[i+1, i] = 1.0
        
        # 添加跳跃连接（特定间隔的节点相连，捕获远程关系）
        # 这里添加间隔为5和10的跳跃连接
        for skip in [5, 10, 20]:
            for i in range(seq_len - skip):
                adj[i, i+skip] = 0.5  # 可以给跳跃连接赋予较小的权重
                adj[i+skip, i] = 0.5
        
        return adj
    
    def _normalize_adj(self, adj):
        """对邻接矩阵进行归一化（按行归一化）"""
        row_sum = adj.sum(1, keepdim=True)
        # 避免除零错误
        row_sum = torch.clamp(row_sum, min=1e-6)
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt.squeeze())
        normalized_adj = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return normalized_adj

class SpectralAttention(nn.Module):
    """光谱注意力机制
    
    实现简单但完整的自注意力，捕获波长之间的关系
    """
    def __init__(self, channels):
        super(SpectralAttention, self).__init__()
        # 生成查询、键和值的投影
        self.query_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.key_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.value_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的权重参数
        
    def forward(self, x):
        """
        输入 x: [batch, channels, length]
        输出: [batch, channels, length]
        """
        device = x.device
        batch_size, C, width = x.size()
        
        # 投影查询、键和值
        query = self.query_conv(x).view(batch_size, -1, width).permute(0, 2, 1)  # [B, W, C]
        key = self.key_conv(x).view(batch_size, -1, width)  # [B, C, W]
        value = self.value_conv(x).view(batch_size, -1, width)  # [B, C, W]
        
        # 计算注意力得分
        energy = torch.bmm(query, key)  # [B, W, W]
        
        # 为了数值稳定性，在softmax前进行缩放
        energy_scaled = energy / (C ** 0.5)  # 缩放因子为通道数的平方根
        
        # 检查并处理潜在的数值问题
        if torch.isnan(energy_scaled).any() or torch.isinf(energy_scaled).any():
            # 使用更安全的方法：移除极端值并应用对数空间softmax
            max_val, _ = torch.max(energy_scaled, dim=2, keepdim=True)
            energy_safe = energy_scaled - max_val  # 减去最大值增加数值稳定性
            exp_x = torch.exp(energy_safe)
            sum_exp_x = torch.sum(exp_x, dim=2, keepdim=True)
            attention = exp_x / (sum_exp_x + 1e-10)  # 加入小的epsilon值防止除零
        else:
            attention = F.softmax(energy_scaled, dim=2)
        
        # 使用注意力权重更新值
        out = torch.bmm(value, attention.permute(0, 2, 1))
        
        # 残差连接
        out = self.gamma * out + x
        
        return out

class SpectralResCNN_GCN(nn.Module):
    """光谱残差CNN-GCN模型
    
    结合残差CNN和图卷积网络处理光谱数据
    """
    def __init__(self, input_size=None, device=None):
        super(SpectralResCNN_GCN, self).__init__()
        
        # 记录输入大小，可能为None表示自动适应
        self.input_size = input_size
        
        # 设置设备
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        
        # 光谱注意力机制 - 捕获波长关系
        self.spectral_attention = SpectralAttention(128)
        
        # GCN模块 - 建模波长点之间的关系
        # 注意：这里无需指定具体的序列长度，将在forward中动态适应
        self.gcn_layers = nn.ModuleList([
            GraphConvLayer(128, 128),
            GraphConvLayer(128, 128)
        ])
        
        # GCN输出处理
        self.gcn_process = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 信息融合层 (CNN特征 + GCN特征)
        self.fusion = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=1),  # 256 = 64(CNN) + 128(GRU) + 64(GCN)
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
        
        # 将模型移动到指定设备
        self.to(self.device)
        
    def forward(self, x, training=False):
        # 确保输入在正确的设备上
        if x.device != self.device:
            x = x.to(self.device)
            
        # 获取实际输入维度
        batch_size, channels, seq_len = x.size()
        
        # 如果input_size为None，自动设置为当前序列长度
        if self.input_size is None:
            self.input_size = seq_len
            print(f"自动适应输入大小: {seq_len}")
        
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
        
        # 应用光谱注意力
        attention_features = self.spectral_attention(rec_features)
        
        # GCN处理 - 首先需要转置数据
        gcn_features = attention_features.permute(0, 2, 1)  # [batch, length, channels]
        for gcn_layer in self.gcn_layers:
            gcn_features = gcn_layer(gcn_features)
            gcn_features = F.relu(gcn_features)
        
        # 处理GCN输出
        gcn_features = self.gcn_process(gcn_features)
        
        # 将GCN特征转回原始格式
        gcn_features = gcn_features.permute(0, 2, 1)  # [batch, channels, length]
        
        # 特征融合 (CNN + GRU + GCN)
        combined_features = torch.cat([res_features, attention_features, gcn_features], dim=1)
        x = self.fusion(combined_features)
        
        # 使用自适应池化层将特征图压缩为固定大小
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        
        # 全连接层 (如果在训练模式下且training=True，则保持dropout启用)
        if training:
            # 保存当前模式
            training_mode = self.training
            # 设置为训练模式启用dropout
            self.train()
            # 前向传播
            x = self.fc(x)
            # 恢复原始模式
            if not training_mode:
                self.eval()
        else:
            x = self.fc(x)
        
        return x 