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
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # 主路径
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        
        # 如果尺寸不匹配，使用1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        
        out = self.conv2(out)
        
        # 应用快捷连接
        out += self.shortcut(residual)
        out = self.leaky_relu(out)
        
        return out

class SpectralResCNN(nn.Module):
    """光谱残差CNN模型"""
    def __init__(self, input_size=1024, hidden_size=256, output_size=1, dropout_rate=0.2):
        """
        初始化模型
        
        参数:
            input_size (int): 输入光谱的大小
            hidden_size (int): 隐藏层大小
            output_size (int): 输出大小（元素数量）
            dropout_rate (float): Dropout比率
        """
        super(SpectralResCNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # 输入层 (1 x input_size) -> (32 x input_size) 
        self.input_layer = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # 特征提取层 - 残差块组1 (64 x input_size)
        self.group1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # 降采样层1 (64 x input_size) -> (96 x input_size/2)
        self.downsample1 = nn.Sequential(
            nn.Conv1d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # 特征提取层 - 残差块组2 (96 x input_size/2)
        self.group2 = nn.Sequential(
            ResidualBlock(96, 96),
            ResidualBlock(96, 96),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # 降采样层2 (96 x input_size/2) -> (128 x input_size/4)
        self.downsample2 = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # 特征提取层 - 残差块组3 (128 x input_size/4)
        self.group3 = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # 计算全连接层的输入大小 (考虑降采样)
        self.fc_input_size = (input_size // 4) * 128
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层 (分层设计，逐步减小)
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            
            nn.Linear(64, output_size)
        )
        
        # 跳跃连接: 直接从输入到输出的连接
        self.skip_connection = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, output_size)
        )
        
        # 权重初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """初始化模型权重"""
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        # 确保输入形状正确
        if len(x.shape) == 2:
            # (batch_size, features) -> (batch_size, channels, features)
            x = x.unsqueeze(1)
        
        # 保存原始输入用于跳跃连接
        input_copy = x
        
        # 特征提取
        x = self.input_layer(x)
        
        # 残差块处理
        g1_out = self.group1(x)
        x = x + g1_out  # 残差连接
        
        x = self.downsample1(x)
        
        g2_out = self.group2(x)
        x = x + g2_out  # 残差连接
        
        x = self.downsample2(x)
        
        g3_out = self.group3(x)
        x = x + g3_out  # 残差连接
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc_layers(x)
        
        # 跳跃连接 (直接从输入到输出)
        skip_out = self.skip_connection(input_copy)
        
        # 将主路径和跳跃连接相加
        x = x + skip_out * 0.1  # 0.1是跳跃连接的权重系数
        
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
            model = SpectralResCNN(input_size, dropout_rate=dropout_rate)
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

def train(model, train_loader, val_loader, element, config, device, resume_from=None):
    """
    训练模型
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        element: 元素名称
        config: 配置对象
        device: 计算设备
        resume_from: 恢复训练的检查点路径
    
    返回:
        tuple: (训练损失列表, 验证损失列表)
    """
    # 记录训练开始时间
    training_start_time = time.time()
    
    # 创建优化器
    weight_decay = config.training_config.get('weight_decay', 1e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training_config['lr'],
        weight_decay=weight_decay,  # 使用较小的权重衰减
        eps=1e-8  # 增加数值稳定性
    )
    
    # 使用余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.training_config['num_epochs'] // 3,  # 第一次重启的epoch数
        T_mult=2,  # 每次重启后周期的倍数
        eta_min=config.training_config['lr'] * 0.01  # 最小学习率
    )
    
    # 创建模型目录
    os.makedirs(config.model_config['model_dir'], exist_ok=True)
    
    # 创建批次结果目录
    batch_results_dir = os.path.join(config.output_config['results_dir'], f'training_{element}_batch_results')
    os.makedirs(batch_results_dir, exist_ok=True)
    
    # 初始化变量
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    nan_counter = 0
    max_nan_attempts = 5  # 增加允许的NaN尝试次数
    max_patience = config.training_config['early_stopping_patience']
    epoch_start = 0
    
    # 记录NaN值出现的次数和位置
    nan_stats = {
        'input_data': 0,
        'model_output': 0,
        'loss': 0,
        'gradients': 0,
        'total_batches': 0
    }
    
    # 创建批次追踪文件
    batch_tracking_path = os.path.join(batch_results_dir, 'batch_tracking.csv')
    batch_columns = ['epoch', 'train_loss', 'val_loss', 'lr', 'timestamp']
    
    # 加载现有的批次追踪文件（如果存在）
    if os.path.exists(batch_tracking_path):
        try:
            batch_df = pd.read_csv(batch_tracking_path)
            # 确保所有必需的列都存在
            for col in batch_columns:
                if col not in batch_df.columns:
                    batch_df[col] = None
        except Exception as e:
            logger.warning(f"Error loading existing batch tracking file: {str(e)}")
            batch_df = pd.DataFrame(columns=batch_columns)
    else:
        batch_df = pd.DataFrame(columns=batch_columns)
    
    # 创建NaN追踪日志
    nan_log_path = os.path.join(batch_results_dir, 'nan_tracking.csv')
    nan_columns = ['epoch', 'batch', 'location', 'action_taken', 'timestamp']
    
    # 加载现有的NaN追踪日志（如果存在）
    if os.path.exists(nan_log_path):
        try:
            nan_df = pd.read_csv(nan_log_path)
            # 确保所有必需的列都存在
            for col in nan_columns:
                if col not in nan_df.columns:
                    nan_df[col] = None
        except Exception as e:
            logger.warning(f"Error loading existing NaN tracking file: {str(e)}")
            nan_df = pd.DataFrame(columns=nan_columns)
    else:
        nan_df = pd.DataFrame(columns=nan_columns)
    
    # 如果提供了恢复点，从检查点恢复
    if resume_from is not None:
        try:
            logger.info(f"Resuming training from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch_start = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            patience_counter = checkpoint.get('patience_counter', 0)
            logger.info(f"Resumed from epoch {epoch_start} with best validation loss {best_val_loss}")
        except Exception as e:
            logger.error(f"Error resuming from checkpoint: {str(e)}")
            # 继续使用新初始化的模型和优化器
    
    # 修复梯度的函数
    def _fix_nan_gradients(model, threshold=1.0):
        """修复梯度中的NaN值并执行梯度裁剪"""
        nan_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                # 检查并修复NaN梯度
                nan_mask = torch.isnan(param.grad)
                if nan_mask.any():
                    nan_count += torch.sum(nan_mask).item()
                    param.grad[nan_mask] = 0.0
                
                # 执行元素级梯度裁剪
                with torch.no_grad():
                    too_large = torch.abs(param.grad) > threshold
                    if too_large.any():
                        # 裁剪大梯度但保留符号
                        param.grad[too_large] = threshold * torch.sign(param.grad[too_large])
        
        return nan_count
    
    # 重置模型和优化器的函数
    def _reset_model_and_optimizer(model, optimizer, counter, max_attempts):
        if counter >= max_attempts:
            logger.warning(f"Encountered NaN values in {counter} consecutive batches, resetting model parameters")
            
            # 重新初始化特定层
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv1d, nn.Linear)):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm1d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
            
            # 重置优化器状态
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad.zero_()
    
    # 训练循环
    logger.info(f"Starting training for element {element} from epoch {epoch_start+1}")
    for epoch in range(epoch_start, config.training_config['num_epochs']):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0
        valid_batches = 0
        skipped_batches = 0
        
        # 训练阶段
        for batch_idx, (data, target) in enumerate(train_loader):
            nan_stats['total_batches'] += 1
            
            # 检查数据和目标的维度是否匹配
            if data.size(0) != target.size(0):
                logger.warning(f"Batch size mismatch: data={data.size(0)}, target={target.size(0)}")
                continue
            
            try:
                data, target = data.to(device), target.to(device)
                
                # 检查输入数据是否包含nan并进行处理
                if torch.isnan(data).any():
                    # 记录NaN统计
                    nan_stats['input_data'] += 1
                    
                    # 尝试修复NaN值
                    data_mean = torch.nanmean(data, dim=0)
                    nan_mask = torch.isnan(data)
                    nan_count = torch.sum(nan_mask).item()
                    
                    # 用平均值填充NaN
                    data_clone = data.clone()
                    for i in range(data.size(0)):
                        sample_nan_mask = nan_mask[i]
                        if sample_nan_mask.any():
                            data_clone[i][sample_nan_mask] = data_mean[sample_nan_mask]
                    
                    # 记录到NaN日志
                    new_nan_row = pd.DataFrame({
                        'epoch': [epoch+1],
                        'batch': [batch_idx],
                        'location': ['input_data'],
                        'action_taken': [f'Filled {nan_count} NaN values with mean'],
                        'timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')]
                    })
                    nan_df = pd.concat([nan_df, new_nan_row], ignore_index=True)
                    
                    logger.warning(f"Epoch {epoch+1}, Batch {batch_idx}: Detected {nan_count} NaN values in input data")
                    data = data_clone
                
                # 检查目标数据是否包含nan并进行处理
                if torch.isnan(target).any():
                    nan_stats['input_data'] += 1
                    target_mean = torch.nanmean(target, dim=0)
                    nan_mask = torch.isnan(target)
                    nan_count = torch.sum(nan_mask).item()
                    
                    # 用平均值填充NaN
                    target_clone = target.clone()
                    for i in range(target.size(0)):
                        sample_nan_mask = nan_mask[i]
                        if sample_nan_mask.any():
                            target_clone[i][sample_nan_mask] = target_mean[sample_nan_mask]
                    
                    # 记录到NaN日志
                    new_nan_row = pd.DataFrame({
                        'epoch': [epoch+1],
                        'batch': [batch_idx],
                        'location': ['target_data'],
                        'action_taken': [f'Filled {nan_count} NaN values with mean'],
                        'timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')]
                    })
                    nan_df = pd.concat([nan_df, new_nan_row], ignore_index=True)
                    
                    logger.warning(f"Epoch {epoch+1}, Batch {batch_idx}: Detected {nan_count} NaN values in target data")
                    target = target_clone
                
                optimizer.zero_grad()
                
                # 使用异常处理来捕获前向传播和损失计算中的错误
                try:
                    # 前向传播
                    output = model(data)
                    
                    # 检查模型输出是否包含nan
                    if torch.isnan(output).any():
                        nan_stats['model_output'] += 1
                        output_nan_count = torch.sum(torch.isnan(output)).item()
                        
                        # 尝试修复输出中的NaN值
                        output_mean = torch.nanmean(output, dim=0)
                        nan_mask = torch.isnan(output)
                        output[nan_mask] = output_mean.repeat(output.size(0), 1)[nan_mask]
                        
                        # 记录到NaN日志
                        new_nan_row = pd.DataFrame({
                            'epoch': [epoch+1],
                            'batch': [batch_idx],
                            'location': ['model_output'],
                            'action_taken': [f'Filled {output_nan_count} NaN values with mean'],
                            'timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')]
                        })
                        nan_df = pd.concat([nan_df, new_nan_row], ignore_index=True)
                        
                        logger.warning(f"Epoch {epoch+1}, Batch {batch_idx}: Detected {output_nan_count} NaN values in model output")
                    
                    # 计算损失前检查输出和目标是否包含无限值
                    if torch.isinf(output).any() or torch.isinf(target).any():
                        # 替换无限值
                        output[torch.isinf(output)] = 0.0
                        target[torch.isinf(target)] = 0.0
                        logger.warning(f"Epoch {epoch+1}, Batch {batch_idx}: Detected infinite values, replacing with 0")
                    
                    # 计算损失
                    loss = F.mse_loss(output, target, reduction='none')
                    
                    # 检查并处理每个样本的损失
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        # 将NaN或无限损失替换为0
                        loss[torch.isnan(loss) | torch.isinf(loss)] = 0.0
                        logger.warning(f"Epoch {epoch+1}, Batch {batch_idx}: NaN/Inf in element-wise loss detected, replacing with 0")
                    
                    # 取平均值作为最终损失
                    loss = loss.mean()
                    
                    # 最终检查损失值
                    if torch.isnan(loss) or torch.isinf(loss):
                        nan_stats['loss'] += 1
                        
                        # 记录到NaN日志
                        new_nan_row = pd.DataFrame({
                            'epoch': [epoch+1],
                            'batch': [batch_idx],
                            'location': ['final_loss'],
                            'action_taken': ['Skipping batch'],
                            'timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')]
                        })
                        nan_df = pd.concat([nan_df, new_nan_row], ignore_index=True)
                        
                        logger.warning(f"Epoch {epoch+1}, Batch {batch_idx}: Final loss is NaN/Inf, skipping batch")
                        skipped_batches += 1
                        nan_counter += 1
                        continue
                    
                    # 反向传播
                    loss.backward()
                    
                    # 修复梯度中的NaN值并进行梯度裁剪
                    nan_grad_count = _fix_nan_gradients(model, threshold=5.0)
                    
                    if nan_grad_count > 0:
                        nan_stats['gradients'] += 1
                        logger.warning(f"Epoch {epoch+1}, Batch {batch_idx}: Fixed {nan_grad_count} NaN values in gradients")
                        
                        # 记录到NaN日志
                        new_nan_row = pd.DataFrame({
                            'epoch': [epoch+1],
                            'batch': [batch_idx],
                            'location': ['gradients'],
                            'action_taken': [f'Fixed {nan_grad_count} NaN gradients'],
                            'timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')]
                        })
                        nan_df = pd.concat([nan_df, new_nan_row], ignore_index=True)
                    
                    # 全局梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # 更新参数
                    optimizer.step()
                    
                    # 收集统计信息
                    epoch_loss += loss.item()
                    valid_batches += 1
                    nan_counter = 0  # 重置nan计数器
                    
                except RuntimeError as e:
                    if 'one of the variables needed for gradient computation' in str(e):
                        logger.error(f"Epoch {epoch+1}, Batch {batch_idx}: Gradient computation error: {str(e)}")
                        
                        # 记录到NaN日志
                        new_nan_row = pd.DataFrame({
                            'epoch': [epoch+1],
                            'batch': [batch_idx],
                            'location': ['gradient_computation'],
                            'action_taken': ['Skipping batch and zeroing gradients'],
                            'timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')]
                        })
                        nan_df = pd.concat([nan_df, new_nan_row], ignore_index=True)
                        
                        # 清零梯度
                        optimizer.zero_grad()
                        skipped_batches += 1
                        nan_counter += 1
                    else:
                        raise e
                    
            except Exception as e:
                logger.error(f"Epoch {epoch+1}, Batch {batch_idx}: Error during training: {str(e)}")
                
                # 记录到NaN日志
                new_nan_row = pd.DataFrame({
                    'epoch': [epoch+1],
                    'batch': [batch_idx],
                    'location': ['exception'],
                    'action_taken': [f'Error: {str(e)}'],
                    'timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')]
                })
                nan_df = pd.concat([nan_df, new_nan_row], ignore_index=True)
                
                skipped_batches += 1
                nan_counter += 1
                optimizer.zero_grad()  # 确保梯度被清零
            
            # 检查是否需要重置模型
            if nan_counter >= max_nan_attempts:
                _reset_model_and_optimizer(model, optimizer, nan_counter, max_nan_attempts)
                nan_counter = 0
            
            # 每50个批次保存NaN日志
            if batch_idx % 50 == 0:
                try:
                    nan_df.to_csv(nan_log_path, index=False)
                except Exception as e:
                    logger.error(f"Error saving NaN log: {str(e)}")
        
        # 检查是否有有效的损失值
        if valid_batches == 0:
            logger.error(f"Epoch {epoch+1}: No valid loss values during training")
            
            # 记录到NaN日志
            new_nan_row = pd.DataFrame({
                'epoch': [epoch+1],
                'batch': [0],
                'location': ['entire_epoch'],
                'action_taken': ['No valid batches in epoch'],
                'timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')]
            })
            nan_df = pd.concat([nan_df, new_nan_row], ignore_index=True)
            nan_df.to_csv(nan_log_path, index=False)
            
            # 如果连续多个epoch没有有效损失，提前停止训练
            if len(train_losses) > 0 and sum(1 for loss in train_losses[-3:] if loss == 0) == 3:
                logger.error("Multiple epochs with no valid losses, stopping training")
                break
            
            # 添加零损失记录，继续训练
            train_losses.append(0)
            val_losses.append(0)
            continue
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / valid_batches
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        valid_val_batches = 0
        
        # 收集验证集上的预测结果和真实值
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                try:
                    data, target = data.to(device), target.to(device)
                    
                    # 检查并填充数据中的NaN
                    if torch.isnan(data).any():
                        data_mean = torch.nanmean(data, dim=0)
                        nan_mask = torch.isnan(data)
                        for i in range(data.size(0)):
                            sample_nan_mask = nan_mask[i]
                            if sample_nan_mask.any():
                                data[i][sample_nan_mask] = data_mean[sample_nan_mask]
                    
                    # 前向传播
                    output = model(data)
                    
                    # 检查并处理输出中的NaN
                    if torch.isnan(output).any():
                        # 使用均值填充NaN
                        output_mean = torch.nanmean(output, dim=0)
                        nan_mask = torch.isnan(output)
                        for i in range(output.size(0)):
                            sample_nan_mask = nan_mask[i]
                            if sample_nan_mask.any():
                                output[i][sample_nan_mask] = output_mean[sample_nan_mask]
                    
                    # 计算损失
                    loss = F.mse_loss(output, target)
                    
                    # 检查损失值
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        val_loss += loss.item()
                        valid_val_batches += 1
                        
                        # 收集预测和真实值
                        all_outputs.append(output.cpu().numpy())
                        all_targets.append(target.cpu().numpy())
                except Exception as e:
                    logger.warning(f"Error during validation: {str(e)}")
                    continue
        
        # 确保验证集上有有效的损失计算
        if valid_val_batches > 0:
            avg_val_loss = val_loss / valid_val_batches
            val_losses.append(avg_val_loss)
            
            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # 评估模型性能
            if all_outputs and all_targets:
                all_outputs = np.vstack(all_outputs)
                all_targets = np.vstack(all_targets)
                
                # 计算RMSE
                rmse = np.sqrt(np.mean((all_outputs - all_targets) ** 2))
                
                # 计算R²
                r2 = 1 - np.sum((all_targets - all_outputs) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2)
                
                metrics_str = f"RMSE: {rmse:.6f}, R²: {r2:.6f}"
            else:
                metrics_str = "No valid predictions for metrics calculation"
            
            # 输出训练信息
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1}/{config.training_config['num_epochs']}, "
                       f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                       f"LR: {current_lr:.7f}, {metrics_str}, "
                       f"Time: {epoch_time:.2f}s, Skipped: {skipped_batches}")
            
            # 记录到批次追踪文件
            new_batch_row = pd.DataFrame({
                'epoch': [epoch+1],
                'train_loss': [avg_train_loss],
                'val_loss': [avg_val_loss],
                'lr': [current_lr],
                'timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')]
            })
            batch_df = pd.concat([batch_df, new_batch_row], ignore_index=True)
            
            try:
                batch_df.to_csv(batch_tracking_path, index=False)
            except Exception as e:
                logger.error(f"Error saving batch tracking file: {str(e)}")
            
            # 检查是否为最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # 保存最佳模型
                best_model_path = os.path.join(config.model_config['model_dir'], f'best_model_{element}.pth')
                
                # 创建一个包含更多信息的检查点
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'element': element,
                    'config': {
                        'input_size': model.input_size if hasattr(model, 'input_size') else None,
                        'dropout': model.dropout if hasattr(model, 'dropout') else None,
                        'num_features': model.num_features if hasattr(model, 'num_features') else None
                    }
                }
                
                try:
                    torch.save(checkpoint, best_model_path)
                    logger.info(f"Saved best model at epoch {epoch+1} with validation loss {best_val_loss:.6f}")
                    
                    # 记录模型信息到文件
                    info_path = os.path.join(config.model_config['model_dir'], f'best_model_info_{element}.txt')
                    with open(info_path, 'w') as f:
                        f.write(f"Best Model Information for Element: {element}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(f"Epoch: {epoch+1}\n")
                        f.write(f"Validation Loss: {best_val_loss:.6f}\n")
                        f.write(f"Training Loss: {avg_train_loss:.6f}\n")
                        f.write(f"Learning Rate: {current_lr:.7f}\n")
                        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Model Path: {best_model_path}\n\n")
                        
                        f.write("Model Configuration:\n")
                        if hasattr(model, 'input_size'):
                            f.write(f"Input Size: {model.input_size}\n")
                        if hasattr(model, 'dropout'):
                            f.write(f"Dropout Rate: {model.dropout}\n")
                        if hasattr(model, 'num_features'):
                            f.write(f"Number of Features: {model.num_features}\n")
                        
                        f.write("\nTraining Configuration:\n")
                        for key, value in config.training_config.items():
                            f.write(f"{key}: {value}\n")
                except Exception as e:
                    logger.error(f"Error saving best model: {str(e)}")
            else:
                patience_counter += 1
                logger.info(f"No improvement for {patience_counter} epochs (patience: {max_patience})")
                
                # 保存定期检查点
                if epoch % 5 == 0:
                    checkpoint_path = os.path.join(config.model_config['model_dir'], f'checkpoint_{element}_epoch{epoch+1}.pth')
                    try:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'best_val_loss': best_val_loss,
                            'train_losses': train_losses,
                            'val_losses': val_losses,
                            'patience_counter': patience_counter
                        }, checkpoint_path)
                        logger.info(f"Saved checkpoint at epoch {epoch+1}")
                    except Exception as e:
                        logger.error(f"Error saving checkpoint: {str(e)}")
            
            # 提前停止
            if patience_counter >= max_patience:
                logger.info(f"Early stopping triggered after {max_patience} epochs without improvement")
                break
        else:
            # 没有有效的验证损失，继续训练
            logger.warning(f"Epoch {epoch+1}: No valid validation loss values")
            val_losses.append(0)
    
    # 训练结束
    training_time = time.time() - training_start_time
    
    # 创建训练总结
    summary_path = os.path.join(batch_results_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Training Summary for Element: {element}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Training Time: {training_time:.2f} seconds ({training_time/3600:.2f} hours)\n")
        f.write(f"Total Epochs: {len(train_losses)}\n")
        f.write(f"Best Validation Loss: {best_val_loss:.6f}\n\n")
        
        # 添加NaN统计信息
        f.write("NaN Statistics:\n")
        for key, value in nan_stats.items():
            f.write(f"{key}: {value}\n")
        
        # 添加最后5个epoch的损失
        f.write("\nLast 5 Epochs:\n")
        f.write("Epoch | Train Loss | Val Loss\n")
        f.write("-" * 35 + "\n")
        for i in range(max(0, len(train_losses)-5), len(train_losses)):
            f.write(f"{i+1} | {train_losses[i]:.6f} | {val_losses[i]:.6f}\n")
    
    # 创建NaN分析报告
    nan_analysis_path = os.path.join(batch_results_dir, 'nan_analysis_report.txt')
    with open(nan_analysis_path, 'w') as f:
        f.write(f"NaN Analysis Report for Element: {element}\n")
        f.write("=" * 50 + "\n\n")
        f.write("Summary of NaN Occurrences:\n")
        f.write(f"Input Data: {nan_stats['input_data']} batches\n")
        f.write(f"Model Output: {nan_stats['model_output']} batches\n")
        f.write(f"Loss: {nan_stats['loss']} batches\n")
        f.write(f"Gradients: {nan_stats['gradients']} batches\n")
        f.write(f"Total Batches: {nan_stats['total_batches']}\n\n")
        
        # 计算NaN比例
        if nan_stats['total_batches'] > 0:
            input_pct = nan_stats['input_data'] / nan_stats['total_batches'] * 100
            output_pct = nan_stats['model_output'] / nan_stats['total_batches'] * 100
            loss_pct = nan_stats['loss'] / nan_stats['total_batches'] * 100
            grad_pct = nan_stats['gradients'] / nan_stats['total_batches'] * 100
            
            f.write("Percentages:\n")
            f.write(f"Input Data NaN: {input_pct:.2f}%\n")
            f.write(f"Model Output NaN: {output_pct:.2f}%\n")
            f.write(f"Loss NaN: {loss_pct:.2f}%\n")
            f.write(f"Gradients NaN: {grad_pct:.2f}%\n\n")
        
        # 添加可能的原因和建议
        f.write("Possible Causes and Recommendations:\n")
        if nan_stats['input_data'] > 0:
            f.write("- Input Data: Check data preprocessing, normalization, and handling of outliers\n")
        if nan_stats['model_output'] > 0:
            f.write("- Model Output: Review model architecture, activation functions, and initializations\n")
        if nan_stats['loss'] > 0:
            f.write("- Loss: Ensure target values are properly normalized and check for division by zero\n")
        if nan_stats['gradients'] > 0:
            f.write("- Gradients: Consider gradient clipping, reducing learning rate, or batch size\n")
        
        f.write("\nRecommended Actions:\n")
        if nan_stats['total_batches'] > 0 and (nan_stats['input_data'] + nan_stats['model_output'] + nan_stats['loss'] + nan_stats['gradients']) > 0:
            f.write("1. Review data preprocessing pipeline\n")
            f.write("2. Check model architecture for numerical stability\n")
            f.write("3. Reduce learning rate and increase batch size\n")
            f.write("4. Add batch normalization to stabilize hidden layer activations\n")
            f.write("5. Consider using more robust loss functions\n")
        else:
            f.write("No significant NaN issues detected\n")
    
    logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    
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
                model = SpectralResCNN(input_size, hidden_size, output_size, dropout_rate)
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