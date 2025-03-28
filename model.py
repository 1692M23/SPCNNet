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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        
        # 计算padding以保持大小不变
        padding = kernel_size // 2
        
        # 主路径
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                     stride=1, padding=padding),
            nn.BatchNorm1d(out_channels)
        )
        
        # 跳跃连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_block(x)
        out += residual
        out = self.relu(out)
        return out

class SpectralResCNN(nn.Module):
    """
    带残差连接的光谱CNN模型
    
    网络结构：
    1. 输入层：
       - 1D卷积层：处理原始光谱
       - BatchNorm：标准化
       - LeakyReLU激活
    
    2. 残差块组1（低级特征）：
       - 包含2个残差块
       - 每个残差块后增加BatchNorm
       - 最大池化后使用Dropout
    
    3. 残差块组2（中级特征）：
       - 包含2个残差块
       - 每个残差块后增加BatchNorm
       - 最大池化后使用Dropout
    
    4. 残差块组3（高级特征）：
       - 包含2个残差块
       - 每个残差块后增加BatchNorm
       - 最大池化后使用Dropout
    
    全连接层：
    - 每层后使用BatchNorm，LeakyReLU和Dropout
    - 使用更平滑的维度变化
    
    改进：
    1. 更多的BatchNorm层稳定训练
    2. 使用LeakyReLU防止死神经元
    3. 适当降低dropout_rate提高训练稳定性
    4. 全连接层中添加BatchNorm层
    """
    def __init__(self, input_size, dropout_rate=0.3):
        super(SpectralResCNN, self).__init__()
        
        # 输入处理层
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1)
        )
        
        # 残差块组1：提取低级特征
        self.res_block1 = nn.Sequential(
            ResidualBlock(16, 32, kernel_size=3),
            nn.BatchNorm1d(32),  # 增加BatchNorm
            ResidualBlock(32, 32, kernel_size=3),
            nn.BatchNorm1d(32),  # 增加BatchNorm
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )
        
        # 残差块组2：提取中级特征
        self.res_block2 = nn.Sequential(
            ResidualBlock(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),  # 增加BatchNorm
            ResidualBlock(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),  # 增加BatchNorm
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )
        
        # 残差块组3：提取高级特征
        self.res_block3 = nn.Sequential(
            ResidualBlock(64, 96, kernel_size=3),  # 降低通道数，避免维度爆炸
            nn.BatchNorm1d(96),  # 增加BatchNorm
            ResidualBlock(96, 96, kernel_size=3),  # 保持一致的通道数
            nn.BatchNorm1d(96),  # 增加BatchNorm
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )
        
        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层：每层后添加BatchNorm
        self.fc = nn.Sequential(
            nn.Linear(96, 128),  # 从96开始，与res_block3输出匹配
            nn.BatchNorm1d(128),  # 增加BatchNorm
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),  # 增加BatchNorm
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),  # 增加BatchNorm
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 1)
        )
        
        # 初始化权重
        self._initialize_weights()
        
        self.dropout_rate = dropout_rate
    
    def _initialize_weights(self):
        """使用Kaiming初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')  # 改为leaky_relu
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')  # 改为leaky_relu
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, training=False):
        """
        前向传播
        Args:
            x: 输入光谱，形状为(batch_size, channels, n_pixels)
            training: 是否使用训练模式（用于MC-Dropout）
        Returns:
            预测的元素丰度值
        """
        # 输入处理
        x = self.input_conv(x)
        
        # 残差块处理
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # 全局池化
        x = self.adaptive_pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
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
            model = SpectralResCNN(input_size, dropout_rate=dropout_rate)
            self.models.append(model)
    
    def train(self, train_loader, val_loader, element, config):
        """训练所有模型"""
        results = []
        for i, model in enumerate(self.models):
            logger.info(f"训练模型 {i+1}/{self.num_models}...")
            train_losses, val_losses = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                element=f"{element}_model{i+1}",
                config=config
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
            logger.warning("检测到 nan 损失值，跳过当前批次")
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

def train_model(model, train_loader, val_loader, element, config):
    """
    训练模型
    
    参数:
        model (torch.nn.Module): 模型
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        element (str): 元素名称
        config: 配置对象
        
    返回:
        tuple: (train_losses, val_losses) 训练和验证损失列表
    """
    device = config.training_config['device']
    model = model.to(device)
    
    # 使用AdamW优化器，添加梯度裁剪
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training_config['lr'],
        weight_decay=config.training_config['weight_decay'],
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
    max_nan_attempts = 3
    max_patience = config.training_config['early_stopping_patience']
    
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
    batch_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'lr', 'timestamp'])
    
    # 创建NaN追踪日志
    nan_log_path = os.path.join(batch_results_dir, 'nan_tracking.csv')
    nan_df = pd.DataFrame(columns=['epoch', 'batch', 'location', 'action_taken', 'timestamp'])
    
    # 训练循环
    for epoch in range(config.training_config['num_epochs']):
        model.train()
        epoch_loss = 0
        valid_batches = 0
        
        # 训练阶段
        for batch_idx, (data, target) in enumerate(train_loader):
            nan_stats['total_batches'] += 1
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
                    'action_taken': [f'填充了{nan_count}个NaN值'],
                    'timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')]
                })
                nan_df = pd.concat([nan_df, new_nan_row], ignore_index=True)
                nan_df.to_csv(nan_log_path, index=False)
                
                logger.warning(f"输入数据中检测到{nan_count}个NaN值并已用均值填充")
                data = data_clone
            
            optimizer.zero_grad()
            
            try:
                # 前向传播
                output = model(data)
                
                # 检查模型输出是否包含nan
                if torch.isnan(output).any():
                    nan_stats['model_output'] += 1
                    output_nan_count = torch.sum(torch.isnan(output)).item()
                    
                    # 记录到NaN日志
                    new_nan_row = pd.DataFrame({
                        'epoch': [epoch+1],
                        'batch': [batch_idx],
                        'location': ['model_output'],
                        'action_taken': ['跳过批次'],
                        'timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')]
                    })
                    nan_df = pd.concat([nan_df, new_nan_row], ignore_index=True)
                    nan_df.to_csv(nan_log_path, index=False)
                    
                    logger.warning(f"模型输出中检测到{output_nan_count}个NaN值，跳过当前批次")
                    nan_counter += 1
                    if nan_counter >= max_nan_attempts:
                        _reset_model_and_optimizer(model, optimizer, nan_counter, max_nan_attempts)
                        nan_counter = 0
                    continue
                
                # 计算损失
                loss = F.mse_loss(output, target)
                
                # 检查损失值是否为nan
                if torch.isnan(loss):
                    nan_stats['loss'] += 1
                    
                    # 记录到NaN日志
                    new_nan_row = pd.DataFrame({
                        'epoch': [epoch+1],
                        'batch': [batch_idx],
                        'location': ['loss'],
                        'action_taken': ['跳过批次'],
                        'timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')]
                    })
                    nan_df = pd.concat([nan_df, new_nan_row], ignore_index=True)
                    nan_df.to_csv(nan_log_path, index=False)
                    
                    logger.warning(f"检测到nan损失值，跳过当前批次")
                    nan_counter += 1
                    if nan_counter >= max_nan_attempts:
                        _reset_model_and_optimizer(model, optimizer, nan_counter, max_nan_attempts)
                        nan_counter = 0
                    continue
                
                # 反向传播
                loss.backward()
                
                # 检查梯度是否包含nan
                nan_in_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        nan_in_grad = True
                        nan_stats['gradients'] += 1
                        
                        # 将NaN梯度置零
                        nan_mask = torch.isnan(param.grad)
                        param.grad[nan_mask] = 0.0
                        
                        grad_nan_count = torch.sum(nan_mask).item()
                        logger.warning(f"参数'{name}'的梯度中检测到{grad_nan_count}个NaN值，已将其置零")
                
                if nan_in_grad:
                    # 记录到NaN日志
                    new_nan_row = pd.DataFrame({
                        'epoch': [epoch+1],
                        'batch': [batch_idx],
                        'location': ['gradients'],
                        'action_taken': ['将NaN梯度置零'],
                        'timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')]
                    })
                    nan_df = pd.concat([nan_df, new_nan_row], ignore_index=True)
                    nan_df.to_csv(nan_log_path, index=False)
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 更新参数
                optimizer.step()
                
                epoch_loss += loss.item()
                valid_batches += 1
                nan_counter = 0  # 重置nan计数器
                
            except Exception as e:
                logger.error(f"训练批次时出错: {str(e)}")
                
                # 记录到NaN日志
                new_nan_row = pd.DataFrame({
                    'epoch': [epoch+1],
                    'batch': [batch_idx],
                    'location': ['exception'],
                    'action_taken': [f'错误: {str(e)}'],
                    'timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')]
                })
                nan_df = pd.concat([nan_df, new_nan_row], ignore_index=True)
                nan_df.to_csv(nan_log_path, index=False)
                
                continue
        
        # 检查是否有有效的损失值
        if valid_batches == 0:
            logger.error("训练过程中没有有效的损失值")
            
            # 记录到NaN日志
            new_nan_row = pd.DataFrame({
                'epoch': [epoch+1],
                'batch': [0],
                'location': ['entire_epoch'],
                'action_taken': ['跳过整个epoch'],
                'timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')]
            })
            nan_df = pd.concat([nan_df, new_nan_row], ignore_index=True)
            nan_df.to_csv(nan_log_path, index=False)
            
            return [], []
        
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
                data, target = data.to(device), target.to(device)
                
                # 检查并填充数据中的NaN
                if torch.isnan(data).any():
                    data_mean = torch.nanmean(data, dim=0)
                    nan_mask = torch.isnan(data)
                    data[nan_mask] = data_mean.repeat(data.size(0), 1)[nan_mask]
                    
                try:
                    # 前向传播
                    output = model(data)
                    
                    # 检查并处理输出中的NaN
                    if torch.isnan(output).any():
                        # 用零替换输出中的NaN
                        output[torch.isnan(output)] = 0.0
                    
                    # 计算损失
                    loss = F.mse_loss(output, target)
                    
                    # 检查损失值
                    if torch.isnan(loss):
                        continue
                    
                    # 收集非NaN预测结果和真实值
                    valid_indices = ~torch.isnan(output).any(dim=1)
                    if valid_indices.any():
                        all_outputs.append(output[valid_indices].cpu().numpy())
                        all_targets.append(target[valid_indices].cpu().numpy())
                    
                    val_loss += loss.item()
                    valid_val_batches += 1
                    
                except Exception as e:
                    logger.error(f"验证批次时出错: {str(e)}")
                    continue
        
        # 检查是否有有效的验证损失值
        if valid_val_batches == 0:
            logger.error("验证过程中没有有效的损失值")
            return [], []
        
        # 计算平均验证损失
        avg_val_loss = val_loss / valid_val_batches
        val_losses.append(avg_val_loss)
        
        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # 生成当前epoch的评估结果和可视化
        if len(all_outputs) > 0 and len(all_targets) > 0:
            try:
                # 合并所有批次的结果
                all_outputs = np.concatenate(all_outputs)
                all_targets = np.concatenate(all_targets)
                
                # 计算评估指标
                mse = np.mean((all_outputs - all_targets) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(all_outputs - all_targets))
                r2 = r2_score(all_targets.flatten(), all_outputs.flatten())
                scatter = np.std(all_outputs - all_targets)
                
                # 在评估指标中添加NaN统计
                metrics_path = os.path.join(batch_results_dir, f'epoch_{epoch+1}_metrics.txt')
                with open(metrics_path, 'w') as f:
                    f.write(f"Epoch {epoch+1} 在 {element} 上的评估结果\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"训练损失: {avg_train_loss:.6f}\n")
                    f.write(f"验证损失: {avg_val_loss:.6f}\n")
                    f.write(f"MSE: {mse:.6f}\n")
                    f.write(f"RMSE: {rmse:.6f}\n")
                    f.write(f"MAE: {mae:.6f}\n")
                    f.write(f"R²: {r2:.6f}\n")
                    f.write(f"散度: {scatter:.6f}\n")
                    f.write(f"学习率: {current_lr:.8f}\n")
                    f.write("\nNaN统计:\n")
                    f.write(f"总批次数: {nan_stats['total_batches']}\n")
                    f.write(f"NaN输入数据: {nan_stats['input_data']} ({nan_stats['input_data']/max(1, nan_stats['total_batches'])*100:.2f}%)\n")
                    f.write(f"NaN模型输出: {nan_stats['model_output']} ({nan_stats['model_output']/max(1, nan_stats['total_batches'])*100:.2f}%)\n") 
                    f.write(f"NaN损失值: {nan_stats['loss']} ({nan_stats['loss']/max(1, nan_stats['total_batches'])*100:.2f}%)\n")
                    f.write(f"NaN梯度: {nan_stats['gradients']} ({nan_stats['gradients']/max(1, nan_stats['total_batches'])*100:.2f}%)\n")
                
                # 生成散点图
                plt.figure(figsize=(10, 6))
                plt.scatter(all_targets, all_outputs, alpha=0.5)
                plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')
                plt.xlabel('真实值')
                plt.ylabel('预测值')
                plt.title(f'Epoch {epoch+1} 预测 vs 真实值 (RMSE: {rmse:.4f})')
                plt.grid(True)
                plt.tight_layout()
                scatter_path = os.path.join(batch_results_dir, f'epoch_{epoch+1}_scatter.png')
                plt.savefig(scatter_path)
                plt.close()
                
                # 更新批次追踪文件
                new_row = pd.DataFrame({
                    'epoch': [epoch+1],
                    'train_loss': [avg_train_loss],
                    'val_loss': [avg_val_loss],
                    'lr': [current_lr],
                    'rmse': [rmse],
                    'mae': [mae],
                    'r2': [r2],
                    'scatter': [scatter],
                    'nan_input_rate': [nan_stats['input_data']/max(1, nan_stats['total_batches'])*100],
                    'nan_output_rate': [nan_stats['model_output']/max(1, nan_stats['total_batches'])*100],
                    'timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')]
                })
                
                batch_df = pd.concat([batch_df, new_row], ignore_index=True)
                batch_df.to_csv(batch_tracking_path, index=False)
                
                # 生成训练进度趋势图
                if len(batch_df) > 1:
                    plt.figure(figsize=(12, 10))
                    
                    plt.subplot(3, 2, 1)
                    plt.plot(batch_df['epoch'], batch_df['train_loss'], 'b-', label='训练损失')
                    plt.plot(batch_df['epoch'], batch_df['val_loss'], 'r-', label='验证损失')
                    plt.xlabel('Epoch')
                    plt.ylabel('损失')
                    plt.title('训练/验证损失趋势')
                    plt.legend()
                    plt.grid(True)
                    
                    plt.subplot(3, 2, 2)
                    plt.plot(batch_df['epoch'], batch_df['rmse'], 'g-')
                    plt.xlabel('Epoch')
                    plt.ylabel('RMSE')
                    plt.title('RMSE趋势')
                    plt.grid(True)
                    
                    plt.subplot(3, 2, 3)
                    plt.plot(batch_df['epoch'], batch_df['mae'], 'm-')
                    plt.xlabel('Epoch')
                    plt.ylabel('MAE')
                    plt.title('MAE趋势')
                    plt.grid(True)
                    
                    plt.subplot(3, 2, 4)
                    plt.plot(batch_df['epoch'], batch_df['r2'], 'c-')
                    plt.xlabel('Epoch')
                    plt.ylabel('R²')
                    plt.title('R²趋势')
                    plt.grid(True)
                    
                    plt.subplot(3, 2, 5)
                    plt.plot(batch_df['epoch'], batch_df['scatter'], 'y-')
                    plt.xlabel('Epoch')
                    plt.ylabel('散度')
                    plt.title('散度趋势')
                    plt.grid(True)
                    
                    plt.subplot(3, 2, 6)
                    plt.plot(batch_df['epoch'], batch_df['lr'], 'k-')
                    plt.xlabel('Epoch')
                    plt.ylabel('学习率')
                    plt.title('学习率趋势')
                    plt.yscale('log')
                    plt.grid(True)
                    
                    plt.tight_layout()
                    trends_path = os.path.join(batch_results_dir, 'training_trends.png')
                    plt.savefig(trends_path)
                    plt.close()
                
                logger.info(f"成功生成Epoch {epoch+1}的评估结果和可视化")
                
            except Exception as e:
                logger.error(f"生成Epoch {epoch+1}评估结果时出错: {str(e)}")
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            _save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, element, {
                'output': {'model_dir': config.model_config['model_dir']}
            })
            
            # 标记这是最佳模型
            best_model_path = os.path.join(batch_results_dir, 'best_model_info.txt')
            with open(best_model_path, 'w') as f:
                f.write(f"最佳模型 - Epoch {epoch+1}\n")
                f.write(f"验证损失: {best_val_loss:.6f}\n")
                f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"早停：{max_patience} 个epoch没有改善")
                break
        
        # 每10个epoch记录一次损失
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/{config.training_config["num_epochs"]}] '
                       f'Train Loss: {avg_train_loss:.6f} '
                       f'Val Loss: {avg_val_loss:.6f}')
    
    # 生成训练完成的汇总报告
    if len(batch_df) > 0:
        try:
            # 找出最佳epoch
            best_epoch = batch_df.loc[batch_df['val_loss'].idxmin(), 'epoch']
            best_val_loss = batch_df['val_loss'].min()
            best_rmse = batch_df.loc[batch_df['val_loss'].idxmin(), 'rmse']
            
            # 保存最终汇总报告
            summary_path = os.path.join(batch_results_dir, 'training_summary.txt')
            with open(summary_path, 'w') as f:
                f.write(f"{element} 元素训练总结\n")
                f.write("=" * 50 + "\n")
                f.write(f"总训练轮次: {len(batch_df)}\n")
                f.write(f"最佳Epoch: {best_epoch} (验证损失: {best_val_loss:.6f}, RMSE: {best_rmse:.6f})\n")
                f.write(f"初始学习率: {batch_df['lr'].iloc[0]:.8f}\n")
                f.write(f"最终学习率: {batch_df['lr'].iloc[-1]:.8f}\n")
                f.write("\n训练详情:\n")
                
                # 添加每10个epoch的详细信息
                step = 10 if len(batch_df) > 50 else max(1, len(batch_df) // 5)
                for i in range(0, len(batch_df), step):
                    epoch_data = batch_df.iloc[i]
                    f.write(f"Epoch {epoch_data['epoch']}: 训练损失={epoch_data['train_loss']:.6f}, 验证损失={epoch_data['val_loss']:.6f}, RMSE={epoch_data['rmse']:.6f}\n")
            
            logger.info(f"已生成训练汇总报告: {summary_path}")
            
        except Exception as e:
            logger.error(f"生成训练汇总报告时出错: {str(e)}")
    
    # 生成NaN值分析报告
    nan_report_path = os.path.join(batch_results_dir, 'nan_analysis_report.txt')
    with open(nan_report_path, 'w') as f:
        f.write(f"{element} 元素NaN值分析报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"总批次数: {nan_stats['total_batches']}\n\n")
        f.write("NaN出现位置统计:\n")
        f.write(f"输入数据中的NaN: {nan_stats['input_data']} ({nan_stats['input_data']/max(1, nan_stats['total_batches'])*100:.2f}%)\n")
        f.write(f"模型输出中的NaN: {nan_stats['model_output']} ({nan_stats['model_output']/max(1, nan_stats['total_batches'])*100:.2f}%)\n")
        f.write(f"损失值中的NaN: {nan_stats['loss']} ({nan_stats['loss']/max(1, nan_stats['total_batches'])*100:.2f}%)\n")
        f.write(f"梯度中的NaN: {nan_stats['gradients']} ({nan_stats['gradients']/max(1, nan_stats['total_batches'])*100:.2f}%)\n\n")
        
        f.write("可能的NaN原因和建议:\n")
        if nan_stats['input_data'] > 0:
            f.write("- 输入数据中存在NaN: 检查数据预处理步骤，确保所有数据都已正确归一化和清洗\n")
        if nan_stats['model_output'] > 0:
            f.write("- 模型输出中存在NaN: 可能是由于数值溢出或激活函数问题，考虑使用更稳定的激活函数或调整网络结构\n")
        if nan_stats['loss'] > 0:
            f.write("- 损失值中存在NaN: 检查损失函数计算过程，调整学习率或使用更鲁棒的损失函数\n")
        if nan_stats['gradients'] > 0:
            f.write("- 梯度中存在NaN: 考虑使用更小的学习率或增加梯度裁剪力度，避免梯度爆炸\n")
        
        if sum([nan_stats['input_data'], nan_stats['model_output'], nan_stats['loss'], nan_stats['gradients']]) == 0:
            f.write("- 训练过程中未检测到NaN值，模型数值稳定性良好\n")
        
    logger.info(f"已生成NaN值分析报告: {nan_report_path}")
    
    return train_losses, val_losses

# 新增的辅助函数，用于重置模型和优化器
def _reset_model_and_optimizer(model, optimizer, nan_counter, max_nan_attempts):
    """重置模型参数和学习率"""
    logger.warning(f"连续 {max_nan_attempts} 次出现nan值，重置模型和学习率")
    
    # 重置模型参数
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    
    # 降低学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1
        
    # 记录降低后的学习率
    new_lr = optimizer.param_groups[0]['lr']
    logger.info(f"学习率已降低至 {new_lr:.8f}")

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
                logger.error(f"评估过程中出错: {str(e)}")
                continue
    
    # 如果出现了NaN值，记录日志
    if nan_stats['input'] > 0 or nan_stats['output'] > 0:
        logger.warning(f"评估过程中共处理: {nan_stats['total_samples']}个样本")
        logger.warning(f"输入数据中的NaN值: {nan_stats['input']}个")
        logger.warning(f"模型输出中的NaN值: {nan_stats['output']}个")
    
    # 合并结果，计算指标
    if len(all_outputs) > 0 and len(all_targets) > 0:
        try:
            all_outputs = np.vstack(all_outputs)
            all_targets = np.vstack(all_targets)
            
            # 检查连接后的数据是否仍包含NaN
            if np.isnan(all_outputs).any() or np.isnan(all_targets).any():
                logger.warning("合并后的评估数据仍包含NaN值，将尝试过滤")
                # 过滤掉包含NaN的行
                valid_rows = ~np.isnan(all_outputs).any(axis=1) & ~np.isnan(all_targets).any(axis=1)
                all_outputs = all_outputs[valid_rows]
                all_targets = all_targets[valid_rows]
                
                if len(all_outputs) == 0:
                    logger.error("过滤NaN后没有有效数据可用于评估")
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
            logger.error(f"计算评估指标时出错: {str(e)}")
    
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
    
    logger.warning(f"{name}中检测到{nan_count}个NaN值，已使用{replacement_strategy}策略替换")
    
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
                logger.error(f"预测过程中出错: {str(e)}")
                # 在异常情况下，生成一个全零的假输出
                fake_output = torch.zeros_like(inputs[:, 0:1])  # 假设输出是一维的
                predictions.append(fake_output.cpu().numpy())
    
    # 如果出现了NaN值，记录日志
    if nan_stats['input'] > 0 or nan_stats['output'] > 0:
        logger.warning(f"预测过程中共处理: {nan_stats['total_samples']}个样本")
        logger.warning(f"输入数据中的NaN值: {nan_stats['input']}个")
        logger.warning(f"模型输出中的NaN值: {nan_stats['output']}个")
    
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
        # 尝试加载模型
        checkpoint = torch.load(model_path, map_location=device)
        
        # 检查checkpoint是否包含模型架构信息
        if 'model_state_dict' in checkpoint:
            # 创建模型实例
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                input_size = config.get('input_size', 1024)
                hidden_size = config.get('hidden_size', 256)
                output_size = config.get('output_size', 1)
                dropout_rate = config.get('dropout_rate', 0.3)
                model = SpectralResCNN(input_size, hidden_size, output_size, dropout_rate)
            else:
                # 使用默认参数
                logger.warning(f"模型配置信息缺失，使用默认参数创建模型")
                model = SpectralResCNN()
                
            # 加载模型权重
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 老版本的checkpoint可能直接保存了模型状态
            model = SpectralResCNN()
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()  # 设置为评估模式
        logger.info(f"成功从 {model_path} 加载模型")
        return model
        
    except Exception as e:
        logger.error(f"加载模型时出错: {str(e)}")
        # 创建一个空模型作为后备选择
        backup_model = SpectralResCNN()
        backup_model.to(device)
        backup_model.eval()
        logger.warning("使用未训练的模型作为后备")
        return backup_model

# 在模型训练完成后添加
def analyze_model_performance(self, element, train_loader, val_loader, test_loader):
    """对训练好的模型进行全面性能分析"""
    # 创建结果目录
    os.makedirs("results/feature_importance", exist_ok=True)
    os.makedirs("results/residual_analysis", exist_ok=True)
    
    logger.info(f"开始分析{element}模型的特征重要性...")
    self.analyze_feature_importance(self.model, val_loader, self.device, element)
    
    logger.info(f"开始分析{element}模型的残差...")
    self.analyze_residuals(self.model, test_loader, self.device, element)
    
    logger.info(f"{element}模型分析完成，结果保存在results目录")

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
    train_losses, val_losses = train_model(
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
        logger.info(f"开始对{element}模型进行性能分析...")
        
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
            logger.info(f"{element}模型性能分析完成, 结果保存在results目录")
        except ImportError:
            logger.warning("无法导入model_analysis模块，跳过性能分析")
    
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
        logger.error(f"加载模型时出错: {str(e)}")
        # 创建一个空模型作为后备选择
        backup_model = SpectralResCNN(input_size)
        backup_model.to(device)
        backup_model.eval()
        logger.warning(f"使用未训练的模型作为后备")
        return backup_model

# 修改原来的load_trained_model函数名为load_trained_model_core，保持旧函数名对旧函数的向后兼容性
load_trained_model_core = load_trained_model
load_trained_model = load_trained_model_compat 