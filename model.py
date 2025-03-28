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
       - ReLU激活
    
    2. 残差块组1（低级特征）：
       - 输入通道：32
       - 输出通道：64
       - 包含2个残差块
       - 每个残差块包含：
         * 两个3x1卷积层
         * 每个卷积层后接BatchNorm和ReLU
         * 跳跃连接
    
    3. 残差块组2（中级特征）：
       - 输入通道：64
       - 输出通道：128
       - 包含2个残差块
       - 结构同残差块组1
    
    4. 残差块组3（高级特征）：
       - 输入通道：128
       - 输出通道：256
       - 包含2个残差块
       - 结构同残差块组1
    
    每个残差块组后接：
    - 最大池化：减少特征维度
    - Dropout：防止过拟合
    
    全连接层：
    - 自适应平均池化：统一特征维度
    - 三个全连接层，逐步降维：
      * 256 -> 512
      * 512 -> 256
      * 256 -> 128
      * 128 -> 1
    - 每个全连接层后接ReLU和Dropout
    
    特点：
    1. 使用残差连接，有助于训练更深的网络
    2. 多尺度特征提取，从低级到高级特征
    3. 使用BatchNorm和Dropout防止过拟合
    4. 渐进式降维，保留重要特征
    """
    def __init__(self, input_size, dropout_rate=0.5):
        super(SpectralResCNN, self).__init__()
        
        # 输入处理层：使用更小的卷积核和更少的通道数
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),  # 减少初始通道数
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1)  # 使用LeakyReLU替代ReLU
        )
        
        # 残差块组1：提取低级特征
        self.res_block1 = nn.Sequential(
            ResidualBlock(16, 32, kernel_size=3),  # 减少通道数
            ResidualBlock(32, 32, kernel_size=3),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 减小池化步长
            nn.Dropout(dropout_rate)  # 减少dropout率
        )
        
        # 残差块组2：提取中级特征
        self.res_block2 = nn.Sequential(
            ResidualBlock(32, 64, kernel_size=3),
            ResidualBlock(64, 64, kernel_size=3),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )
        
        # 残差块组3：提取高级特征
        self.res_block3 = nn.Sequential(
            ResidualBlock(64, 128, kernel_size=3),
            ResidualBlock(128, 128, kernel_size=3),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )
        
        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层：使用更平滑的维度变化
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 1)
        )
        
        # 初始化权重
        self._initialize_weights()
        
        self.dropout_rate = dropout_rate
    
    def _initialize_weights(self):
        """使用Kaiming初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        
        # 全连接层（使用功能性dropout以支持MC-Dropout）
        for i, layer in enumerate(self.fc):
            x = layer(x)
            if isinstance(layer, nn.Linear) and i < len(self.fc) - 1:
                x = F.leaky_relu(x, 0.1)
                # 在训练或MC模式下，应用dropout
                if training or self.training:
                    x = F.dropout(x, p=self.dropout_rate, training=True)
        
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
    
    # 创建批次追踪文件
    batch_tracking_path = os.path.join(batch_results_dir, 'batch_tracking.csv')
    batch_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'lr', 'timestamp'])
    
    # 训练循环
    for epoch in range(config.training_config['num_epochs']):
        model.train()
        epoch_loss = 0
        valid_batches = 0
        
        # 训练阶段
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 检查输入数据是否包含nan
            if torch.isnan(data).any():
                logger.warning(f"检测到输入数据包含nan值，跳过当前批次")
                continue
                
            optimizer.zero_grad()
            
            try:
                output = model(data)
                loss = F.mse_loss(output, target)
                
                # 检查损失值是否为nan
                if torch.isnan(loss):
                    logger.warning(f"检测到nan损失值，跳过当前批次")
                    nan_counter += 1
                    if nan_counter >= max_nan_attempts:
                        logger.warning(f"连续 {max_nan_attempts} 次出现nan值，重置模型和学习率")
                        # 重置模型参数
                        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
                        # 降低学习率
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.1
                        nan_counter = 0
                    continue
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                valid_batches += 1
                nan_counter = 0  # 重置nan计数器
                
            except Exception as e:
                logger.error(f"训练批次时出错: {str(e)}")
                continue
        
        # 检查是否有有效的损失值
        if valid_batches == 0:
            logger.error("训练过程中没有有效的损失值")
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
                
                # 检查输入数据是否包含nan
                if torch.isnan(data).any():
                    continue
                    
                try:
                    output = model(data)
                    loss = F.mse_loss(output, target)
                    
                    # 收集预测结果和真实值
                    all_outputs.append(output.cpu().numpy())
                    all_targets.append(target.cpu().numpy())
                    
                    # 检查损失值是否为nan
                    if torch.isnan(loss):
                        continue
                        
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
                r2 = 1 - (np.sum((all_targets - all_outputs) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2))
                scatter = np.std(all_outputs - all_targets)
                
                # 保存评估指标
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
        
    model.eval()
    test_loss = 0
    predictions = []
    targets = []
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for spectra, abundances in test_loader:
            spectra = spectra.to(device)
            abundances = abundances.to(device)
            
            outputs = model(spectra)
            loss = criterion(outputs.squeeze(), abundances)
            
            test_loss += loss.item()
            predictions.extend(outputs.squeeze().cpu().numpy())
            targets.extend(abundances.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 计算评估指标
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2))
    std_diff = np.std(predictions - targets)
    
    return {
        'test_loss': test_loss / len(test_loader),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'dex': std_diff,
        'predictions': predictions,
        'targets': targets
    }

def predict(model, spectra, device):
    """
    模型预测函数
    """
    model.eval()
    with torch.no_grad():
        spectra = spectra.to(device)
        outputs = model(spectra)
        return outputs.squeeze().cpu().numpy()

# =============== 4. 工具函数 ===============
def load_trained_model(input_size, element, config):
    """
    加载训练好的模型
    
    参数:
        input_size (int): 输入大小
        element (str): 元素名称
        config: 配置对象
        
    返回:
        torch.nn.Module: 加载的模型
    """
    device = config.training_config['device']
    
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
            return None
            
        # 创建模型实例
        model = SpectralResCNN(input_size)
        model = model.to(device)
        
        # 加载模型权重
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # 检查加载的文件是否为字典格式
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                logger.info(f"加载带有state_dict的模型: {model_path}")
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 直接加载整个模型
                logger.info(f"加载完整模型: {model_path}")
                model = checkpoint
                model = model.to(device)
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            return None
            
        # 设置为评估模式
        model.eval()
        logger.info(f"成功加载元素 {element} 的模型")
        
        return model
        
    except Exception as e:
        logger.error(f"加载模型时出错: {str(e)}")
        return None

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