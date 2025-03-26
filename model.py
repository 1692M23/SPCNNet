import torch
import torch.nn as nn
import config
import os
import numpy as np
import logging
import torch.nn.functional as F

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
        T_0=config.training_config['epochs'] // 3,  # 第一次重启的epoch数
        T_mult=2,  # 每次重启后周期的倍数
        eta_min=config.training_config['lr'] * 0.01  # 最小学习率
    )
    
    # 创建模型目录
    os.makedirs(config.model_config['model_dir'], exist_ok=True)
    
    # 初始化变量
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    nan_counter = 0
    max_nan_attempts = 3
    max_patience = config.training_config['early_stopping_patience']
    
    # 训练循环
    for epoch in range(config.training_config['epochs']):
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
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                # 检查输入数据是否包含nan
                if torch.isnan(data).any():
                    continue
                    
                try:
                    output = model(data)
                    loss = F.mse_loss(output, target)
                    
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
        scheduler.step()
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            _save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, element, {
                'output': {'model_dir': config.model_config['model_dir']}
            })
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"早停：{max_patience} 个epoch没有改善")
                break
        
        # 每10个epoch记录一次损失
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/{config.training_config["epochs"]}] '
                       f'Train Loss: {avg_train_loss:.6f} '
                       f'Val Loss: {avg_val_loss:.6f}')
    
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
        input_size (int): 输入维度
        element (str): 元素名称
        config: 配置对象
        
    返回:
        torch.nn.Module: 加载的模型
    """
    model_path = os.path.join(config.model_config['model_dir'], f"{element}_model.pth")
    if not os.path.exists(model_path):
        logger.warning(f"未找到 {element} 的模型文件")
        return None
    
    try:
        # 创建模型实例
        model = SpectralResCNN(input_size).to(config.training_config['device'])
        
        # 加载模型状态
        checkpoint = torch.load(model_path, map_location=config.training_config['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"成功加载 {element} 模型，验证损失: {checkpoint['val_loss']:.6f}")
        return model
        
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        return None

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