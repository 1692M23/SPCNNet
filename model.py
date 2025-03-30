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
import sys

# 处理torch_xla导入问题
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
    try:
        import torch_xla.distributed.parallel_loader as pl
        HAS_PARALLEL_LOADER = True
    except ImportError:
        HAS_PARALLEL_LOADER = False
        print("torch_xla.distributed.parallel_loader导入失败，将禁用并行加载功能")
except ImportError:
    HAS_XLA = False
    HAS_PARALLEL_LOADER = False
    print("torch_xla导入失败，将禁用TPU支持")

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

# 检测TPU是否可用并返回相应工具函数
def is_tpu_available():
    """检查是否可以使用TPU"""
    return HAS_XLA

def move_to_device(tensor, device):
    """
    将张量移动到指定设备，支持TPU、GPU和CPU
    
    参数:
        tensor: 要移动的张量
        device: 目标设备
        
    返回:
        移动后的张量
    """
    if tensor is None:
        return None
        
    # 检查是否已经在目标设备上
    if hasattr(tensor, 'device') and tensor.device == device:
        return tensor
        
    # 针对TPU进行特殊处理
    if str(device).startswith('xla'):
        try:
            if HAS_XLA:
                return xm.send_cpu_data_to_device(tensor, device)
            else:
                logger.warning("尝试使用TPU但torch_xla未安装")
        except Exception as e:
            logger.warning(f"移动数据到TPU设备时出错: {str(e)}")
            pass
            
    # 常规设备移动
    return tensor.to(device)

def sync_device(device=None):
    """
    同步设备操作，确保计算完成
    对于TPU等异步执行的设备尤其重要
    
    参数:
        device: 需要同步的设备
    """
    if device is None:
        return
        
    # 针对TPU的特殊处理
    if str(device).startswith('xla'):
        if HAS_XLA:
            xm.mark_step()
    # 针对GPU的处理
    elif str(device).startswith('cuda'):
        torch.cuda.synchronize(device)

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

def train(model, train_loader, val_loader, num_epochs=50, patience=10, device=None, element=None, config=None):
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
    
    # 确定设备类型
    device_type = 'cpu'
    if str(device).startswith('cuda'):
        device_type = 'cuda'
    elif str(device).startswith('xla'):
        device_type = 'tpu'
    
    logger.info(f"在 {device_type} 设备上训练模型: {device}")
    
    # 创建训练状态恢复文件路径
    training_state_dir = os.path.join(config['model_config']['model_dir'], 'training_states')
    os.makedirs(training_state_dir, exist_ok=True)
    training_state_file = os.path.join(training_state_dir, f'training_state_{element}.json')
    
    # 检查是否存在训练状态文件
    resume_training = False
    current_stage = 1  # 默认从第一阶段开始
    start_epoch = 0    # 默认从第0轮开始
    best_val_loss = float('inf')
    patience_counter = 0
    
    if os.path.exists(training_state_file):
        try:
            import json
            with open(training_state_file, 'r') as f:
                training_state = json.load(f)
                
            current_stage = training_state.get('current_stage', 1)
            start_epoch = training_state.get('current_epoch', 0) + 1  # 从下一轮开始
            best_val_loss = training_state.get('best_val_loss', float('inf'))
            patience_counter = training_state.get('patience_counter', 0)
            
            # 如果上次训练已经完成，重新开始
            if training_state.get('training_completed', False):
                logger.info(f"找到已完成的训练状态文件，将重新开始训练")
                current_stage = 1
                start_epoch = 0
                best_val_loss = float('inf')
                patience_counter = 0
            else:
                logger.info(f"从中断点恢复训练: 阶段{current_stage}, 轮次{start_epoch}")
                resume_training = True
                
                # 如果发现阶段1已经完成，准备开始阶段2
                if current_stage == 1 and training_state.get('stage1_completed', False):
                    logger.info("第一阶段已完成，将从第二阶段开始")
                    current_stage = 2
                    start_epoch = 0
        except Exception as e:
            logger.warning(f"读取训练状态文件失败: {str(e)}，将重新开始训练")
    
    # 设置优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), 
                          lr=config['training']['lr'],
                          weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 如果恢复训练，尝试加载模型和优化器状态
    if resume_training:
        checkpoint_path = os.path.join(config['model_config']['model_dir'], f'checkpoint_{element}.pth')
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info(f"成功加载模型和优化器状态")
            except Exception as e:
                logger.warning(f"加载模型和优化器状态失败: {str(e)}")
    
    # 设置损失函数
    criterion = nn.MSELoss()
    
    # 训练记录
    train_losses = []
    val_losses = []
    
    # 第一阶段或第二阶段的训练过程
    def train_stage(stage, start_from_epoch=0, initial_best_val_loss=float('inf'), initial_patience=0):
        """训练指定阶段"""
        nonlocal train_losses, val_losses
        best_val_loss = initial_best_val_loss
        patience_counter = initial_patience
        stage_completed = False
        
        # 冻结或解冻参数，根据阶段设置
        if stage == 1:
            logger.info("开始第一阶段训练 - 特征提取器")
            # 获取模型的所有参数
            all_params = list(model.parameters())
            # 冻结最后一层的参数
            for param in all_params[-2:]:  # 最后一个线性层的权重和偏置
                param.requires_grad = False
        else:
            logger.info("开始第二阶段训练 - 全模型微调")
            # 解冻所有层
            for param in model.parameters():
                param.requires_grad = True
                
            # 重置优化器和学习率 - 第二阶段使用较小的学习率
            nonlocal optimizer, scheduler
            optimizer = optim.Adam(model.parameters(), 
                                  lr=config['training']['lr'] * 0.1,
                                  weight_decay=config['training']['weight_decay'])
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        
        # 从指定轮次开始训练
        for epoch in range(start_from_epoch, config['training']['num_epochs']):
            model.train()
            total_loss = 0
            batch_count = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # 将数据移动到正确的设备 - 支持TPU/GPU/CPU
                data = move_to_device(data, device)
                target = move_to_device(target, device)
                
                # 检查数据是否包含NaN
                if torch.isnan(data).any() or torch.isnan(target).any():
                    logger.warning("检测到输入数据包含NaN值，跳过该批次")
                    continue
                
                # TPU特定处理：清除之前的梯度
                optimizer.zero_grad()
                
                # 前向传播
                output = model(data)
                loss = criterion(output, target)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 优化器步进
                optimizer.step()
                
                # TPU特定处理：确保梯度更新已应用
                if device_type == 'tpu':
                    if HAS_XLA:
                        xm.mark_step()
                
                total_loss += loss.item()
                batch_count += 1
                
                # 每个批次后添加进度显示 (无论TPU/GPU/CPU)
                if batch_idx % 2 == 0 or batch_idx == len(train_loader) - 1:  # 修改为每2个批次显示一次
                    # 确保TPU上的操作完成后再输出
                    if device_type == 'tpu' and HAS_XLA:
                        import torch_xla.core.xla_model as xm
                        xm.mark_step()
                    
                    # 直接使用print输出，确保控制台可见
                    progress_msg = f"阶段{stage} - Epoch {epoch+1}/{config['training']['num_epochs']} - 批次 {batch_idx+1}/{len(train_loader)} ({(batch_idx+1)*100/len(train_loader):.1f}%) - 损失: {loss.item():.6f}"
                    print(progress_msg, flush=True)  # 添加flush=True强制立即显示
                    logger.info(progress_msg)
            
            # 更新学习率
            scheduler.step()
            
            # 验证
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    # 将数据移动到正确的设备 - 支持TPU/GPU/CPU
                    data = move_to_device(data, device)
                    target = move_to_device(target, device)
                    
                    # 前向传播
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    
                    # TPU特定处理：确保计算已完成
                    if device_type == 'tpu':
                        if HAS_XLA:
                            xm.mark_step()
            
            val_loss /= len(val_loader)
            train_loss = total_loss / batch_count if batch_count > 0 else float('inf')
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # 强制同步点和输出
            if device_type == 'tpu' and HAS_XLA:
                xm.mark_step()

            # 使用print直接输出而不仅仅依赖logger
            print(f"第{stage}阶段 - 轮次: {epoch+1}/{config['training']['num_epochs']}, 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")
            sys.stdout.flush()  # 强制刷新输出缓冲区
            logger.info(f"第{stage}阶段 - 轮次: {epoch+1}/{config['training']['num_epochs']}, 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")
            
            # 更新训练状态并保存
            save_training_state(element, stage, epoch, best_val_loss, patience_counter, False, stage==1 and stage_completed)
            
            # 保存当前检查点，用于恢复训练
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, element, 'checkpoint')
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                save_model(model, optimizer, scheduler, epoch, val_loss, config['model_config']['model_dir'], element)
            else:
                patience_counter += 1
                if patience_counter >= config['training']['early_stopping_patience']:
                    logger.info(f"第{stage}阶段 - 早停触发，最佳验证损失: {best_val_loss:.6f}")
                    stage_completed = True
                    break
        
        # 如果没有提前停止，标记阶段完成
        if not stage_completed and epoch == config['training']['num_epochs'] - 1:
            stage_completed = True
            
        return best_val_loss, stage_completed
    
    # 保存训练状态的辅助函数
    def save_training_state(element, current_stage, current_epoch, best_val_loss, patience_counter, 
                           training_completed=False, stage1_completed=False):
        """保存当前训练状态"""
        import json
        
        training_state = {
            'element': element,
            'current_stage': current_stage,
            'current_epoch': current_epoch,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'training_completed': training_completed,
            'stage1_completed': stage1_completed,
            'timestamp': time.time()
        }
        
        with open(training_state_file, 'w') as f:
            json.dump(training_state, f, indent=4)
    
    # 保存当前检查点的辅助函数，用于恢复训练
    def save_checkpoint(model, optimizer, scheduler, epoch, loss, element, checkpoint_type='checkpoint'):
        """保存训练检查点"""
        if isinstance(config, dict):
            model_dir = config.get('model_config', {}).get('model_dir', 'models')
        else:
            try:
                model_dir = config.model_config['model_dir']
            except:
                model_dir = 'models'
        
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_path = os.path.join(model_dir, f'{checkpoint_type}_{element}.pth')
        
        # 只保存状态字典，不保存整个模型对象
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss
        }
        
        try:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"成功保存检查点: {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"保存检查点失败: {str(e)}")
            return False
    
    # 根据当前阶段进行训练
    if current_stage == 1:
        # 执行第一阶段训练
        best_val_loss, stage1_completed = train_stage(1, start_epoch, best_val_loss, patience_counter)
        
        # 如果第一阶段完成，继续第二阶段
        if stage1_completed:
            save_training_state(element, 1, config['training']['num_epochs']-1, best_val_loss, 0, False, True)
            best_val_loss, _ = train_stage(2, 0, float('inf'), 0)
    else:
        # 从第二阶段继续
        best_val_loss, _ = train_stage(2, start_epoch, best_val_loss, patience_counter)
    
    # 训练全部完成，更新状态
    save_training_state(element, 2, config['training']['num_epochs']-1, best_val_loss, 0, True, True)
    
    # 训练结束，保存最终模型
    # 从配置中获取模型目录，处理配置可能是字典或模块的情况
    if config is not None:
        if isinstance(config, dict):
            model_dir = config.get('model_config', {}).get('model_dir', 'models')
        else:
            model_dir = getattr(config, 'model_config', {}).get('model_dir', 'models')
    else:
        model_dir = 'models'
    
    # 确保目录存在
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, f'SpectralResCNN_GCN_{element}.pth')
    
    try:
        # 只保存状态字典
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"成功保存最终模型: {final_model_path}")
    except Exception as e:
        logger.error(f"保存最终模型失败: {str(e)}")
        # 尝试备用保存方式
        try:
            backup_path = os.path.join(model_dir, f'{element}_model.pth')
            torch.save(model.state_dict(), backup_path)
            logger.info(f"成功保存备用最终模型: {backup_path}")
        except Exception as e2:
            logger.error(f"保存备用最终模型也失败: {str(e2)}")
    
    # 同时保存检查点格式的模型以备后续使用
    if isinstance(config, dict):
        checkpoint_dir = config.get('model_config', {}).get('model_dir', 'models')
    else:
        checkpoint_dir = getattr(config, 'model_config', {}).get('model_dir', 'models')
        
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{element}.pth')
    
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': best_val_loss
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"成功保存检查点: {checkpoint_path}")
    except Exception as e:
        logger.error(f"保存检查点失败: {str(e)}")
    
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
    
    # 确定设备类型
    device_type = 'cpu'
    if str(device).startswith('cuda'):
        device_type = 'cuda'
    elif str(device).startswith('xla'):
        device_type = 'tpu'
    
    logger.info(f"在 {device_type} 设备上评估模型: {device}")
    
    # 确保模型在正确的设备上
    if next(model.parameters()).device != device:
        logger.info(f"将模型从 {next(model.parameters()).device} 移动到 {device}")
        model = move_to_device(model, device)  # 使用通用设备移动函数
    
    model.eval()
    
    all_outputs = []
    all_targets = []
    
    # 添加NaN统计
    nan_stats = {'input': 0, 'output': 0, 'total_samples': 0}
    
    with torch.no_grad():
        for data, target in test_loader:
            nan_stats['total_samples'] += data.size(0)
            
            # 将数据移到相应设备
            data = move_to_device(data, device)
            target = move_to_device(target, device)
            
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
                
                # TPU特定处理：确保计算已完成
                if device_type == 'tpu':
                    if HAS_XLA:
                        xm.mark_step()
                
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
                    # 如果在TPU上，先移动到CPU再转换为numpy
                    if device_type == 'tpu':
                        try:
                            import torch_xla.core.xla_model as xm
                            cpu_output = xm.send_to_host_async(output[valid_indices])
                            cpu_target = xm.send_to_host_async(target[valid_indices])
                            all_outputs.append(cpu_output.numpy())
                            all_targets.append(cpu_target.numpy())
                        except (ImportError, AttributeError):
                            # 降级到标准处理
                            all_outputs.append(output[valid_indices].cpu().numpy())
                            all_targets.append(target[valid_indices].cpu().numpy())
                    else:
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
    """加载训练好的模型，确保正确处理状态字典"""
    try:
        if os.path.exists(model_path):
            # 创建模型实例
            model = SpectralResCNN_GCN()
            
            # 加载状态字典
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # 尝试直接加载，假设是状态字典
                model.load_state_dict(checkpoint)
                
            # 移动到设备
            if device:
                model = model.to(device)
                
            model.eval()
            logger.info(f"成功加载模型: {model_path}")
            return model
        else:
            logger.error(f"模型文件不存在: {model_path}")
            return None
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        return None

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
            adj = self._build_spectral_adjacency(seq_len)
            adj = move_to_device(adj, device)  # 使用通用设备移动函数
            # 归一化邻接矩阵
            adj = self._normalize_adj(adj)
            self.adj_cache[adj_key] = adj
        
        # 使用缓存的邻接矩阵
        adj_matrix = self.adj_cache[adj_key]
        # 确保邻接矩阵在正确的设备上
        if adj_matrix.device != device:
            adj_matrix = move_to_device(adj_matrix, device)  # 使用通用设备移动函数
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
    支持TPU、GPU和CPU设备
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
        
        # 计算注意力得分 - TPU优化版本
        try:
            # 计算注意力得分
            energy = torch.bmm(query, key)  # [B, W, W]
            
            # 为了数值稳定性，在softmax前进行缩放
            energy_scaled = energy / (C ** 0.5)  # 缩放因子为通道数的平方根
            
            # 检查数值问题 - 在TPU上尤其重要
            has_nan = torch.isnan(energy_scaled).any()
            has_inf = torch.isinf(energy_scaled).any()
            
            # 对于TPU，我们避免使用条件判断，改用统一的计算路径
            # 首先应用掩码将NaN和Inf替换为安全值
            if has_nan or has_inf:
                logger.warning("检测到注意力计算中的NaN或Inf值，应用数值稳定化")
                # 创建掩码并替换NaN/Inf值
                nan_mask = torch.isnan(energy_scaled)
                inf_mask = torch.isinf(energy_scaled)
                problem_mask = nan_mask | inf_mask
                
                # 将问题值替换为0
                energy_scaled = torch.where(problem_mask, torch.zeros_like(energy_scaled), energy_scaled)
                
            # 使用数值稳定的softmax计算
            # 1. 计算每行的最大值
            max_val, _ = torch.max(energy_scaled, dim=2, keepdim=True)
            
            # 2. 减去最大值以提高数值稳定性
            energy_safe = energy_scaled - max_val
            
            # 3. 对exp应用掩码以避免NaN
            exp_x = torch.exp(energy_safe)
            
            # 4. 计算和并添加小的epsilon以避免除零
            sum_exp_x = torch.sum(exp_x, dim=2, keepdim=True) + 1e-10  # 添加小的epsilon
            
            # 5. 计算softmax
            attention = exp_x / sum_exp_x
            
            # 在TPU上，我们在计算后进行同步以确保计算完成
            if str(device).startswith('xla'):
                try:
                    import torch_xla.core.xla_model as xm
                    xm.mark_step()
                except ImportError:
                    pass
                
        except Exception as e:
            logger.error(f"注意力计算出错: {str(e)}")
            # 在错误情况下，使用应急注意力（均匀分布）
            attention = torch.ones(batch_size, width, width, device=device) / width
        
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
        
        # 设置设备，支持CPU、GPU和TPU
        if device is None:
            if hasattr(config, 'training_config') and 'device' in config.training_config:
                device = config.training_config['device']
            else:
                # 尝试检测TPU
                if is_tpu_available():
                    try:
                        import torch_xla.core.xla_model as xm
                        device = xm.xla_device()
                        logger.info(f"使用TPU设备: {device}")
                    except Exception as e:
                        logger.warning(f"TPU初始化失败: {str(e)}")
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                else:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    
        self.device = device
        logger.info(f"模型将使用设备: {self.device}")
        
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
        # 确保输入在正确的设备上，支持TPU
        x = move_to_device(x, self.device)
            
        # 获取实际输入维度
        batch_size, channels, seq_len = x.size()
        
        # 如果input_size为None，自动设置为当前序列长度
        if self.input_size is None:
            self.input_size = seq_len
            logger.info(f"自动适应输入大小: {seq_len}")
        
        # 特征提取
        x = self.feature_extractor(x)
        
        # 当在TPU上运行时，可能需要进行同步以确保计算完成
        if str(self.device).startswith('xla'):
            sync_device(self.device)
        
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
        
        # TPU同步点 - 多次重形状操作后
        if str(self.device).startswith('xla'):
            sync_device(self.device)
        
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
        
        # TPU同步点 - 在最终分类前
        if str(self.device).startswith('xla'):
            sync_device(self.device)
        
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

def load_checkpoint(model, optimizer, scheduler, element, checkpoint_type='checkpoint'):
    """加载训练检查点，兼容新旧格式"""
    checkpoint_path = os.path.join(config.model_config['model_dir'], f'{checkpoint_type}_{element}.pth')
    
    if not os.path.exists(checkpoint_path):
        logger.warning(f"找不到检查点文件: {checkpoint_path}")
        return model, optimizer, scheduler, 0, float('inf')
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 处理新格式检查点（包含state_dict）
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            epoch = checkpoint.get('epoch', 0)
            loss = checkpoint.get('loss', float('inf'))
        # 处理旧格式检查点（保存整个模型）
        elif 'model' in checkpoint:
            logger.warning("检测到旧格式检查点，尝试提取状态字典")
            try:
                # 提取状态字典而不是整个模型
                if hasattr(checkpoint['model'], 'state_dict'):
                    model_state_dict = checkpoint['model'].state_dict()
                    model.load_state_dict(model_state_dict)
                else:
                    logger.error("无法从检查点中提取模型状态字典")
                
                # 对优化器和调度器同样处理
                if optimizer is not None and 'optimizer' in checkpoint and hasattr(checkpoint['optimizer'], 'state_dict'):
                    optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
                
                if scheduler is not None and 'scheduler' in checkpoint and hasattr(checkpoint['scheduler'], 'state_dict'):
                    scheduler.load_state_dict(checkpoint['scheduler'].state_dict())
            except Exception as e:
                logger.error(f"从旧格式检查点加载状态字典失败: {str(e)}")
                
            epoch = checkpoint.get('epoch', 0)
            loss = checkpoint.get('loss', float('inf'))
        else:
            logger.error("未知的检查点格式")
            return model, optimizer, scheduler, 0, float('inf')
        
        logger.info(f"成功加载检查点 (轮次 {epoch}): {checkpoint_path}")
        return model, optimizer, scheduler, epoch, loss
        
    except Exception as e:
        logger.error(f"加载检查点失败: {str(e)}")
        return model, optimizer, scheduler, 0, float('inf')

def save_checkpoint(model, optimizer, scheduler, epoch_val, loss, element, checkpoint_type='checkpoint'):
    """保存训练检查点"""
    if isinstance(config, dict):
        model_dir = config.get('model_config', {}).get('model_dir', 'models')
    else:
        try:
            model_dir = config.model_config['model_dir']
        except:
            model_dir = 'models'
    
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, f'{checkpoint_type}_{element}.pth')
    
    # 只保存状态字典，不保存整个模型对象
    checkpoint = {
        'epoch': epoch_val,  # 使用传入的epoch_val参数，避免使用未定义的epoch
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss
    }
    
    try:
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"成功保存检查点: {checkpoint_path}")
        return True
    except Exception as e:
        logger.error(f"保存检查点失败: {str(e)}")
        return False