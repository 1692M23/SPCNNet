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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sys
from torch.cuda.amp import GradScaler, autocast
import torch.nn.utils as torch_utils # Added for gradient clipping
from tqdm import tqdm

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

# 配置logger (继承自全局设置)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO) # 不再需要单独设置级别

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
    残差块 (添加可配置通道和核大小)
    """
    def __init__(self, channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2 # 自动计算 padding 保持尺寸
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        # <<< Add momentum=0.5 to BatchNorm1d >>>
        self.bn1 = nn.BatchNorm1d(channels, momentum=0.5)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        # <<< Add momentum=0.5 to BatchNorm1d >>>
        self.bn2 = nn.BatchNorm1d(channels, momentum=0.5)
        
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

class SpectralResCNN_GCN(nn.Module):
    """光谱残差CNN-GRU模型 (简化版，支持架构参数化)"""
    def __init__(self, 
                 input_size=None, 
                 device=None, 
                 use_gru=True, 
                 # --- 架构参数 ---
                 initial_channels=64, initial_kernel_size=7, initial_dropout=0.2,
                 num_res_blocks=3, res_channels=64, res_kernel_size=3, res_dropout=0.0, # 可在 ResBlock 内加 Dropout
                 gru_hidden_size=64, gru_num_layers=1, gru_dropout=0.0,
                 fc_hidden_layers=[256, 64], fc_dropout=[0.5, 0.3],
                 use_adaptive_pooling=True
                 ):
        super(SpectralResCNN_GCN, self).__init__()
        
        self.input_size = input_size # 主要用于信息，模型本身会适应
        self.use_gru = use_gru
        self.use_gcn = False # GCN 已移除
        self.use_adaptive_pooling = use_adaptive_pooling
        logger.info(f"初始化简化版模型，使用GRU: {use_gru}")
        logger.info(f"架构参数: initial_ch={initial_channels}, initial_k={initial_kernel_size}, num_res={num_res_blocks}, res_ch={res_channels}, res_k={res_kernel_size}, gru_hidden={gru_hidden_size if use_gru else 'N/A'}, fc_layers={fc_hidden_layers}, dropout={fc_dropout}")

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"模型将使用设备: {self.device}")
        
        # --- 初始特征提取层 (参数化) ---
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, initial_channels, kernel_size=initial_kernel_size, padding=initial_kernel_size//2),
            # <<< Add momentum=0.5 to BatchNorm1d >>>
            nn.BatchNorm1d(initial_channels, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(initial_dropout),
            nn.MaxPool1d(2)
        )
        current_channels = initial_channels
        
        # --- 残差模块 (参数化) ---
        self.res_blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResidualBlock(current_channels, kernel_size=res_kernel_size))
            # 注意：当前ResidualBlock内没有dropout，如需添加要修改ResidualBlock类
            # 可以在块之间添加 Dropout
            if res_dropout > 0:
                 self.res_blocks.append(nn.Dropout(res_dropout))
                 
        # --- 循环模块 (参数化) ---
        gru_output_size = 0
        if self.use_gru:
            # GRU 输入通道是初始特征提取后的通道数
            self.gru = nn.GRU(current_channels, gru_hidden_size, 
                              num_layers=gru_num_layers, 
                              bidirectional=True, batch_first=True,
                              dropout=gru_dropout if gru_num_layers > 1 else 0)
            gru_output_size = gru_hidden_size * 2 # 双向
        
        # --- 信息融合层 (输入大小动态确定) ---
        fusion_input_size = current_channels + gru_output_size # CNN/Res + Optional GRU
        # 融合层输出通道数，固定为 64? 还是也参数化?
        fusion_output_channels = 64 # 暂时固定
        self.fusion = nn.Sequential(
            nn.Conv1d(fusion_input_size, fusion_output_channels, kernel_size=1),
            # <<< Add momentum=0.5 to BatchNorm1d >>>
            nn.BatchNorm1d(fusion_output_channels, momentum=0.5),
            nn.ReLU()
        )
        
        # --- 池化层 (可选的自适应池化) ---
        if self.use_adaptive_pooling:
            self.pool = nn.AdaptiveAvgPool1d(1)
            pool_output_features = fusion_output_channels
        else:
            # 如果不用自适应池化，需要一个 Flatten 层
            # 注意：此时 Flatten 后的维度依赖于序列长度，不推荐
            self.pool = nn.Identity() # 或者 None? 需要在 forward 处理
            # 此时需要知道池化前的序列长度来计算 FC 输入大小，很麻烦
            logger.warning("未使用自适应池化，全连接层输入维度可能不确定！推荐使用自适应池化。")
            pool_output_features = -1 # 标记为不确定
        
        # --- 全连接层 (参数化) ---
        self.fc = nn.Sequential()
        last_layer_features = pool_output_features 
        if last_layer_features == -1:
             # 尝试从一个样本推断？或者需要固定输入大小？暂时设为默认值
             logger.warning("无法确定 FC 输入大小，使用默认值 64*1。可能出错！")
             last_layer_features = 64 # 假设池化后是64
             
        # 解析 fc_dropout 列表
        if not isinstance(fc_dropout, list) or len(fc_dropout) < len(fc_hidden_layers) + 1:
             # 如果 fc_dropout 不是列表或长度不够，则使用单个值或默认值
             dropout_val = fc_dropout if isinstance(fc_dropout, (float, int)) else 0.5
             fc_dropout = [dropout_val] * (len(fc_hidden_layers) + 1)
             
        dropout_idx = 0
        for hidden_size in fc_hidden_layers:
            self.fc.add_module(f"fc_{dropout_idx}", nn.Linear(last_layer_features, hidden_size))
            self.fc.add_module(f"relu_{dropout_idx}", nn.ReLU())
            self.fc.add_module(f"dropout_{dropout_idx}", nn.Dropout(fc_dropout[dropout_idx]))
            last_layer_features = hidden_size
            dropout_idx += 1
            
        # 输出层
        self.fc.add_module("fc_out", nn.Linear(last_layer_features, 1))
        # 输出层后通常不加激活或Dropout
        
        self.to(self.device)
        
    def forward(self, x):
        # (Forward 逻辑基本不变，但注意处理 non-adaptive pooling 的情况)
        x = move_to_device(x, self.device)
        batch_size, channels, seq_len = x.size()

        cnn_features = self.feature_extractor(x)
        
        res_features = cnn_features
        for layer in self.res_blocks:
            res_features = layer(res_features)
        
        features_to_combine = [res_features]
        
        if self.use_gru:
            rec_input = cnn_features.permute(0, 2, 1)
            gru_features, _ = self.gru(rec_input)
            gru_features = gru_features.permute(0, 2, 1)
            features_to_combine.append(gru_features)
            
        if len(features_to_combine) > 1:
            combined_features = torch.cat(features_to_combine, dim=1)
            fused_features = self.fusion(combined_features)
        elif len(features_to_combine) == 1: 
             # Apply fusion even if only ResNet features are present?
             # Current fusion expects specific input size based on use_gru
             # Let's assume fusion is applied if input matches expected size
             expected_fusion_in = self.fusion[0].in_channels
             if res_features.shape[1] == expected_fusion_in:
                  fused_features = self.fusion(res_features)
             else:
                  # Maybe just pass through or add a specific layer?
                  logger.warning(f"Skipping fusion? Input channels {res_features.shape[1]} != Expected {expected_fusion_in}")
                  fused_features = res_features # Pass through for now
        else:
             logger.error("Error: No features to process.")
             return torch.zeros((batch_size, 1), device=self.device) # Return dummy output

        # --- 处理池化 --- 
        if self.use_adaptive_pooling:
            pooled_features = self.pool(fused_features)
            flattened_features = pooled_features.view(pooled_features.size(0), -1)
        else: 
            # 如果没有自适应池化，需要 Flatten。这需要知道 fused_features 的序列长度
            # 这是一个潜在的问题点，因为序列长度可能变化
            # 假设 fused_features 的形状是 [batch, channels, length]
            try:
                flattened_features = fused_features.view(fused_features.size(0), -1)
                # 更新 FC 输入大小 (如果第一次运行或大小改变)
                if not hasattr(self.fc[0], 'in_features') or self.fc[0].in_features != flattened_features.shape[1]:
                     logger.warning(f"Dynamically setting FC input size to {flattened_features.shape[1]}")
                     # This requires recreating the FC layer or adjusting weights, complex!
                     # For now, this branch is problematic if input length varies.
            except Exception as e:
                 logger.error(f"Error flattening features without adaptive pooling: {e}")
                 return torch.zeros((batch_size, 1), device=self.device)
        
        output = self.fc(flattened_features)
        return output

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
            model_predictions = np.stack(model_predictions, axis=0)  # [num_models, mc_samples, batch, 1]
            
            # 计算最终预测值（所有模型和样本的平均值）
            predictions = np.mean(model_predictions, axis=(0, 1))
            
            # 计算预测不确定性（所有预测的标准差）
            uncertainty = np.std(model_predictions, axis=(0, 1))
            
            all_predictions.append((predictions, uncertainty))
        
        return all_predictions

# =============== 2. 训练相关 ===============
def train(model, train_loader, val_loader, config, device=None, element=None, start_epoch=0, 
          initial_best_val_loss=float('inf'), initial_patience=0, 
          stage1_completed=False, training_completed=False,
          augment_fn=None): # Added augment_fn parameter
    """
    训练模型的核心函数，支持断点续训和两阶段训练。
    
    参数:
        model: 待训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config (module): 包含训练和模型配置的 *模块*
        device: 计算设备
        element (str): 元素名称，用于保存模型和状态
        start_epoch (int): 起始的epoch
        initial_best_val_loss (float): 初始的最佳验证损失
        initial_patience (int): 初始的早停计数器
        stage1_completed (bool): 第一阶段训练是否完成
        training_completed (bool): 整个训练是否完成
        augment_fn (callable, optional): Batch-level augmentation function.

    返回:
        tuple: (训练好的模型, history_dict)
    """
    logger = logging.getLogger('train')
    
    # 从配置模块中获取训练参数
    train_cfg = config.training_config # 修改：使用属性访问
    lr = train_cfg.get('lr', 0.001)
    weight_decay = train_cfg.get('weight_decay', 1e-4)
    num_epochs = train_cfg.get('num_epochs', 100)
    patience = train_cfg.get('early_stopping_patience', 10)
    scheduler_type = train_cfg.get('scheduler', 'cosine')
    scheduler_params = train_cfg.get('scheduler_params', {})
    lr_min = train_cfg.get('lr_min', 1e-6)
    gradient_clip_val = train_cfg.get('gradient_clip_val', 1.0)
    
    # 确定设备
    if device is None:
        device = setup_device()
    
    # 如果模型不在指定设备上，移动模型
    if next(model.parameters()).device != device:
        model = model.to(device)
        logger.info(f"模型已移动到设备: {device}")
    
    # <<< Initialize Optimizer based on Config >>>
    optimizer_type = train_cfg.get('optimizer_type', 'AdamW').lower()
    # Note: lr and weight_decay are read earlier from train_cfg
    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        logger.info(f"使用 AdamW 优化器 (lr={lr}, weight_decay={weight_decay})")
    elif optimizer_type == 'adam':
        # Adam handles weight decay differently internally, config wd should be ~0
        if weight_decay > 1e-5:
             logger.warning(f"Adam 优化器通常与 weight_decay=0 或极小值配合使用，当前设置为 {weight_decay}")
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        logger.info(f"使用 Adam 优化器 (lr={lr}, weight_decay={weight_decay})")
    # Add other optimizers like RMSprop if needed
    # elif optimizer_type == 'rmsprop':
    #     optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    #     logger.info(f"使用 RMSprop 优化器 (lr={lr}, weight_decay={weight_decay})")
    else:
        logger.warning(f"未知的优化器类型: {optimizer_type}，将默认使用 AdamW。")
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # <<< End Optimizer Initialization >>>
    
    # 初始化学习率调度器
    scheduler = None
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_params.get('T_0', 10), 
            T_mult=scheduler_params.get('T_mult', 1),
            eta_min=scheduler_params.get('eta_min', lr_min)
        )
    elif scheduler_type == 'reduce_lr_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_params.get('factor', 0.5),
            patience=scheduler_params.get('patience', 5),
            verbose=True,
            min_lr=lr_min
        )
    else:
        logger.warning(f"不支持的学习率调度器类型: {scheduler_type}，将不使用调度器。")
    
    # --- 修改损失函数初始化逻辑 ---
    loss_type = train_cfg.get('loss_function', 'MSE') # 默认使用 MSE
    loss_params = train_cfg.get('loss_params', {})   # 获取损失函数参数

    logger.info(f"使用损失函数: {loss_type}，参数: {loss_params}")

    if loss_type == 'WeightedMSE':
        # 从 loss_params 获取参数，如果不存在则使用默认值
        threshold = loss_params.get('threshold', 0.2)
        high_weight = loss_params.get('high_weight', 2.0)
        criterion = WeightedMSELoss(threshold=threshold, high_weight=high_weight)
        logger.info(f"  WeightedMSELoss - threshold: {threshold}, high_weight: {high_weight}")
    elif loss_type == 'Huber':
        # 从 loss_params 获取 delta 参数
        delta = loss_params.get('delta', 1.0) # HuberLoss 默认 delta=1.0
        criterion = nn.HuberLoss(delta=delta)
        logger.info(f"  HuberLoss - delta: {delta}")
    elif loss_type == 'MAE':
        criterion = nn.L1Loss()
    elif loss_type == 'MSE':
        criterion = nn.MSELoss()
    else:
        logger.warning(f"未知的损失函数类型: {loss_type}，将使用默认的 MSELoss。")
        criterion = nn.MSELoss()
    # --- 结束修改 ---
    
    # 混合精度训练 (如果设备支持)
    scaler = None
    if device.type == 'cuda':
        scaler = GradScaler()
        logger.info("启用CUDA混合精度训练")
    elif str(device).startswith('xla'):
        logger.info("TPU训练通常自动处理混合精度")
        
    # 状态变量
    best_val_loss = initial_best_val_loss
    patience_counter = initial_patience
    # <<< Initialize history dict >>>
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_r2': [], 'lr': []}
    
    # 加载检查点 (如果存在且需要)
    model_cfg = config.model_config # 修改：使用属性访问
    checkpoint_path = os.path.join(model_cfg['model_dir'], f'{element}_checkpoint.pth') # 修改：使用属性访问
    if train_cfg.get('resume_training', True) and os.path.exists(checkpoint_path):
        try:
            logger.info(f"从检查点加载状态: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            patience_counter = checkpoint.get('patience_counter', 0)
            # Load scaler state if exists
            if scaler and 'scaler_state_dict' in checkpoint:
                 scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info(f"从 epoch {start_epoch} 继续训练")
        except Exception as e:
            logger.warning(f"加载检查点失败: {e}，将从头开始训练。")
            start_epoch = 0
            best_val_loss = float('inf')
            patience_counter = 0
    else:
         logger.info("未找到检查点或未启用恢复，将从头开始训练。")
         start_epoch = 0
         best_val_loss = float('inf')
         patience_counter = 0

    # ------------------- 训练循环 -------------------
    logger.info(f"开始训练 {element}，共 {num_epochs} 个 epochs")
    
    # <<< Get Warm-up Parameters from Config >>>
    warmup_enabled = config.warmup_config.get('enabled', False) # Default to False if not found
    warmup_epochs = config.warmup_config.get('epochs', 5)
    warmup_start_factor = config.warmup_config.get('start_factor', 0.1)
    
    initial_lr = optimizer.param_groups[0]['lr'] # Get the initial LR from optimizer
    warmup_start_lr = initial_lr * warmup_start_factor
    
    if warmup_enabled:
        logger.info(f"启用学习率 Warm-up: {warmup_epochs} epochs, 从 {warmup_start_lr:.2e} 到 {initial_lr:.2e}")
    # <<< End Get Warm-up Parameters >>>
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        # <<< Adjust LR for Warm-up Phase (If Enabled) >>>
        current_lr = optimizer.param_groups[0]['lr'] # Get current LR before potential update
        if warmup_enabled and epoch < warmup_epochs:
            # Linear warm-up
            # Avoid division by zero if warmup_epochs is 0 or 1
            if warmup_epochs > 0:
                lr = warmup_start_lr + (initial_lr - warmup_start_lr) * (epoch / warmup_epochs)
            else:
                lr = initial_lr # Should not happen if warmup_enabled is true and epochs > 0
                
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            current_lr = lr # Update current_lr for logging
            # logger.info(f"Warm-up Epoch {epoch+1}/{warmup_epochs}, LR set to {lr:.6e}") # Log less verbosely
        elif warmup_enabled and epoch == warmup_epochs:
            # Ensure LR is set back to the initial value after warm-up before scheduler takes over
            logger.info(f"Warm-up 结束，恢复初始学习率: {initial_lr:.6e}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr
            current_lr = initial_lr
        # else: LR is controlled by the scheduler (or remains initial_lr if no scheduler step happens)
        # <<< End Adjust LR for Warm-up Phase >>>
        
        # --- 训练阶段 ---
        model.train()
        running_train_loss = 0.0
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for i, (inputs, targets) in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 应用数据增强 (如果提供了函数)
            if augment_fn:
                inputs = augment_fn(inputs)

            optimizer.zero_grad()
            
            # 混合精度
            if scaler:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                # Gradient Clipping before scaler.step()
                if gradient_clip_val > 0:
                    scaler.unscale_(optimizer) # Unscale gradients before clipping
                    torch_utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                scaler.step(optimizer)
                scaler.update()
            elif str(device).startswith('xla'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                # Gradient Clipping
                if gradient_clip_val > 0:
                    torch_utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                xm.optimizer_step(optimizer, barrier=True) # TPU optimizer step
            else: # CPU
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                 # Gradient Clipping
                if gradient_clip_val > 0:
                    torch_utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                optimizer.step()
            
            running_train_loss += loss.item()
            train_pbar.set_postfix(loss=f'{loss.item():.4f}')
        
        epoch_train_loss = running_train_loss / len(train_loader)
        history['train_loss'].append(epoch_train_loss)
        # <<< Use current_lr for history logging >>>
        history['lr'].append(current_lr)
        # <<< End Use current_lr >>>

        # --- 验证阶段 ---
        model.eval()
        running_val_loss = 0.0
        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for i, (inputs, targets) in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                if scaler:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                running_val_loss += loss.item()
                val_pbar.set_postfix(loss=f'{loss.item():.4f}')
        
        epoch_val_loss = running_val_loss / len(val_loader)
        history['val_loss'].append(epoch_val_loss)
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # --- Calculate and Log Validation MAE and R2 --- 
        epoch_val_mae = np.nan
        epoch_val_r2 = np.nan
        all_val_outputs = []
        all_val_targets = []
        model.eval() # Ensure model is in eval mode
        with torch.no_grad():
             for inputs, targets in val_loader:
                 inputs, targets = inputs.to(device), targets.to(device)
                 if scaler:
                     with autocast():
                         outputs = model(inputs)
                 else:
                     outputs = model(inputs)
                 all_val_outputs.append(outputs.cpu().float().numpy())
                 all_val_targets.append(targets.cpu().float().numpy())
        
        try:
            if all_val_outputs and all_val_targets:
                y_pred_val = np.vstack(all_val_outputs).astype(np.float32).flatten()
                y_true_val = np.vstack(all_val_targets).astype(np.float32).flatten()
                valid_mask_val = ~np.isnan(y_pred_val) & ~np.isnan(y_true_val)
                y_pred_val_valid = y_pred_val[valid_mask_val]
                y_true_val_valid = y_true_val[valid_mask_val]
                if len(y_pred_val_valid) > 0:
                    epoch_val_mae = mean_absolute_error(y_true_val_valid, y_pred_val_valid)
                    epoch_val_r2 = r2_score(y_true_val_valid, y_pred_val_valid)
        except Exception as metric_err:
             logger.warning(f"Epoch {epoch+1} - 计算验证 MAE/R2 时出错: {metric_err}")

        history['val_mae'].append(epoch_val_mae)
        history['val_r2'].append(epoch_val_r2)
        # --- End Calculation --- 

        # <<< Updated Logging with MAE/R2 and correct LR >>>
        logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | Val MAE: {epoch_val_mae:.4f} | Val R2: {epoch_val_r2:.4f} | LR: {current_lr:.6e} | Time: {epoch_duration:.2f}s")

        # --- 更新学习率和早停逻辑 (基于Val Loss) ---
        # <<< Scheduler step should happen AFTER warm-up phase (If Enabled) >>>
        if (not warmup_enabled or epoch >= warmup_epochs) and scheduler:
             if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                 scheduler.step(epoch_val_loss)
             else:
                 # Ensure other schedulers are also stepped only after warm-up
                 # CosineAnnealingWarmRestarts step depends on the epoch number relative to T_0 etc.
                 # This might need adjustment if using Cosine scheduler with warm-up
                 scheduler.step()
        # <<< End Scheduler Step Logic >>>
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            logger.info(f"  New best validation loss: {best_val_loss:.6f}. Saving model...")
            # 保存最佳模型
            # 修改：使用属性访问 config.model_config
            best_model_path = os.path.join(config.model_config['model_dir'], f'{element}_best_model.pth')
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
            }
            torch.save(save_dict, best_model_path)
        else:
            patience_counter += 1
            logger.info(f"  Validation loss did not improve. Patience: {patience_counter}/{patience}")

        # --- 保存检查点 ---
        checkpoint_save_path = os.path.join(config.model_config['model_dir'], f'{element}_checkpoint.pth')
        save_dict_ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
        }
        if scheduler:
            save_dict_ckpt['scheduler_state_dict'] = scheduler.state_dict()
        if scaler:
            save_dict_ckpt['scaler_state_dict'] = scaler.state_dict()
        torch.save(save_dict_ckpt, checkpoint_save_path)
        # logger.debug(f"Checkpoint saved to {checkpoint_save_path}")
        
        # --- 早停检查 ---
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs.")
            break
            
    # ------------------- 训练结束 -------------------
    logger.info("训练完成。")
    
    # --- 加载最佳模型（基于损失）---
    best_model_path = os.path.join(config.model_config['model_dir'], f'{element}_best_model.pth')
    best_model_loaded = False
    if os.path.exists(best_model_path):
        logger.info(f"加载最佳模型(基于损失): {best_model_path}")
        try:
            best_checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(best_checkpoint['model_state_dict'])
            best_model_loaded = True # 标记最佳模型已加载
            # 从检查点获取对应的最佳验证损失
            best_val_loss_from_ckpt = best_checkpoint.get('best_val_loss', best_val_loss) 
        except Exception as e:
            logger.warning(f"加载最佳模型失败: {e}。返回当前模型状态。")
            best_val_loss_from_ckpt = best_val_loss # 使用训练结束时的 best_val_loss
    else:
        logger.warning("未找到保存的最佳模型文件，返回当前模型状态。")
        best_val_loss_from_ckpt = best_val_loss # 使用训练结束时的 best_val_loss

    # <<< 在训练结束后计算一次验证 R² >>>
    final_val_loss = np.nan
    final_val_r2 = np.nan
    if best_model_loaded or not os.path.exists(best_model_path): # 如果加载了最佳模型或压根没保存过最佳模型（用最终模型）
        logger.info("使用最佳/最终模型状态在验证集上计算最终 R²...")
        try:
            # 确保 evaluate_model 使用正确的损失函数
            loss_type = config.training_config.get('loss_function', 'MSE')
            loss_params = config.training_config.get('loss_params', {})
            
            # <<< FIX: Add handling for Huber and WeightedMSE Loss >>>
            if loss_type == 'MSE': 
                final_criterion = nn.MSELoss()
                logger.info("最终评估使用: MSELoss")
            elif loss_type == 'MAE': 
                final_criterion = nn.L1Loss()
                logger.info("最终评估使用: MAELoss (L1Loss)")
            elif loss_type == 'Huber':
                delta = loss_params.get('delta', 1.0)
                final_criterion = nn.HuberLoss(delta=delta)
                logger.info(f"最终评估使用: HuberLoss (delta={delta})")
            elif loss_type == 'WeightedMSE': # Assuming WeightedMSELoss class is defined
                threshold = loss_params.get('threshold', 0.2)
                high_weight = loss_params.get('high_weight', 2.0)
                final_criterion = WeightedMSELoss(threshold=threshold, high_weight=high_weight)
                logger.info(f"最终评估使用: WeightedMSELoss (thresh={threshold}, weight={high_weight})")
            else: 
                logger.warning(f"未知的损失函数类型 '{loss_type}' 用于最终评估，将默认使用 MSELoss。")
                final_criterion = nn.MSELoss()
            # <<< END FIX >>>
            
            final_val_loss, final_val_metrics, _, _ = evaluate_model(model, val_loader, device, final_criterion)
            final_val_r2 = final_val_metrics.get('r2', np.nan)
            logger.info(f"最终验证损失: {final_val_loss:.6f}, 最终验证 R²: {final_val_r2:.4f}")
        except Exception as final_eval_err:
            logger.error(f"计算最终验证指标时出错: {final_eval_err}")
    else: # 加载失败时，不计算最终 R2
        logger.warning("因最佳模型加载失败，未计算最终验证 R²。")
    # <<< 结束最终 R² 计算 >>>

    # 修改返回值：返回加载了最佳权重的模型、最佳验证损失(来自检查点或训练过程)和最终验证 R²
    # <<< MODIFIED: Return history dict >>>
    return model, best_val_loss_from_ckpt, final_val_r2, history

# =============== 3. 评估相关 ===============
def evaluate_model(model, data_loader, device, loss_fn):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    logger.info("[Evaluate V3] Starting evaluation...") # Add version marker
    
    # Determine if AMP should be enabled based on device
    amp_enabled = (str(device).startswith('cuda'))
    logger.info(f"[Evaluate V3] AMP enabled: {amp_enabled}")

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device, non_blocking=True) 
            targets = targets.to(device, non_blocking=True)
            
            # Ensure input is float32 before potential autocast or NaN handling
            inputs = inputs.float()
            targets = targets.float()
            
            # Handle potential NaNs in input 
            inputs, _, _ = handle_nan_values(inputs, replacement_strategy='mean', name=f"评估输入_batch{i}")
            
            outputs = None # Initialize outputs
            loss = torch.tensor(0.0, device=device, dtype=torch.float32) # Initialize loss

            try:
                 # <<< CORE CHANGE: Wrap model call and loss calculation inside autocast >>>
                 with torch.cuda.amp.autocast(enabled=amp_enabled):
                     outputs = model(inputs) 
                     # Ensure outputs and targets are float32 for loss function
                     loss = loss_fn(outputs.float(), targets.float())
                 # <<< END AUTOCAST >>>
                     
                 # Handle potential NaNs/Infs in loss AFTER calculation
                 if torch.isnan(loss) or torch.isinf(loss):
                     logger.warning(f"[Evaluate Batch {i}] 检测到 NaN/Inf 损失！Loss: {loss.item()}")
                     loss = torch.tensor(0.0, device=device, dtype=torch.float32) 
                     # If loss is bad, outputs might also be bad, reset them
                     if outputs is None or torch.isnan(outputs).any() or torch.isinf(outputs).any():
                          logger.warning(f"[Evaluate Batch {i}] 模型输出包含 NaN/Inf 或计算失败。")
                          outputs = torch.zeros_like(targets, dtype=torch.float32)
                 
                 # Handle NaNs/Infs in outputs if they somehow survived or occurred outside loss calc
                 if outputs is None or torch.isnan(outputs).any() or torch.isinf(outputs).any():
                     logger.warning(f"[Evaluate Batch {i}] 最终输出包含 NaN/Inf，将被替换为零。")
                     outputs = torch.zeros_like(targets, dtype=torch.float32)

            except RuntimeError as e:
                 logger.error(f"[Evaluate Batch {i}] RuntimeError during autocast/model/loss: {e}")
                 # traceback.print_exc() # Uncomment for full traceback during debugging
                 loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                 outputs = torch.zeros_like(targets, dtype=torch.float32)
            except Exception as e:
                 logger.error(f"[Evaluate Batch {i}] Unexpected error: {e}")
                 # traceback.print_exc()
                 loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                 outputs = torch.zeros_like(targets, dtype=torch.float32)

            # Accumulate loss
            total_loss += loss.item() * inputs.size(0) 
            # Append results (ensure float32 for numpy)
            all_outputs.append(outputs.cpu().float().numpy())
            all_targets.append(targets.cpu().float().numpy())
            
    logger.info("[Evaluate V3] Aggregation complete.")
    avg_loss = total_loss / len(data_loader.dataset) if len(data_loader.dataset) > 0 else 0
    
    # Stack results carefully
    try:
        if not all_outputs or not all_targets:
             raise ValueError("No valid outputs or targets collected during evaluation.")
        y_pred = np.vstack(all_outputs).astype(np.float32).flatten()
        y_true = np.vstack(all_targets).astype(np.float32).flatten()
    except ValueError as ve:
        logger.error(f"[Evaluate V3] Error during vstack: {ve}")
        metrics = {k: np.nan for k in ['mse', 'rmse', 'mae', 'r2']}
        return avg_loss, metrics, np.array([]), np.array([])

    # Filter NaNs from final arrays before calculating metrics
    valid_mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    y_pred_valid = y_pred[valid_mask]
    y_true_valid = y_true[valid_mask]
    
    metrics = {}
    if len(y_pred_valid) > 0:
        try:
            metrics['mse'] = mean_squared_error(y_true_valid, y_pred_valid)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true_valid, y_pred_valid)
            metrics['r2'] = r2_score(y_true_valid, y_pred_valid)
            # <<< Add dex calculation (standard deviation of residuals) >>>
            residuals = y_true_valid - y_pred_valid
            metrics['dex'] = np.std(residuals)
            # <<< End dex calculation >>>
        except Exception as e:
            logger.error(f"计算评估指标时出错: {e}")
            metrics = {k: np.nan for k in ['mse', 'rmse', 'mae', 'r2']}
            metrics['dex'] = np.nan # Ensure dex is also NaN on error
    else:
         logger.warning("评估中没有有效的预测值/目标值，无法计算指标。")
         metrics = {k: np.nan for k in ['mse', 'rmse', 'mae', 'r2']}
         metrics['dex'] = np.nan # Ensure dex is NaN

    return avg_loss, metrics, y_pred, y_true # Return original y_pred/y_true for potential full analysis

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
def load_trained_model(model_path, device=None, use_gru=True, use_gcn=True):
    """
    加载训练好的模型，确保正确处理状态字典
    
    参数:
        model_path: 模型文件路径
        device: 设备（CPU或GPU）
        use_gru: 是否使用双向GRU网络
        use_gcn: 是否使用图卷积网络
    """
    try:
        if os.path.exists(model_path):
            # 创建模型实例
            input_size = 3921  # 默认输入大小
            model = SpectralResCNN_GCN(input_size, device=device, use_gru=use_gru, use_gcn=use_gcn)
            logger.info(f"创建模型实例，使用GRU={use_gru}, 使用GCN={use_gcn}")
            
            # 加载状态字典
            try:
                # 首先尝试使用map_location避免设备不匹配问题
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # 检查状态字典中是否包含GRU和GCN配置
                if isinstance(checkpoint, dict):
                    # 尝试从检查点获取GRU和GCN设置
                    if 'use_gru' in checkpoint:
                        use_gru = checkpoint['use_gru']
                        logger.info(f"从检查点获取GRU设置: {use_gru}")
                    if 'use_gcn' in checkpoint:
                        use_gcn = checkpoint['use_gcn']
                        logger.info(f"从检查点获取GCN设置: {use_gcn}")
                        
                    # 如果检查点中的设置与传入的不同，重新创建模型
                    if ('use_gru' in checkpoint and checkpoint['use_gru'] != use_gru) or \
                       ('use_gcn' in checkpoint and checkpoint['use_gcn'] != use_gcn):
                        logger.info(f"检查点中的GRU/GCN设置与请求不同，重新创建模型")
                        model = SpectralResCNN_GCN(input_size, device=device, 
                                                 use_gru=use_gru, use_gcn=use_gcn)
                
                # 加载模型状态
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # 尝试加载状态字典
                    try:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        logger.info(f"从检查点的model_state_dict加载模型状态")
                    except Exception as e:
                        logger.warning(f"加载状态字典时出现不匹配，尝试部分加载: {str(e)}")
                        # 尝试部分加载状态字典
                        state_dict = checkpoint['model_state_dict']
                        model_dict = model.state_dict()
                        
                        # 过滤掉不匹配的键
                        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
                        
                        # 更新模型字典并加载
                        model_dict.update(state_dict)
                        model.load_state_dict(model_dict)
                        logger.info(f"部分加载模型状态字典成功")
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    # 尝试加载状态字典
                    try:
                        model.load_state_dict(checkpoint['state_dict'])
                        logger.info(f"从检查点的state_dict加载模型状态")
                    except Exception as e:
                        logger.warning(f"加载状态字典时出现不匹配，尝试部分加载: {str(e)}")
                        # 尝试部分加载状态字典
                        state_dict = checkpoint['state_dict']
                        model_dict = model.state_dict()
                        
                        # 过滤掉不匹配的键
                        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
                        
                        # 更新模型字典并加载
                        model_dict.update(state_dict)
                        model.load_state_dict(model_dict)
                        logger.info(f"部分加载模型状态字典成功")
                else:
                    # 尝试直接加载，假设是状态字典
                    try:
                        model.load_state_dict(checkpoint)
                        logger.info(f"直接加载状态字典成功")
                    except Exception as e:
                        logger.error(f"直接加载状态字典失败: {str(e)}")
                        # 如果所有尝试都失败，创建一个新模型
                        logger.warning(f"无法加载模型，将使用未初始化的模型")
            except RecursionError as re:
                logger.error(f"加载模型时递归深度超出错误: {str(re)}")
                logger.warning("检测到可能是保存了整个模型而非状态字典，创建新模型")
                # 出现递归深度错误时，返回未初始化的模型
            except RuntimeError as e:
                if "maximum recursion depth exceeded" in str(e):
                    logger.error(f"加载模型时递归深度超出错误: {str(e)}")
                    logger.warning("检测到可能是保存了整个模型而非状态字典，创建新模型")
                else:
                    logger.error(f"加载模型运行时错误: {str(e)}")
                # 出现递归深度错误时，返回未初始化的模型
            
            # 移动到设备
            if device:
                model = model.to(device)
                
            model.eval()
            logger.info(f"模型已加载并设置为评估模式")
            return model
        else:
            logger.error(f"模型文件不存在: {model_path}")
            # 创建新模型作为备选
            model = SpectralResCNN_GCN(3921, device=device, use_gru=use_gru, use_gcn=use_gcn)
            if device:
                model = model.to(device)
            model.eval()
            logger.warning(f"使用未训练的新模型作为备选")
            return model
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        # 创建新模型作为备选
        model = SpectralResCNN_GCN(3921, device=device, use_gru=use_gru, use_gcn=use_gcn)
        if device:
            model = model.to(device)
        model.eval()
        logger.warning(f"加载失败，使用未训练的新模型作为备选")
        return model

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
        device = element_or_device
        # 尝试从全局config获取GRU和GCN设置
        use_gru = True
        use_gcn = True
        if hasattr(config, 'use_gru'):
            use_gru = config.use_gru
        if hasattr(config, 'use_gcn'):
            use_gcn = config.use_gcn
        return load_trained_model_core(input_size_or_path, device, use_gru, use_gcn)
    
    # 旧方式: load_trained_model(input_size, element, config)
    input_size = input_size_or_path
    element = element_or_device
    device = config.training_config['device']
    
    # 获取GRU和GCN设置
    use_gru = getattr(config, 'use_gru', True)
    use_gcn = getattr(config, 'use_gcn', True)
    logger.info(f"从配置获取GRU/GCN设置: 使用GRU={use_gru}, 使用GCN={use_gcn}")
    
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
            os.path.join(config.model_config['model_dir'], f"best_model_{element}.pth"),  # 另一种最佳模型格式
            os.path.join(config.model_config['model_dir'], f"checkpoint_{element}.pth"),  # 检查点格式
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
            backup_model = SpectralResCNN_GCN(input_size, device=device, use_gru=use_gru, use_gcn=use_gcn)
            backup_model.to(device)
            backup_model.eval()
            logger.warning(f"使用未训练的模型作为后备")
            return backup_model
        
        # 找到模型文件，加载它
        return load_trained_model_core(model_path, device, use_gru, use_gcn)
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # 创建一个空模型作为后备选择
        backup_model = SpectralResCNN_GCN(input_size, device=device, use_gru=use_gru, use_gcn=use_gcn)
        backup_model.to(device)
        backup_model.eval()
        logger.warning(f"Using untrained model as backup")
        return backup_model

# 修改原来的load_trained_model函数名为load_trained_model_core，保持旧函数名对旧函数的向后兼容性
load_trained_model_core = load_trained_model
load_trained_model = load_trained_model_compat 

def save_model(model, optimizer, scheduler, epoch_val, val_loss, model_dir, element):
    """保存模型检查点"""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'best_model_{element}.pth')
    
    # 获取GRU和GCN设置（如果模型是SpectralResCNN_GCN类型）
    use_gru = True
    use_gcn = True
    if isinstance(model, SpectralResCNN_GCN):
        use_gru = model.use_gru
        use_gcn = model.use_gcn
        logger.info(f"保存模型设置: 使用GRU={use_gru}, 使用GCN={use_gcn}")
    
    # 只保存状态字典
    checkpoint = {
        'epoch': epoch_val,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'use_gru': use_gru,  # 保存GRU设置
        'use_gcn': use_gcn,  # 保存GCN设置
    }
    
    try:
        torch.save(checkpoint, model_path)
        logger.info(f"成功保存最佳模型: {model_path}")
        return True
    except Exception as e:
        logger.error(f"保存最佳模型失败: {str(e)}")
        return False

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

def load_checkpoint(model, optimizer, scheduler, element, checkpoint_type='checkpoint'):
    """加载训练检查点，兼容新旧格式"""
    if isinstance(config, dict):
        model_dir = config.get('model_config', {}).get('model_dir', 'models')
    else:
        try:
            model_dir = config.model_config['model_dir']
        except:
            model_dir = 'models'
            
    checkpoint_path = os.path.join(model_dir, f'{checkpoint_type}_{element}.pth')
    
    if not os.path.exists(checkpoint_path):
        logger.warning(f"找不到检查点文件: {checkpoint_path}")
        return model, optimizer, scheduler, 0, float('inf')
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 处理新格式检查点（包含state_dict）
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
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
            # 尝试直接加载为状态字典
            try:
                model.load_state_dict(checkpoint)
                logger.info(f"尝试直接加载检查点为状态字典成功")
                epoch = 0
                loss = float('inf')
            except Exception as e:
                logger.error(f"尝试直接加载检查点为状态字典失败: {str(e)}")
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
        'epoch': epoch_val,  # 使用epoch_val而不是epoch
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

# 在model.py中添加自定义损失函数类
class WeightedMSELoss(nn.Module):
    def __init__(self, threshold=0.2, high_weight=2.0):
        super().__init__()
        self.threshold = threshold
        self.high_weight = high_weight
        
    def forward(self, pred, target):
        weights = torch.ones_like(target)
        weights[target > self.threshold] = self.high_weight
        return torch.mean(weights * (pred - target) ** 2)

# +++ 新增 MC Dropout 预测函数 +++
def predict_with_mc_dropout(model, data_loader, device, mc_samples=50):
    """
    Performs prediction using Monte Carlo Dropout for uncertainty estimation.

    Args:
        model (torch.nn.Module): The trained model.
        data_loader (DataLoader): DataLoader for the dataset to predict on.
        device (torch.device): The device to perform computations on.
        mc_samples (int): The number of Monte Carlo samples (forward passes).

    Returns:
        tuple: (mean_predictions, uncertainties, targets)
               - mean_predictions (np.ndarray): Mean predictions over MC samples.
               - uncertainties (np.ndarray): Standard deviation over MC samples (uncertainty).
               - targets (np.ndarray): True target values from the data loader.
    """
    logger.info(f"开始使用 MC Dropout 进行预测，采样次数: {mc_samples}")
    
    # <<< 确保 device 是 torch.device 对象或使用字符串比较 >>>
    # 方法1：转换为 torch.device (如果尚未转换)
    # if isinstance(device, str):
    #     device = torch.device(device)
    # 方法2：直接使用字符串比较 (更简单)
    amp_enabled = (str(device) == 'cuda') # <-- 修改这里
    logger.info(f"MC Dropout - AMP enabled: {amp_enabled}")
    
    model.to(device)
    
    # 1. Enable Dropout layers
    model.train() 
    
    all_mc_predictions = []
    all_targets = []
    first_batch = True

    # 2. Perform multiple forward passes
    with torch.no_grad(): # No need to track gradients during prediction
        # Outer loop for MC samples
        for i in tqdm(range(mc_samples), desc="MC Dropout Sampling", leave=False):
            batch_predictions = []
            batch_targets = [] # Collect targets only once
            
            # Inner loop through data loader
            for inputs, targets_batch in data_loader:
                inputs, targets_batch = inputs.to(device), targets_batch.to(device)
                inputs, _, _ = handle_nan_values(inputs, replacement_strategy='mean', name="MC 输入")
                
                # <<< 使用前面计算的 amp_enabled >>>
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = model(inputs)
                # <<< 结束修改 >>>
                outputs, _, _ = handle_nan_values(outputs, replacement_strategy='zero', name="MC 输出")

                batch_predictions.append(outputs.cpu().numpy())
                if i == 0: # Collect targets only in the first MC sample iteration
                     batch_targets.append(targets_batch.cpu().numpy())
            
            # Concatenate predictions for this MC sample
            try: 
                sample_preds = np.vstack(batch_predictions).flatten()
                all_mc_predictions.append(sample_preds)
            except ValueError as e:
                 logger.error(f"Error stacking results in MC sample {i} (check output dimensions?): {e}")
                 # Optionally skip this sample or return error
                 continue 
                 
            # Concatenate targets after the first MC sample iteration
            if i == 0 and batch_targets:
                 try:
                     all_targets = np.vstack(batch_targets).flatten()
                 except ValueError as e:
                     logger.error(f"Error stacking targets (check target dimensions?): {e}")
                     # Cannot proceed without targets
                     model.eval() # Restore model state before returning
                     return np.array([]), np.array([]), np.array([])

    if not all_mc_predictions:
        logger.error("MC Dropout failed to produce any predictions.")
        model.eval()
        return np.array([]), np.array([]), np.array([])
        
    # 3. Calculate statistics across MC samples
    try:
        all_mc_predictions_np = np.stack(all_mc_predictions, axis=0) # Shape: [mc_samples, num_samples]
        mean_predictions = np.mean(all_mc_predictions_np, axis=0)
        uncertainties = np.std(all_mc_predictions_np, axis=0)
    except Exception as e:
         logger.error(f"Error calculating mean/std across MC samples: {e}")
         model.eval()
         return np.array([]), np.array([]), np.array([])

    # 4. Restore model to evaluation mode
    model.eval() 

    logger.info(f"MC Dropout 预测完成。平均预测值 shape: {mean_predictions.shape}, 不确定性 shape: {uncertainties.shape}")
    return mean_predictions, uncertainties, all_targets
# +++ 结束新增函数 +++