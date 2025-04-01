#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主模块：训练、验证和测试恒星光谱元素丰度预测模型
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import pandas as pd
import glob
import traceback
from torchvision import transforms

# 导入自定义模块
import config
from model import SpectralResCNN, SpectralResCNN_GCN, SpectralResCNNEnsemble, train, evaluate_model, load_trained_model, save_model, load_checkpoint, handle_nan_values
from evaluation import evaluate_all_elements, plot_predictions_vs_true, plot_metrics_comparison
from utils import CacheManager, ProgressManager, ask_clear_cache, setup_analysis_directories, set_seed
from multi_element_processor import MultiElementProcessor
from fits_cache import FITSCache
from hyperparameter_tuning_replacement import hyperparameter_tuning as run_grid_search_tuning
from model_analysis import analyze_model_performance, show_batch_results, analyze_feature_importance, analyze_residuals

# 添加数据增强函数
def add_noise(data, noise_level=0.01):
    """简单的向数据张量添加高斯噪声"""
    # 确保 data 和 noise 在同一设备上
    device = data.device
    noise = torch.randn_like(data, device=device) * noise_level
    noisy_data = data + noise
    return noisy_data

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.output_config['log_dir'], 'main.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('main')

# 创建缓存管理器
cache_manager = CacheManager(cache_dir=os.path.join(config.output_config['cache_dir'], 'main'))

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

def load_data(data_path, element=None):
    """加载数据集并返回numpy数组"""
    try:
        data = np.load(data_path)
        # 检查并适应不同的键名约定
        if 'spectra' in data:
            X = data['spectra']
            y = data['abundance'] if 'abundance' in data else data['y']
        else:
            X = data['X']
            y = data['y']
        
        # 检查元素标签是否存在
        elements = data['elements'] if 'elements' in data else None
        
        # 如果指定了元素，检查元素信息
        if element is not None:
            # 尝试从命令行参数获取元素名
            def get_element_from_cmd():
                import sys
                for i, arg in enumerate(sys.argv):
                    if arg == '--element' and i < len(sys.argv) - 1:
                        return sys.argv[i+1]
                    elif arg == '--elements' and i < len(sys.argv) - 1:
                        elements_str = sys.argv[i+1]
                        elements_list = elements_str.split()
                        if elements_list:
                            return elements_list[0]
                return None
            
            if elements is not None:
                if isinstance(elements, dict) and element in elements:
                    logger.info(f"找到元素 {element} 的索引")
                elif isinstance(elements, dict):
                    logger.warning(f"元素 {element} 不在索引字典中")
                    # 尝试从命令行获取元素
                    cmd_element = get_element_from_cmd()
                    if cmd_element:
                        logger.info(f"使用命令行指定的元素: {cmd_element}")
                        element = cmd_element
                    else:
                        logger.warning(f"无法从命令行参数中找到元素，使用提供的元素: {element}")
                    elements = {element: 0}  # 创建模拟字典
                else:
                    logger.warning(f"elements不是字典类型，无法查找特定元素索引")
                    # 尝试从命令行获取元素
                    cmd_element = get_element_from_cmd()
                    if cmd_element:
                        logger.info(f"使用命令行指定的元素: {cmd_element}")
                        element = cmd_element
                    else:
                        logger.warning(f"无法从命令行参数中找到元素，使用提供的元素: {element}")
                    elements = {element: 0}  # 创建模拟字典
            else:
                # 如果没有elements信息，创建一个简单的映射
                logger.warning(f"数据中没有元素索引信息，假设abundance对应 {element}")
                elements = {element: 0}  # 创建模拟字典，用于单元素数据
                
        logger.info(f"成功加载数据: {X.shape}, {y.shape}")
        # 添加详细日志，说明元素处理情况
        if element is not None:
            logger.info(f"元素处理情况: 当前使用元素 = {element}, elements字典包含键 = {list(elements.keys()) if isinstance(elements, dict) else '不是字典'}")
        return X, y, elements
    except Exception as e:
        logger.error(f"加载数据时出错: {e}")
        raise

def create_data_loaders(spectra, labels, batch_size=32, shuffle=True, augment=False):
    """创建数据加载器，增加数据增强选项"""
    # 确保数据是 torch.Tensor
    if not isinstance(spectra, torch.Tensor):
        spectra = torch.FloatTensor(spectra)
    if not isinstance(labels, torch.Tensor):
        labels = torch.FloatTensor(labels)

    # 对于一维y，添加一个维度
    if len(labels.shape) == 1:
        labels = labels.unsqueeze(1)

    # 对于二维X，添加通道维度
    if len(spectra.shape) == 2:
        spectra = spectra.unsqueeze(1) # Shape: [batch, 1, length]

    dataset = TensorDataset(spectra, labels)

    if augment:
        # 定义增强操作 (这里只用了加噪声，可以扩展)
        # 注意：直接在TensorDataset上做复杂变换比较麻烦
        # 更好的方式是定义一个完整的Dataset类，在__getitem__中应用变换
        # 这里为了简单起见，我们将在训练循环中直接对batch进行增强
        # 因此，这里只返回原始数据集，并在训练循环中处理增强
        logger.info("Data augmentation enabled (will be applied per batch during training).")
        pass # Augmentation will be handled in the training loop if enabled

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def setup_training_directories():
    """
    创建训练所需的目录
    """
    # 创建必要的目录
    for directory in config.REQUIRED_DIRS:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"已创建目录: {directory}")
    
    # 创建分析目录
    if hasattr(config, 'analysis_config') and config.analysis_config.get('perform_analysis', False):
        setup_analysis_directories()
        logger.info("已创建模型分析目录")

def calculate_dataset_stats(data_loader):
    """
    计算数据集的基本统计信息（均值）
    
    参数:
        data_loader (DataLoader): 数据加载器
        
    返回:
        float: 数据平均值
    """
    total_sum = 0.0
    total_count = 0
    
    with torch.no_grad():
        for batch in data_loader:
            # 获取特征数据
            features = batch[0]
            batch_size = features.size(0)
            
            # 计算该批次的总和
            batch_sum = torch.sum(features).item()
            
            # 累加
            total_sum += batch_sum
            total_count += batch_size * features.size(-1)  # 考虑特征的维度
    
    # 计算平均值
    if total_count > 0:
        return total_sum / total_count
    else:
        return 0.0

def train_and_evaluate_model(model, train_loader, val_loader, test_loader, element, device, config, augment_fn=None):
    """
    训练和评估模型
    """
    try:
        import json
        device = config.training_config['device']
        
        # 创建训练状态目录
        os.makedirs('models/training_states', exist_ok=True)
        state_file = f'models/training_states/training_state_{element}.json'
        
        # 如果存在训练状态文件，加载它
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
                start_epoch = state.get('epoch', 0)
                best_val_loss = state.get('best_val_loss', float('inf'))
                logger.info(f"从训练状态文件加载：epoch={start_epoch}, best_val_loss={best_val_loss}")
        else:
            start_epoch = 0
            best_val_loss = float('inf')
            logger.info("未找到训练状态文件，从头开始训练")
        
        # 训练模型
        best_model, val_loss = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            element=element,
            device=device,
            config=config,
            augment_fn=augment_fn,
            start_epoch=start_epoch
        )

        # <<< Define loss_fn before calling evaluate_model >>>
        # Assuming MSELoss is the appropriate loss function here, same as used in train?
        # Need to get loss_type from config if it varies
        loss_type = config.training_config.get('loss_function', 'MSE')
        if loss_type == 'MSE':
             loss_fn = torch.nn.MSELoss()
        elif loss_type == 'MAE': # Example for MAE
             loss_fn = torch.nn.L1Loss()
        else:
             logger.warning(f"未知的损失函数类型 '{loss_type}'，将默认使用 MSELoss。")
             loss_fn = torch.nn.MSELoss()
        logger.info(f"在最终评估中使用损失函数: {loss_type}")
        # <<< End define loss_fn >>>

        # 训练完成后，使用 test_loader 评估最佳模型
        if best_model is not None:
            logger.info(f"使用测试集评估元素 {element} 的最佳模型")
            # <<< Pass loss_fn to evaluate_model >>>
            _, test_metrics, _, _ = evaluate_model(best_model, test_loader, device, loss_fn)
            logger.info(f"元素 {element} 的测试指标: {test_metrics}")
        else:
            logger.warning(f"元素 {element} 未能训练出有效模型，无法进行测试评估")
            test_metrics = {'mse': float('nan'), 'mae': float('nan'), 'r2': float('nan')} # 或者其他表示失败的值

        return best_model, val_loss, test_metrics
        
    except Exception as e:
        logger.error(f"训练元素 {element} 时出错: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def hyperparameter_tuning(element, train_loader, val_loader, config):
    """
    执行超参数调优 (修改为传递 DataLoader)

    Args:
        element (str): 元素名称
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config (module): 配置模块

    Returns:
        dict: 找到的最佳超参数
    """
    logger.info(f"开始为元素 {element} 进行超参数调优")

    tuning_cfg = config.tuning_config
    device = config.training_config['device']

    try:
        # 不再需要从DataLoader提取完整数据集
        # logger.info(f"Hyperparameter tuning data prepared:")
        # logger.info(f"  Train loader batches: {len(train_loader)}")
        # logger.info(f"  Val loader batches: {len(val_loader)}")

        # 从 config 获取 param_grid
        param_grid = tuning_cfg.get('search_space', {}).get('stage1')
        if not param_grid:
            logger.warning("配置中未找到有效的 param_grid for stage1，使用默认网格。")
            param_grid = None
        else:
             logger.info(f"使用配置中的参数网格进行调优: {param_grid}")

        # 调用 run_grid_search_tuning 函数，传递加载器
        logger.info("调用 run_grid_search_tuning 函数 (传递 DataLoader)...")
        best_params = run_grid_search_tuning(
            element=element,
            train_loader=train_loader, # 传递加载器
            val_loader=val_loader,   # 传递加载器
            param_grid=param_grid,
            device=device,
            config_module=config # 传递整个config模块给调优函数，供其内部使用
        )

        if best_params:
            logger.info(f"超参数调优完成。找到的最佳参数: {best_params}")
            # 确保返回的参数不包含非基本类型（如device对象）
            if isinstance(best_params, dict):
                 cleaned_params = {k: v for k, v in best_params.items() if isinstance(v, (str, int, float, bool, list, dict))}
                 return cleaned_params
            else:
                 return best_params # 或者返回None，如果格式不确定
        else:
            logger.warning("超参数调优未能找到最佳参数，将使用默认值。")
            return None

    except Exception as e:
        logger.error(f"超参数调优过程中发生错误: {e}", exc_info=True)
        return None

def visualize_training(element, train_metrics, val_metrics, output_dir=None):
    """
    可视化训练过程
    
    参数:
        element (str): 元素名称
        train_metrics (dict): 训练指标
        val_metrics (dict): 验证指标
        output_dir (str): 输出目录
    """
    if output_dir is None:
        output_dir = config.output_config['plots_dir']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics['loss'], label='Training Loss')
    plt.plot(val_metrics['loss'], label='Validation Loss')
    plt.title(f'Training Process for {element}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, f'{element}_training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制MAE曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics['mae'], label='Training MAE')
    plt.plot(val_metrics['mae'], label='Validation MAE')
    plt.title(f'Training Process for {element}')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, f'{element}_training_mae.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"已保存训练过程可视化图表")

class TrainingStateManager:
    """
    训练状态管理器，用于管理训练过程的中断和恢复
    """
    def __init__(self, element, model_dir='models'):
        self.element = element
        self.states_dir = os.path.join(model_dir, 'training_states')
        os.makedirs(self.states_dir, exist_ok=True)
        self.state_file = os.path.join(self.states_dir, f'training_state_{element}.json')
        self.checkpoint_file = os.path.join(model_dir, f'checkpoint_{element}.pth')
        
    def save_state(self, stage, epoch, best_val_loss, patience_counter, 
                  stage1_completed=False, training_completed=False):
        """保存训练状态"""
        import json
        import time
        
        state = {
            'element': self.element,
            'current_stage': stage,
            'current_epoch': epoch,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'stage1_completed': stage1_completed,
            'training_completed': training_completed,
            'timestamp': time.time()
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=4)
            
        return True
    
    def load_state(self):
        """加载训练状态"""
        if not os.path.exists(self.state_file):
            logger.info(f"找不到训练状态文件: {self.state_file}")
            return None
        
        try:
            import json
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            logger.info(f"加载训练状态: 阶段={state.get('current_stage')}, 轮次={state.get('current_epoch')}")
            return state
        except Exception as e:
            logger.error(f"加载训练状态失败: {str(e)}")
            return None
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, loss):
        """保存检查点"""
        import torch
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss
        }
        
        torch.save(checkpoint, self.checkpoint_file)
        return True
    
    def load_checkpoint(self, model, optimizer=None, scheduler=None, device=None):
        """加载检查点"""
        if not os.path.exists(self.checkpoint_file):
            logger.info(f"找不到检查点文件: {self.checkpoint_file}")
            return None
        
        try:
            import torch
            
            # 根据设备加载检查点
            if device:
                checkpoint = torch.load(self.checkpoint_file, map_location=device)
            else:
                checkpoint = torch.load(self.checkpoint_file)
            
            # 加载模型状态
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 如果提供了优化器，加载优化器状态
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 如果提供了调度器，加载调度器状态
            if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            logger.info(f"加载检查点: 轮次={checkpoint.get('epoch')}, 损失={checkpoint.get('loss'):.6f}")
            return checkpoint
        except Exception as e:
            logger.error(f"加载检查点失败: {str(e)}")
            return None
            
    def clear_state(self):
        """清除训练状态"""
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        return True

def process_element(element, model_type, input_size, use_gpu=True, config=None):
    # 在函数开始处添加
    print(f"正在处理元素: {element}", flush=True)
    logger.info(f"正在处理元素: {element}")
    
    # 获取命令行参数中的GRU和GCN设置
    use_gru = getattr(config, 'use_gru', True)  # 默认使用GRU
    use_gcn = getattr(config, 'use_gcn', True)  # 默认使用GCN
    
    logger.info(f"模型配置: 使用GRU={use_gru}, 使用GCN={use_gcn}")
    
    # 创建训练状态管理器
    state_manager = TrainingStateManager(
        element=element, 
        model_dir=config.model_config.get('model_dir', 'models')
    )
    
    # 配置设备
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info("GPU不可用，使用CPU")
    else:
        # 如果明确不使用GPU，尝试使用TPU
        try:
            if HAS_XLA:
                device = xm.xla_device()
                logger.info(f"使用TPU设备: {device}")
            else:
                logger.info("TPU不可用（未安装PyTorch XLA），使用CPU")
                device = torch.device('cpu')
        except Exception as e:
            logger.warning(f"TPU初始化失败: {str(e)}，使用CPU")
            device = torch.device('cpu')
    
    # 设置配置中的设备
    config.training_config['device'] = device
    
    # 将GRU和GCN使用设置添加到配置中
    config.use_gru = use_gru
    config.use_gcn = use_gcn
    
    # 加载数据
    logger.info(f"加载元素 {element} 的数据")
    train_path = os.path.join('processed_data', 'train_dataset.npz')
    val_path = os.path.join('processed_data', 'val_dataset.npz')
    test_path = os.path.join('processed_data', 'test_dataset.npz')
    
    X_train, y_train, _ = load_data(train_path, element)
    X_val, y_val, _ = load_data(val_path, element)
    X_test, y_test, _ = load_data(test_path, element)
    
    # 获取实际输入大小
    actual_input_size = X_train.shape[1] if len(X_train.shape) == 2 else X_train.shape[2]
    if input_size is None:
        input_size = actual_input_size
        logger.info(f"使用实际输入大小: {input_size}")
    
    # 创建模型
    if model_type is None:
        # 使用配置中的模型类型
        model_type = config.model_config.get('model_type', 'SpectralResCNN_GCN')
    
    logger.info(f"为元素 {element} 创建 {model_type} 模型，输入大小: {input_size}")
    
    if model_type == 'SpectralResCNN_GCN':
        model = SpectralResCNN_GCN(input_size=input_size, device=device, use_gru=use_gru, use_gcn=use_gcn)
    elif model_type == 'SpectralResCNN':
        model = SpectralResCNN(input_size=input_size)
    elif model_type == 'SpectralResCNNEnsemble':
        model = SpectralResCNNEnsemble(input_size=input_size)
    else:
        logger.error(f"未知的模型类型: {model_type}")
        raise ValueError(f"未知的模型类型: {model_type}")
    
    # 创建数据加载器
    batch_size = config.training_config.get('batch_size', 32)
    logger.info(f"创建数据加载器，批量大小: {batch_size}")
    
    train_loader = create_data_loaders(X_train, y_train, batch_size=batch_size)
    val_loader = create_data_loaders(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = create_data_loaders(X_test, y_test, batch_size=batch_size, shuffle=False)
    
    # 确定数据增强函数
    augment_fn = None
    if config.data_config.get('augmentation_enabled', False):
        noise_level = config.data_config.get('augmentation_params', {}).get('noise_level', 0.01)
        # 修改 lambda 表达式以匹配新的 add_noise 签名
        augment_fn = lambda data: add_noise(data, noise_level)
        logger.info(f"数据增强已启用，噪声水平: {noise_level}")

    # 检查是否存在训练状态
    state = state_manager.load_state()
    if state and not state.get('training_completed', False):
        logger.info(f"发现未完成的训练状态，准备从中断点恢复")
        
        # 创建优化器和调度器以便加载检查点
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=config.training_config.get('lr', 0.001),
                                    weight_decay=config.training_config.get('weight_decay', 1e-4))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        # 加载检查点
        checkpoint = state_manager.load_checkpoint(model, optimizer, scheduler, device)
        if checkpoint:
            logger.info("成功恢复模型状态，将从中断点继续训练")
        else:
            logger.warning("无法恢复模型状态，将从头开始训练")
    
    # 训练和评估模型
    logger.info(f"开始训练 {element} 模型")
    
    # 如果为TPU，添加特殊数据加载器处理
    if str(device).startswith('xla'):
        try:
            if HAS_PARALLEL_LOADER:
                logger.info("使用TPU并行数据加载器")
                # 用TPU特定的数据加载器包装原始加载器
                train_loader = pl.MpDeviceLoader(train_loader, device)
                val_loader = pl.MpDeviceLoader(val_loader, device)
                test_loader = pl.MpDeviceLoader(test_loader, device)
            else:
                logger.warning("无法使用TPU并行数据加载器，使用标准加载器")
        except ImportError:
            logger.warning("无法使用TPU并行数据加载器，使用标准加载器")
    
    # 进行训练和评估
    best_model, val_loss, test_metrics = train_and_evaluate_model(
        model, train_loader, val_loader, test_loader, element, device, config, augment_fn=augment_fn
    )
    
    # 训练完成，清除中断恢复状态
    state_manager.save_state(2, config.training_config.get('num_epochs', 100), val_loss, 0, 
                            stage1_completed=True, training_completed=True)
    
    logger.info(f"元素 {element} 的处理完成")
    logger.info(f"验证损失: {val_loss:.6f}")
    logger.info(f"测试指标: {test_metrics}")
    logger.info(f"R² Score (Test): {test_metrics.get('r2', 'N/A')}") # Explicitly log R²
    
    # --- 开始添加可视化代码 ---
    try:
        # 需要重新获取测试集的预测值和真实值
        logger.info(f"为 {element} 生成可视化图表")
        # <<< Ensure model is on the correct device before inference >>>
        model = model.to(device) 
        model.eval() # 确保模型在评估模式
        all_outputs = []
        all_targets = []
        
        # <<< Define element_plot_dir HERE >>>
        plots_dir = config.output_config.get('plots_dir', 'plots') # Get top-level plots directory
        element_plot_dir = os.path.join(plots_dir, 'evaluation', element) # Define element-specific plot directory
        os.makedirs(element_plot_dir, exist_ok=True) # Ensure the directory exists
        # <<< End define element_plot_dir >>>
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                 # 处理输入数据中的NaN值 (如果需要)
                inputs, _, _ = handle_nan_values(inputs, replacement_strategy='mean', name="绘图输入数据")
                
                # <<< Re-apply AUTOCAST HERE for visualization inference >>>
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                     outputs = model(inputs)
                # <<< END AUTOCAST >>>

                # 处理输出中的NaN值 (如果需要)
                outputs, _, _ = handle_nan_values(outputs, replacement_strategy='zero', name="绘图模型输出")

                # 移动到 CPU 并转换为 numpy
                if str(device).startswith('xla'):
                     # Ensure tensor operations are complete before moving for TPU
                     xm.mark_step()
                     # Use synchronous send for simplicity during visualization/debugging
                     cpu_outputs = outputs.cpu().numpy()
                     cpu_targets = targets.cpu().numpy()
                else:
                    cpu_outputs = outputs.cpu().numpy()
                    cpu_targets = targets.cpu().numpy()
                
                all_outputs.append(cpu_outputs)
                all_targets.append(cpu_targets)

        if all_outputs and all_targets:
            y_pred = np.vstack(all_outputs).flatten()
            y_true = np.vstack(all_targets).flatten()

            # 清理 NaN 值以防万一
            valid_mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
            y_pred = y_pred[valid_mask]
            y_true = y_true[valid_mask]

            if len(y_pred) > 0:
                # 绘制散点图: 预测值 vs 真实值
                plt.figure(figsize=(10, 6))
                # Use valid data for plotting
                plt.scatter(y_true, y_pred, alpha=0.5) 
                # 添加 y=x 对角线作为参考
                # Calculate limits based on valid data
                min_val = min(np.min(y_true), np.min(y_pred)) if len(y_true)>0 else 0
                max_val = max(np.max(y_true), np.max(y_pred)) if len(y_true)>0 else 1
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
                plt.title(f'Prediction vs True for {element}')
                plt.xlabel('True Values')
                plt.ylabel('Predicted Values')
                plt.grid(True)
                plt.legend()
                # Use defined element_plot_dir
                scatter_path = os.path.join(element_plot_dir, f'{element}_scatter_pred_true.png') 
                plt.savefig(scatter_path)
                plt.close()
                logger.info(f"散点图已保存至: {scatter_path}")

                # 绘制残差图: 残差 vs 真实值
                residuals = y_true - y_pred
                plt.figure(figsize=(10, 6))
                plt.scatter(y_true, residuals, alpha=0.5)
                plt.axhline(0, color='red', linestyle='--')
                plt.title(f'Residuals vs True for {element}')
                plt.xlabel('True Values')
                plt.ylabel('Residuals (True - Predicted)')
                plt.grid(True)
                # Use defined element_plot_dir
                residual_path = os.path.join(element_plot_dir, f'{element}_residuals.png') 
                plt.savefig(residual_path)
                plt.close()
                logger.info(f"残差图已保存至: {residual_path}")
                
                # 绘制残差分布直方图
                plt.figure(figsize=(10, 6))
                plt.hist(residuals, bins=30, alpha=0.7, density=True)
                # 添加均值和标准差信息
                mean_residual = np.mean(residuals)
                std_residual = np.std(residuals)
                plt.axvline(mean_residual, color='red', linestyle='--', label=f'Mean: {mean_residual:.4f}')
                plt.axvline(0, color='green', linestyle='-', label='Zero Error') # Add zero line
                plt.title(f'Residual Distribution for {element} (Mean: {mean_residual:.4f}, Std: {std_residual:.4f})')
                plt.xlabel('Residual Value')
                plt.ylabel('Frequency')
                plt.grid(True)
                plt.legend()
                # Use defined element_plot_dir
                hist_path = os.path.join(element_plot_dir, f'{element}_residuals_hist.png') 
                plt.savefig(hist_path)
                plt.close()
                logger.info(f"残差直方图已保存至: {hist_path}")

            else:
                 logger.warning(f"元素 {element} 没有有效的预测数据，无法生成可视化图表。")

    except NameError as ne:
        logger.error(f"生成可视化图表时变量未定义: {ne}") # Specific logging for NameError
        logger.error(traceback.format_exc())
    except Exception as plot_err:
        logger.error(f"为 {element} 生成可视化图表时出错: {plot_err}")
        logger.error(traceback.format_exc())
    # --- 结束可视化代码 ---

    return best_model, val_loss, test_metrics

def process_multiple_elements(csv_file, fits_dir, element_columns=None, 
                             test_size=0.2, val_size=0.1, batch_size=32, 
                             tune_hyperparams=False, device=None, batch_size_hyperopt=1000, batches_per_round=2):
    """
    处理多元素数据
    
    参数:
        csv_file (str): CSV文件路径，包含元素丰度数据
        fits_dir (str): FITS文件目录
        element_columns (list): 要处理的元素列名列表，如果为None则自动检测
        test_size (float): 测试集比例
        val_size (float): 验证集比例
        batch_size (int): 训练批次大小
        tune_hyperparams (bool): 是否调优超参数
        device (str): 使用的设备（'cpu'或'cuda'）
        batch_size_hyperopt (int): 超参数调优的批次大小
        batches_per_round (int): 每轮超参数调优使用的批次数
        
    返回:
        dict: 各元素的评估结果
    """
    logger.info(f"开始处理多元素数据: {csv_file}, 元素列表: {element_columns}")
    
    # 创建输出目录
    output_dir = os.path.join(config.data_config['processed_data_dir'], 'multi_element')
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建多元素处理器
    processor = MultiElementProcessor(
        fits_dir=fits_dir,
        cache_dir=os.path.join(config.output_config['cache_dir'], 'multi_element'),
        output_dir=output_dir
    )
    
    # 准备数据集 (返回dict: element -> {train, val, test})
    datasets = processor.prepare_datasets(
        csv_file=csv_file,
        element_columns=element_columns,
        test_size=test_size,
        val_size=val_size
    )
    
    if not datasets:
        logger.warning("未找到有效的元素数据")
        return {}
    
    # 设置设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 为每个元素训练模型
    results = {}
    for element, data in datasets.items():
        logger.info(f"开始处理元素 {element}")
        
        # 提取数据
        X_train, y_train = data['train']
        X_val, y_val = data['val']
        X_test, y_test = data['test']
        
        logger.info(f"元素 {element} 数据集大小: 训练集:{X_train.shape}, 验证集:{X_val.shape}, 测试集:{X_test.shape}")
            
        # 创建数据加载器
        train_loader = create_data_loaders(X_train, y_train, batch_size=batch_size)
        val_loader = create_data_loaders(X_val, y_val, batch_size=batch_size, shuffle=False)
        test_loader = create_data_loaders(X_test, y_test, batch_size=batch_size, shuffle=False)
        
        # 创建训练配置
        element_config = config
        element_config.training_config['device'] = device
        
        # 检查是否使用批处理分析
        if hasattr(config, 'analysis_config') and config.analysis_config.get('enabled', False):
            if config.analysis_config.get('batch_processing', {}).get('enabled', False):
                logger.info(f"为{element}启用批处理分析")
                # 设置批处理参数
                batch_size_analysis = config.analysis_config.get('batch_processing', {}).get('batch_size', 32)
                save_batch_results = config.analysis_config.get('batch_processing', {}).get('save_batch_results', True)
                # 确保分析目录存在
                analysis_dir = os.path.join(config.output_config['results_dir'], 'analysis', element)
                os.makedirs(analysis_dir, exist_ok=True)
        
        # 是否执行超参数调优
        best_hyperparams = None
        if tune_hyperparams:
            logger.info(f"开始为元素 {element} 进行超参数调优")
            
            # 准备超参数网格
            param_grid = config.hyperparameter_config['param_grid']
            
            # 进行超参数调优
            from hyperparameter_tuning_replacement import hyperparameter_tuning
            best_hyperparams = hyperparameter_tuning(
                element=element,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                param_grid=param_grid,
                device=device,
                batch_size=batch_size_hyperopt,
                batches_per_round=batches_per_round
            )
            
            # 更新配置
            for param, value in best_hyperparams.items():
                if param in element_config.training_config:
                    element_config.training_config[param] = value
                    
            logger.info(f"元素 {element} 超参数调优完成: {best_hyperparams}")
        
        # 使用最佳超参数训练模型
        logger.info(f"开始为元素 {element} 训练最终模型")
        
        try:
            # 训练模型
            best_model, test_metrics = process_element(element, config.model_config.get('model_type'), config.model_config.get('input_size'), config.training_config['device'] == 'cuda', config=config)
            
            # 保存结果
            result_info = {
                'element': element,
                'test_metrics': test_metrics
            }
            
            if best_hyperparams:
                result_info['hyperparams'] = best_hyperparams
                
            results[element] = result_info
            
            # 分析特征重要性
            if hasattr(config, 'analysis_config') and config.analysis_config.get('perform_feature_importance', False):
                logger.info(f"分析元素 {element} 的特征重要性")
                try:
                    # 获取分析配置
                    feature_config = config.analysis_config.get('feature_importance', {})
                    sample_size = feature_config.get('sample_size', 1000)
                    num_features = feature_config.get('num_top_features', 20)
                    save_plots = feature_config.get('save_plots', True)
                    batch_size = config.analysis_config.get('batch_size', 32)
                    
                    # 执行特征重要性分析
                    analyze_feature_importance(
                        model=best_model,
                        element=element,
                        test_loader=test_loader,
                        device=device,
                        sample_size=min(sample_size, len(X_test)),
                        num_top_features=num_features,
                        save_plots=save_plots,
                        batch_size=batch_size,
                        batch_id=None,  # 使用全部数据
                        save_batch_results=True
                    )
                    logger.info(f"元素 {element} 特征重要性分析完成")
                except Exception as e:
                    logger.error(f"元素 {element} 特征重要性分析失败: {str(e)}")
            
            # 分析残差
            if hasattr(config, 'analysis_config') and config.analysis_config.get('perform_residual_analysis', False):
                logger.info(f"分析元素 {element} 的残差")
                try:
                    # 获取分析配置
                    residual_config = config.analysis_config.get('residual_analysis', {})
                    save_predictions = residual_config.get('save_predictions', True)
                    save_plots = residual_config.get('save_plots', True)
                    batch_size = config.analysis_config.get('batch_size', 32)
                    
                    # 执行残差分析
                    analyze_residuals(
                        model=best_model,
                        element=element,
                        test_loader=test_loader,
                        device=device,
                        save_predictions=save_predictions,
                        save_plots=save_plots,
                        batch_size=batch_size,
                        batch_id=None,  # 使用全部数据
                        save_batch_results=True
                    )
                    logger.info(f"元素 {element} 残差分析完成")
                except Exception as e:
                    logger.error(f"元素 {element} 残差分析失败: {str(e)}")
                    
            # 保存结果
            result_file = os.path.join(config.output_config['results_dir'], f'results_{element}.json')
            import json
            with open(result_file, 'w') as f:
                # 将不可序列化的对象转换为可序列化的形式
                serializable_results = {}
                for key, value in result_info.items():
                    if key == 'test_metrics':
                        serializable_results[key] = {k: float(v) for k, v in value.items()}
                    elif key == 'hyperparams':
                        serializable_results[key] = {k: float(v) if isinstance(v, (int, float)) else v for k, v in value.items()}
                    else:
                        serializable_results[key] = float(value) if isinstance(value, (int, float, np.float32, np.float64)) else value
                        
                json.dump(serializable_results, f, indent=4)
                
            logger.info(f"元素 {element} 处理完成")
            
        except Exception as e:
            logger.error(f"元素 {element} 处理失败: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
    # 所有元素处理完成，评估整体结果
    logger.info("所有元素处理完成")
    if results:
        logger.info("结果摘要:")
        for element, result in results.items():
            test_metrics = result['test_metrics']
            logger.info(f"元素 {element}: 均方误差: {test_metrics['mse']:.4f}, R^2: {test_metrics['r2']:.4f}")
    
    return results

def show_batch_results(element, result_type='training', config=None):
    """
    显示分批次处理结果
    
    参数:
        element: 元素名称
        result_type: 结果类型，'training', 'evaluation', 'prediction'
        config: 配置对象
    """
    if config is None:
        import config
        
    # 确定结果目录
    if result_type == 'training':
        batch_results_dir = os.path.join(config.output_config['results_dir'], f'training_{element}_batch_results')
    elif result_type == 'evaluation':
        batch_results_dir = os.path.join(config.output_config['results_dir'], f'evaluation_{element}_batch_results')
    elif result_type == 'prediction':
        batch_results_dir = os.path.join(config.output_config['results_dir'], f'prediction_{element}_batch_results')
    else:
        logger.error(f"不支持的结果类型: {result_type}")
        return
    
    if not os.path.exists(batch_results_dir):
        logger.warning(f"找不到 {element} 的批次结果目录: {batch_results_dir}")
        return
    
    # 查找批次追踪文件
    batch_tracking_path = os.path.join(batch_results_dir, 'batch_tracking.csv')
    if not os.path.exists(batch_tracking_path):
        logger.warning(f"找不到 {element} 的批次追踪文件: {batch_tracking_path}")
        return
    
    # 读取批次追踪数据
    import pandas as pd
    batch_df = pd.read_csv(batch_tracking_path)
    
    # 显示汇总信息
    logger.info(f"{element} {result_type}批次处理结果:")
    logger.info(f"总批次数: {len(batch_df)}")
    
    if result_type == 'training':
        # 找出最佳批次（Epoch）
        if 'val_loss' in batch_df.columns:
            best_batch = batch_df.loc[batch_df['val_loss'].idxmin()]
            logger.info(f"最佳Epoch: {best_batch['epoch']} (验证损失: {best_batch['val_loss']:.6f}, RMSE: {best_batch['rmse']:.6f})")
        
        # 按时间顺序显示每个Epoch的信息
        logger.info("\n训练Epoch详情:")
        for idx, row in batch_df.iterrows():
            if idx < 5 or idx > len(batch_df) - 6 or idx % 10 == 0:  # 显示前5个、后5个和每10个epoch
                logger.info(f"Epoch {row['epoch']}: 训练损失={row['train_loss']:.6f}, 验证损失={row['val_loss']:.6f}, RMSE={row['rmse']:.6f}")
    
    elif result_type == 'evaluation':
        # 找出最佳批次
        if 'rmse' in batch_df.columns:
            best_batch = batch_df.loc[batch_df['rmse'].idxmin()]
            logger.info(f"最佳批次: {best_batch['batch_id']} (RMSE: {best_batch['rmse']:.6f}, MAE: {best_batch['mae']:.6f}, R²: {best_batch['r2']:.6f})")
        
        # 显示每个批次的信息
        logger.info("\n评估批次详情:")
        for _, row in batch_df.iterrows():
            logger.info(f"批次 {row['batch_id']}: RMSE={row['rmse']:.6f}, MAE={row['mae']:.6f}, R²={row['r2']:.6f}")
    
    elif result_type == 'prediction':
        # 计算平均值和标准差
        if 'mean' in batch_df.columns and 'std' in batch_df.columns:
            avg_mean = batch_df['mean'].mean()
            avg_std = batch_df['std'].mean()
            logger.info(f"总体预测均值: {avg_mean:.6f}, 总体预测标准差: {avg_std:.6f}")
        
        # 显示每个批次的信息
        logger.info("\n预测批次详情:")
        for _, row in batch_df.iterrows():
            logger.info(f"批次 {row['batch_id']}: 均值={row['mean']:.6f}, 标准差={row['std']:.6f}, 最小值={row['min']:.6f}, 最大值={row['max']:.6f}")
    
    # 显示结果文件位置信息
    logger.info(f"\n查看详细结果:")
    logger.info(f"1. 批次指标摘要: {batch_results_dir}/batch_*_metrics.txt")
    
    if result_type == 'training':
        logger.info(f"2. 训练趋势图: {batch_results_dir}/training_trends.png")
        logger.info(f"3. 批次散点图: {batch_results_dir}/epoch_*_scatter.png")
        logger.info(f"4. 训练总结报告: {batch_results_dir}/training_summary.txt")
    elif result_type == 'evaluation':
        logger.info(f"2. 批次趋势图: {batch_results_dir}/batch_trends.png")
        logger.info(f"3. 批次散点图: {batch_results_dir}/batch_*_scatter.png")
        logger.info(f"4. 批次误差分布图: {batch_results_dir}/batch_*_error_hist.png")
        logger.info(f"5. 评估总结报告: {batch_results_dir}/evaluation_summary.txt")
    elif result_type == 'prediction':
        logger.info(f"2. 批次趋势图: {batch_results_dir}/batch_trends.png")
        logger.info(f"3. 批次分布图: {batch_results_dir}/batch_*_distribution.png")
        logger.info(f"4. 预测总结报告: {batch_results_dir}/prediction_summary.txt")
        logger.info(f"5. 总体分布图: {batch_results_dir}/total_distribution.png")

def use_preprocessor(task='train', element='MG_FE', input_file=None, output_dir='processed_data', **kwargs):
    """
    使用preprocessdata7.py预处理器处理数据
    
    参数:
        task (str): 任务类型，'train', 'predict', 'preprocess'
        element (str): 元素名称，如'MG_FE'
        input_file (str): 输入文件路径，可以是CSV或FITS文件
        output_dir (str): 输出目录
        **kwargs: 传递给preprocessdata7.LAMOSTPreprocessor的其他参数
        
    返回:
        dict: 处理结果，包含数据路径等信息
    """
    try:
        # 检查preprocessdata7.py是否可用
        import importlib
        try:
            pp7 = importlib.import_module('preprocessdata7')
            if not hasattr(pp7, 'LAMOSTPreprocessor'):
                logger.error("preprocessdata7模块中找不到LAMOSTPreprocessor类")
                return {'success': False, 'error': 'Missing LAMOSTPreprocessor class'}
                
            logger.info(f"使用preprocessdata7.LAMOSTPreprocessor执行{task}任务，元素: {element}")
            
            # 创建预处理器实例
            preprocessor = pp7.LAMOSTPreprocessor(**kwargs)
            
            # 根据任务类型执行不同操作
            if task == 'preprocess':
                # 直接执行预处理
                if input_file is None:
                    logger.error("预处理任务需要指定input_file")
                    return {'success': False, 'error': 'Missing input_file'}
                
                result = preprocessor.process_all_data()
                return {
                    'success': True,
                    'result': result,
                    'train_data': os.path.join(output_dir, 'train_dataset.npz'),
                    'val_data': os.path.join(output_dir, 'val_dataset.npz'),
                    'test_data': os.path.join(output_dir, 'test_dataset.npz')
                }
                
            elif task == 'train':
                # 加载已处理的数据集
                train_path = os.path.join(output_dir, 'train_dataset.npz')
                val_path = os.path.join(output_dir, 'val_dataset.npz')
                test_path = os.path.join(output_dir, 'test_dataset.npz')
                
                # 检查文件是否存在
                if not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(test_path):
                    logger.warning("找不到预处理数据集，尝试执行预处理")
                    if input_file is None:
                        logger.error("预处理任务需要指定input_file")
                        return {'success': False, 'error': 'Missing input_file'}
                    
                    # 执行预处理
                    result = preprocessor.process_all_data()
                
                return {
                    'success': True,
                    'train_data': train_path,
                    'val_data': val_path,
                    'test_data': test_path
                }
                
            elif task == 'predict':
                # 预测任务
                if input_file is None:
                    logger.error("预测任务需要指定input_file")
                    return {'success': False, 'error': 'Missing input_file'}
                
                # 导入模型
                from model import load_trained_model
                
                # 加载模型
                model = load_trained_model(config.model_config['input_size'], element, config)
                if model is None:
                    logger.error(f"无法加载元素 {element} 的模型")
                    return {'success': False, 'error': f'Failed to load model for {element}'}
                
                # 执行预测
                result = preprocessor.predict_abundance(input_file, model)
                return {
                    'success': True,
                    'result': result
                }
                
            else:
                logger.error(f"不支持的任务类型: {task}")
                return {'success': False, 'error': f'Unsupported task: {task}'}
                
        except (ImportError, ModuleNotFoundError) as e:
            logger.error(f"无法导入preprocessdata7模块: {str(e)}")
            return {'success': False, 'error': f'Failed to import preprocessdata7: {str(e)}'}
            
    except Exception as e:
        logger.error(f"使用preprocessor时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='恒星光谱丰度预测模型')
    
    # 基本参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子，用于结果复现')
    parser.add_argument('--mode', type=str, choices=['train', 'tune', 'test', 'predict', 'all', 'show_results', 'analyze', 'preprocess'],
                       default='train', help='运行模式')
    parser.add_argument('--data_path', type=str, default=None,
                       help='数据文件路径，可以是.npz格式的预处理数据或CSV格式的原始数据')
    parser.add_argument('--train_data_path', type=str, default=None,
                       help='训练数据文件路径，优先级高于data_path')
    parser.add_argument('--val_data_path', type=str, default=None,
                       help='验证数据文件路径，优先级高于data_path')
    parser.add_argument('--test_data_path', type=str, default=None,
                       help='测试数据文件路径，优先级高于data_path')
    
    # 元素参数
    parser.add_argument('--elements', nargs='+', default=None,
                       help='要处理的元素列表')
    parser.add_argument('--element', type=str, default=None,
                       help='要处理的单个元素，与--elements互斥')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='学习率')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    parser.add_argument('--early_stopping', type=int, default=None,
                       help='早停轮数')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备，可选值：cpu, cuda, tpu')
    
    # 超参数调优参数
    parser.add_argument('--batch_size_hyperopt', type=int, default=None,
                       help='超参数调优时的批次大小')
    parser.add_argument('--batches_per_round', type=int, default=None,
                       help='每轮评估的批次数')
    parser.add_argument('--tune_hyperparams', action='store_true',
                       help='是否进行超参数调优')
    
    # 结果和分析参数
    parser.add_argument('--result_type', type=str,
                       choices=['training', 'evaluation', 'prediction', 'analysis'],
                       default='training', help='结果类型')
    parser.add_argument('--perform_analysis', action='store_true',
                       help='进行模型分析')
    parser.add_argument('--analysis_type', type=str,
                       choices=['feature_importance', 'residual_analysis', 'both'],
                       default='both', help='分析类型')
    parser.add_argument('--analysis_batch_size', type=int, default=None,
                       help='分析时的批次大小')
    parser.add_argument('--save_batch_results', action='store_true',
                       help='是否保存每个批次的结果')
    
    # 预处理参数
    parser.add_argument('--use_preprocessor', action='store_true',
                       help='使用preprocessdata7预处理数据')
    parser.add_argument('--csv_files', nargs='+', default=None,
                     help='preprocessdata7使用的CSV数据文件列表')
    parser.add_argument('--fits_dir', type=str, default='fits',
                       help='preprocessdata7使用的FITS文件目录')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                      help='preprocessdata7处理后的输出目录')
    parser.add_argument('--log_step', type=float, default=0.0001,
                      help='preprocessdata7使用的对数步长')
    parser.add_argument('--n_splits', type=int, default=5,
                      help='preprocessdata7使用的交叉验证折数')
    parser.add_argument('--compute_common_range', action='store_true',
                      help='preprocessdata7是否计算共同波长范围')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                     help='权重衰减系数')
    parser.add_argument('--force_new_model', action='store_true',
                     help='强制使用新模型')
    # 添加GRU和GCN控制参数
    parser.add_argument('--use_gru', action='store_true',
                     help='使用双向GRU网络')
    parser.add_argument('--no_gru', action='store_true',
                     help='不使用双向GRU网络')
    parser.add_argument('--use_gcn', action='store_true',
                     help='使用图卷积网络')
    parser.add_argument('--no_gcn', action='store_true',
                     help='不使用图卷积网络')
    
    return parser.parse_args()

def determine_device(requested_device):
    """根据请求和可用性确定计算设备"""
    logger = logging.getLogger('main') # Ensure logger is accessible or pass it
    if requested_device == 'cuda':
        if torch.cuda.is_available():
            logger.info("设备设置为 CUDA")
            return torch.device('cuda')
        else:
            logger.warning("请求使用CUDA，但CUDA不可用，将使用CPU")
            return torch.device('cpu')
    elif requested_device == 'tpu':
        if HAS_XLA: # Check if TPU support is available
            try:
                device = xm.xla_device()
                logger.info(f"设备设置为 TPU: {device}")
                return device
            except Exception as e:
                logger.warning(f"请求使用TPU，但初始化失败: {e}，将使用CPU")
                return torch.device('cpu')
        else:
            logger.warning("请求使用TPU，但torch_xla未安装，将使用CPU")
            return torch.device('cpu')
    elif requested_device == 'cpu':
         logger.info("设备设置为 CPU")
         return torch.device('cpu')
    else: # Default case if requested_device is None or unrecognized
         if torch.cuda.is_available():
             logger.info("未指定设备或无法识别，默认使用 CUDA (可用)")
             return torch.device('cuda')
         # elif HAS_XLA: # Optionally default to TPU if CUDA not available
         #    try:
         #        device = xm.xla_device()
         #        logger.info(f"未指定设备或无法识别，默认使用 TPU (可用): {device}")
         #        return device
         #    except Exception:
         #        logger.info("未指定设备或无法识别，CUDA/TPU均不可用，默认使用 CPU")
         #        return torch.device('cpu')
         else:
             logger.info("未指定设备或无法识别，默认使用 CPU")
             return torch.device('cpu')

def main():
    """程序主入口函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    if hasattr(args, 'seed') and args.seed is not None:
        set_seed(args.seed)
    else:
        set_seed(42)  # 使用默认种子
    
    logger = logging.getLogger('main')
    logger.info("开始处理命令行参数")
    
    # 处理element和elements参数
    if args.element is not None:
        # 单元素模式优先
        elements = [args.element]
    elif args.elements is not None:
        # 多元素模式
        elements = args.elements
    else:
        # 默认使用配置中的元素
        elements = config.training_config['elements']
        
    # 使用命令行参数更新配置
    if args.batch_size is not None:
        config.training_config['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config.training_config['lr'] = args.learning_rate
    if args.epochs is not None:
        config.training_config['num_epochs'] = args.epochs
    if args.early_stopping is not None:
        config.training_config['early_stopping_patience'] = args.early_stopping
    if args.device is not None:
        config.training_config['device'] = args.device
        
    # 处理GRU和GCN设置
    # 优先级规则：no_X 参数优先于 use_X 参数
    use_gru = True  # 默认使用GRU
    use_gcn = True  # 默认使用GCN
    
    if args.use_gru:
        use_gru = True
    if args.no_gru:
        use_gru = False  # no_gru 优先级高于 use_gru
    if args.use_gcn:
        use_gcn = True
    if args.no_gcn:
        use_gcn = False  # no_gcn 优先级高于 use_gcn
    
    # 将GRU和GCN设置添加到配置中
    config.use_gru = use_gru
    config.use_gcn = use_gcn
    
    logger.info(f"模型配置: 使用GRU={use_gru}, 使用GCN={use_gcn}")
    
    # 更新分析配置
    if args.perform_analysis:
        if not hasattr(config, 'analysis_config'):
            config.analysis_config = {}
        config.analysis_config['enabled'] = True
        
        if args.analysis_batch_size is not None:
            config.analysis_config['batch_size'] = args.analysis_batch_size
            
        if args.save_batch_results:
            config.analysis_config['batch_results'] = {
                'enabled': True,
                'save_batch_results': True,
                'generate_batch_summary': True,
                'generate_trend_plots': True
            }
    
    # 创建必要的目录
    setup_training_directories()
    
    # 如果指定使用preprocessdata7.py作为预处理器
    if args.use_preprocessor:
        logger.info("使用preprocessdata7进行数据处理")
        
        # 构造预处理器参数
        preprocessor_kwargs = {
            'csv_files': args.csv_files,
            'fits_dir': args.fits_dir,
            'output_dir': args.output_dir,
            'log_step': args.log_step,
            'n_splits': args.n_splits,
            'compute_common_range': args.compute_common_range
        }
        
        # 预处理模式
        if args.mode == 'preprocess':
            logger.info("执行数据预处理")
            result = use_preprocessor(
                task='preprocess',
                element=elements[0] if elements else 'MG_FE',
                **preprocessor_kwargs
            )
            
            if result['success']:
                logger.info("预处理完成，数据保存在以下位置:")
                logger.info(f"训练集: {result['train_data']}")
                logger.info(f"验证集: {result['val_data']}")
                logger.info(f"测试集: {result['test_data']}")
            else:
                logger.error(f"预处理失败: {result.get('error', '未知错误')}")
            
            return
    
    # 根据模式执行不同操作
    if args.mode == 'show_results':
        # 显示批次结果
        if len(elements) == 0:
            logger.error("显示结果时必须指定元素名称")
            return
        
        for element in elements:
            if args.result_type == 'analysis':
                # 显示分析批次结果
                analysis_type = args.analysis_type
                if analysis_type == 'both' or analysis_type == 'feature_importance':
                    show_batch_results(element, 'feature_importance')
                if analysis_type == 'both' or analysis_type == 'residual_analysis':
                    show_batch_results(element, 'residual_analysis')
            else:
                # 显示其他类型的批次结果
                show_batch_results(element, args.result_type, config)
        return
    
    if args.mode == 'analyze':
        # 执行模型分析
        for element in elements:
            logger.info(f"开始对 {element} 模型进行性能分析")
            
            # 加载训练好的模型
            model = load_trained_model(config.model_config['input_size'], element, config)
            
            if model is None:
                logger.error(f"无法加载 {element} 的模型，跳过分析")
                continue
            
            # 加载数据集 - 支持preprocessdata7.py
            try:
                if args.use_preprocessor:
                    # 使用preprocessdata7加载数据
                    result = use_preprocessor(task='train', element=element, **preprocessor_kwargs)
                    if not result['success']:
                        logger.error(f"使用preprocessdata7加载数据失败: {result.get('error', '未知错误')}")
                        continue
                        
                    val_data = load_data(result['val_data'], element)
                    test_data = load_data(result['test_data'], element)
                    train_data = load_data(result['train_data'], element)
                else:
                    # 使用默认路径
                    val_data = load_data(os.path.join('processed_data', 'val_dataset.npz'), element)
                    test_data = load_data(os.path.join('processed_data', 'test_dataset.npz'), element)
                    train_data = load_data(os.path.join('processed_data', 'train_dataset.npz'), element)
                
                # 创建数据加载器
                val_loader = create_data_loaders(val_data[0], val_data[1], 
                                               batch_size=config.analysis_config.get('batch_size', 32))
                test_loader = create_data_loaders(test_data[0], test_data[1],
                                                batch_size=config.analysis_config.get('batch_size', 32))
                train_loader = create_data_loaders(train_data[0], train_data[1],
                                                 batch_size=config.analysis_config.get('batch_size', 32))
            except Exception as e:
                logger.error(f"加载 {element} 的数据集失败: {str(e)}")
                continue
            
            # 设置设备
            device = config.training_config['device']
            
            analysis_type = args.analysis_type
            batch_size = args.analysis_batch_size if args.analysis_batch_size else config.analysis_config.get('batch_size', 32)
            save_batch_results = args.save_batch_results or config.analysis_config.get('batch_results', {}).get('save_batch_results', True)
            
            if analysis_type == 'both' or analysis_type == 'feature_importance':
                logger.info(f"开始分析 {element} 模型的特征重要性")
                feature_importance_path = analyze_feature_importance(
                    model, val_loader, device, element,
                    batch_size=batch_size,
                    save_batch_results=save_batch_results
                )
                logger.info(f"特征重要性分析完成，结果保存在: {feature_importance_path}")
            
            if analysis_type == 'both' or analysis_type == 'residual_analysis':
                logger.info(f"开始分析 {element} 模型的残差")
                residual_results = analyze_residuals(
                    model, test_loader, device, element,
                    batch_size=batch_size,
                    save_batch_results=save_batch_results
                )
                logger.info(f"残差分析完成，结果保存在: {residual_results['plot']}")
            
            if analysis_type == 'both':
                # 执行全面分析
                logger.info(f"开始对 {element} 模型进行全面性能分析")
                analyze_model_performance(
                    model, element, train_loader, val_loader, test_loader,
                    device, config.model_config['input_size'],
                    batch_size=batch_size,
                    save_batch_results=save_batch_results
                )
                logger.info(f"{element} 模型性能分析完成")
        
        return
    
    # --- Restructure Mode Handling --- 

    # Block 1: Handle ONLY train mode (NOT all)
    if args.mode == 'train':
        logger.info("模式: train - 开始训练...")
        for i, element in enumerate(elements):
            logger.info(f"训练 {element} 元素丰度预测模型")
            
            # Logic to get element-specific loaders (train_loader_element, etc.)
            # This might need adjustment if train_data is not structured for easy slicing
            element_indices = train_data[2] if len(train_data) > 2 else None
            train_loader_element, val_loader_element, test_loader_element = train_loader, val_loader, test_loader # Default
            if element_indices and isinstance(element_indices, dict) and element in element_indices:
                element_idx = element_indices[element]
                if len(train_data[1].shape) > 1:
                    # ... (code to create element specific loaders) ...
                     element_label = train_data[1][:, element_idx]
                     val_element_label = val_data[1][:, element_idx]
                     test_element_label = test_data[1][:, element_idx]
                     train_loader_element = create_data_loaders(train_data[0], element_label, batch_size=config.training_config['batch_size'])
                     val_loader_element = create_data_loaders(val_data[0], val_element_label, batch_size=config.training_config['batch_size'], shuffle=False)
                     test_loader_element = create_data_loaders(test_data[0], test_element_label, batch_size=config.training_config['batch_size'], shuffle=False)
                     logger.info(f"为元素 {element} 创建特定数据加载器用于训练")
                # else: 1D labels assumed to be correct, use original loaders
            else:
                 logger.warning(f"训练模式下无法找到元素 {element} 的索引，使用共享数据加载器")

            try:
                # Pass the potentially element-specific loaders to process_element
                # NOTE: process_element itself creates loaders again internally, this might be redundant
                # Consider refactoring process_element to accept loaders directly? 
                # For now, call it as before, it will use the updated config.
                process_element(element, 
                                config.model_config.get('model_type'), 
                                config.model_config.get('input_size'), 
                                (config.training_config['device'].type == 'cuda'), 
                                config=config
                               )
            except Exception as e:
                logger.error(f"训练元素 {element} 时出错: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue

    # Block 2: Handle tune mode OR all mode (tune/load -> update -> process)
    elif args.mode == 'tune' or args.mode == 'all':
        logger.info(f"模式: {args.mode} - 开始处理...")
        for i, element in enumerate(elements):
            # The complex logic for loading best_params, tuning, updating config,
            # and THEN calling process_element is already here (from line 1696 onwards).
            # No major change needed inside this block, just ensure the condition is correct.
            logger.info(f"为 {element} 元素进行调优/加载参数和后续处理")
            
            # --- BEGIN Re-inserting detailed logic --- 
            best_params = None
            best_params_loaded = False
            best_params_file = os.path.join(config.output_config['results_dir'], 'hyperopt', element, 'best_params.json')

            # 1. Try loading existing best parameters first
            logger.info(f"[Main Loop Debug] 检查最佳参数文件: {best_params_file}") 
            if not getattr(args, 'force_new_model', False) and os.path.exists(best_params_file):
                logger.info(f"[Main Loop Debug] 文件存在且未使用 force_new_model，尝试加载... ")
                try:
                    import json
                    with open(best_params_file, 'r') as f:
                        best_params = json.load(f)
                    if isinstance(best_params, dict) and best_params: 
                        logger.info(f"[Main Loop Debug] 成功加载并验证 best_params: {best_params}")
                        best_params_loaded = True
                    else:
                        logger.warning(f"[Main Loop Debug] 加载的 best_params.json 文件无效或为空。 best_params: {best_params}")
                        best_params = None 
                        best_params_loaded = False
                except Exception as load_err:
                    logger.warning(f"[Main Loop Debug] 加载 best_params.json 文件失败: {load_err}")
                    best_params = None 
                    best_params_loaded = False
            else:
                 if getattr(args, 'force_new_model', False):
                      logger.info("[Main Loop Debug] 使用了 force_new_model，跳过加载最佳参数文件。")
                 elif not os.path.exists(best_params_file):
                      logger.info("[Main Loop Debug] 最佳参数文件不存在。")
                 best_params_loaded = False
            
            logger.info(f"[Main Loop Debug] 加载尝试后状态: best_params_loaded={best_params_loaded}, best_params={best_params}")
            
            # 2. Run tuning if needed 
            if not best_params_loaded and args.tune_hyperparams: 
                 logger.info(f"[Main Loop] 未加载参数且设置了 tune_hyperparams，开始调优...")
                 # Prepare data, device, loaders for tuning
                 try:
                     # 首先加载数据
                     logger.info(f"[Main Loop] 为调优加载 {element} 的数据...")
                     train_path = os.path.join('processed_data', 'train_dataset.npz')
                     val_path = os.path.join('processed_data', 'val_dataset.npz')
                     
                     X_train, y_train, train_elements = load_data(train_path, element)
                     X_val, y_val, val_elements = load_data(val_path, element)
                     
                     # 确定是否需要提取特定元素的标签
                     if train_elements and isinstance(train_elements, dict) and element in train_elements:
                         element_idx = train_elements[element]
                         if len(y_train.shape) > 1:
                             y_train = y_train[:, element_idx]
                             y_val = y_val[:, element_idx]
                     
                     # 设置设备
                     current_device = determine_device(args.device)
                     logger.info(f"[Main Loop] 为调优设置设备: {current_device}")
                     
                     # 创建数据加载器
                     tune_batch_size = getattr(args, 'batch_size_hyperopt', config.tuning_config.get('batch_size', 64))
                     train_loader_tune = create_data_loaders(X_train, y_train, batch_size=tune_batch_size, shuffle=True)
                     val_loader_tune = create_data_loaders(X_val, y_val, batch_size=tune_batch_size, shuffle=False)
                     
                     logger.info(f"[Main Loop] 调优数据准备完成。训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
                     
                     # 准备参数网格
                     base_param_grid = None 
                     if hasattr(config, 'tuning_config') and hasattr(config.tuning_config, 'param_grid'):
                          base_param_grid = config.tuning_config.param_grid
                          logger.info(f"[Main Loop] 从配置加载调优网格: {base_param_grid}")
                     else:
                          logger.info("[Main Loop] 配置中无调优网格，使用默认值。")
                     
                     loaders_ready = True
                     
                 except Exception as prep_err:
                      logger.error(f"[Main Loop] 准备调优数据/配置时出错: {prep_err}")
                      logger.error(traceback.format_exc())
                      train_loader_tune, val_loader_tune = None, None
                      loaders_ready = False
                 
                 # Call tuning function if prep was successful
                 if loaders_ready:
                     try: 
                          config_module = config 
                          logger.info("[Main Loop] 调用 run_grid_search_tuning...")
                          tuning_result_params = run_grid_search_tuning(
                               element=element,
                               train_loader=train_loader_tune,
                               val_loader=val_loader_tune,
                               device=current_device,
                               config_module=config_module,
                               param_grid=base_param_grid 
                          )
                          best_params = tuning_result_params 
                          logger.info("[Main Loop] run_grid_search_tuning 调用结束")
                          
                          if best_params:
                               logger.info(f"[Main Loop] 调优找到的最佳参数: {best_params}")
                               os.makedirs(os.path.dirname(best_params_file), exist_ok=True)
                               # Serialize and save best_params to json
                               import json
                               serializable_params = {} # Make sure params are JSON serializable
                               for k, v in best_params.items():
                                   if isinstance(v, np.integer): serializable_params[k] = int(v)
                                   elif isinstance(v, np.floating): serializable_params[k] = float(v)
                                   elif isinstance(v, np.ndarray): serializable_params[k] = v.tolist()
                                   else: serializable_params[k] = v
                               with open(best_params_file, 'w') as f:
                                    json.dump(serializable_params, f, indent=4)
                               logger.info(f"[Main Loop] 最佳参数已保存到: {best_params_file}")
                          else:
                               logger.warning(f"[Main Loop] 调优完成但未找到最佳参数。")
                     except Exception as tune_err:
                          logger.error(f"[Main Loop] 超参数调优过程中发生错误: {tune_err}")
                          logger.error(traceback.format_exc())
                          best_params = None # Reset best_params on error
                 else: 
                      logger.error("[Main Loop] 未能创建调优加载器，跳过调优。")
            
            elif not args.tune_hyperparams and args.mode == 'tune':
                 logger.info(f"模式为 'tune' 但未设置 --tune_hyperparams 标志，跳过元素 {element} 的调优。")

            # Log state before the final config update check
            logger.info(f"[Main Loop Debug] 进入配置更新检查前状态: best_params={best_params}")
            
            # 3. Update config with best_params if found/loaded
            if best_params: 
                logger.info(f"[Main Loop] best_params 有效，进入配置更新块。")
                # --- Re-insert robust config update logic --- 
                try:
                    logger.info(f"[Main Loop] Config 更新前: use_gru={getattr(config, 'use_gru', 'N/A')}, use_gcn={getattr(config, 'use_gcn', 'N/A')}")
                    update_made = False
                    # Update LR, BS, WD etc.
                    lr = best_params.get('learning_rate', best_params.get('lr'))
                    if lr is not None: config.training_config['lr'] = lr; logger.info(f"  [Main Loop] 更新 LR: {lr}"); update_made=True
                    bs = best_params.get('batch_size')
                    if bs is not None: config.training_config['batch_size'] = bs; logger.info(f"  [Main Loop] 更新 Batch Size: {bs}"); update_made=True
                    wd = best_params.get('weight_decay')
                    if wd is not None: config.training_config['weight_decay'] = wd; logger.info(f"  [Main Loop] 更新 Weight Decay: {wd}"); update_made=True
                    # Update GRU/GCN
                    gru_setting = best_params.get('use_gru')
                    if gru_setting is not None and getattr(config, 'use_gru', None) != gru_setting: 
                        config.use_gru = gru_setting; logger.info(f"  [Main Loop] 更新 use_gru: {gru_setting}"); update_made = True
                    gcn_setting = best_params.get('use_gcn')
                    if gcn_setting is not None and getattr(config, 'use_gcn', None) != gcn_setting: 
                        config.use_gcn = gcn_setting; logger.info(f"  [Main Loop] 更新 use_gcn: {gcn_setting}"); update_made = True
                    logger.info(f"[Main Loop] Config 更新后: use_gru={getattr(config, 'use_gru', 'N/A')}, use_gcn={getattr(config, 'use_gcn', 'N/A')}")
                    if not update_made: logger.info("[Main Loop] 配置未发生实际更改。")
                    logger.info(f"[Main Loop] 配置更新完成。")
                except Exception as config_update_err:
                     logger.error(f"[Main Loop] 使用最佳超参数更新配置时发生错误: {config_update_err}")
                # --- End re-insert config update --- 
            else:
                 logger.warning(f"[Main Loop] best_params 无效或未找到，跳过配置更新。将使用当前默认/命令行配置。")
            
            # --- Call process_element AFTER tuning/config update only in 'all' mode ---
            if args.mode == 'all': 
                 # <<< Replace Placeholder: Insert process_element call block >>>
                 logger.info(f"[Main Loop] 模式为 'all'，使用最终配置为元素 {element} 执行训练和评估...") 
                 try:
                     final_device = determine_device(args.device) 
                     config.training_config['device'] = final_device
                     final_use_gru = getattr(config, 'use_gru', True) 
                     final_use_gcn = getattr(config, 'use_gcn', True)
                     logger.info(f"[Main Loop] 调用 process_element 前确认配置: use_gru={final_use_gru}, use_gcn={final_use_gcn}")
                     
                     process_element(element, 
                                     config.model_config.get('model_type'), 
                                     config.model_config.get('input_size'), 
                                     (final_device.type == 'cuda'), 
                                     config=config 
                                     )
                 except Exception as process_err:
                     logger.error(f"在调用 process_element 处理元素 {element} 时出错: {str(process_err)}")
                     logger.error(f"Traceback: {traceback.format_exc()}")
                     logger.info(f"[Main Loop] 跳过元素 {element} 的后续处理，继续下一个元素。")
                     continue 
                 # <<< End Replace Placeholder >>>
            elif args.mode == 'tune':
                 # <<< Fix Indentation Here >>>
                 logger.info(f"模式为 'tune'，元素 {element} 处理完成（仅调优/加载参数）。")
            # --- End call block ---

    # Block 3: Handle ONLY test mode
    elif args.mode == 'test':
        logger.info("模式: test - 开始测试...")
        for i, element in enumerate(elements):
            logger.info(f"测试 {element} 元素丰度预测模型")
            try:
                # Load best model (assuming it was trained previously)
                # Need to determine model structure based on *some* config
                # Maybe load best_params if available? Or use defaults?
                # Let's assume process_element handles loading the best *trained* model if exists
                 logger.info(f"[Test Mode] 调用 process_element 进行评估...")
                 # process_element might try to train if no checkpoint found, need modification
                 # OR: add a separate evaluate_only function?
                 # Temporary solution: Call process_element, it should load best model if train finished
                 process_element(element, 
                                 config.model_config.get('model_type'), 
                                 config.model_config.get('input_size'), 
                                 (config.training_config['device'].type == 'cuda'), 
                                 config=config
                                )
            except Exception as e:
                 logger.error(f"测试元素 {element} 时出错: {e}")
                 logger.error(traceback.format_exc())
                 continue
    
    # Block 4: Handle ONLY predict mode (Keep existing logic)
    elif args.mode == 'predict':
         logger.info("模式: predict - 开始预测...")
         # ... (Existing predict mode logic) ...
    
    # Remove the old redundant blocks at the end
    # if args.mode == 'test' or args.mode == 'all': ... (DELETE this block)


if __name__ == '__main__':
    main() 