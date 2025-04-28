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
import json 
from torchvision import transforms
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import seaborn as sns
from model import load_trained_model, predict, SpectralResCNN_GCN, predict_with_mc_dropout # <--- 移动导入到顶部
from utils import set_seed, determine_device 

# 导入自定义模块
import config
from model import SpectralResCNN_GCN, SpectralResCNNEnsemble, train, evaluate_model, load_trained_model, save_model, load_checkpoint, handle_nan_values
from evaluation import evaluate_all_elements, plot_predictions_vs_true, plot_metrics_comparison
from utils import CacheManager, ProgressManager, ask_clear_cache, setup_analysis_directories, set_seed
from multi_element_processor import MultiElementProcessor
from fits_cache import FITSCache
from hyperparameter_tuning_replacement import hyperparameter_tuning as run_grid_search_tuning
from model_analysis import analyze_model_performance, show_batch_results, analyze_feature_importance, analyze_residuals

# ============ 全局日志配置 ============ 
def setup_logging():
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'main.log')

    # 获取根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO) # 设置根 logger 的级别

    # 移除所有现有的处理器，避免重复添加
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 创建文件处理器 (INFO 级别)
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器 (可以设为 INFO 或 WARNING)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) # 控制台也显示 INFO 信息
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到根 logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info("日志系统设置完成。日志将写入 {} 并显示在控制台。".format(log_file_path))

# ========================================

# 配置logger (保留单个getLogger调用，继承全局配置)
logger = logging.getLogger(__name__)

# 添加数据增强函数
def add_noise(data, noise_level=0.01):
    """简单的向数据张量添加高斯噪声"""
    # 确保 data 和 noise 在同一设备上
    device = data.device
    noise = torch.randn_like(data, device=device) * noise_level
    noisy_data = data + noise
    return noisy_data

def load_data(data_path, element=None):
    """加载数据集并返回numpy数组"""
    try:
        # <<< MODIFIED: Added allow_pickle=True >>>
        data = np.load(data_path, allow_pickle=True)
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

def visualize_mc_uncertainty(element, mean_predictions, uncertainties, targets, output_dir):
    """
    Visualize MC Dropout uncertainty analysis results.
    """
    logger.info(f"Generating MC Dropout uncertainty plots for element {element}...")
    os.makedirs(output_dir, exist_ok=True)

    # --- Remove Font Settings --- 
    # plt.rcParams['font.sans-serif'] = ['SimHei'] 
    # plt.rcParams['axes.unicode_minus'] = False   
    # --- End Font Settings --- 

    # Ensure consistent input lengths
    min_len = min(len(mean_predictions), len(uncertainties), len(targets))
    if min_len == 0:
        logger.warning(f"MC Dropout input data is empty (element: {element}), skipping visualization.")
        return
    mean_predictions = mean_predictions[:min_len]
    uncertainties = uncertainties[:min_len]
    targets = targets[:min_len]
        
    # Calculate absolute errors
    abs_errors = np.abs(targets - mean_predictions)

    # Clean potential NaN/Inf values
    valid_mask = np.isfinite(uncertainties) & np.isfinite(abs_errors)
    if np.sum(valid_mask) < 2: 
        logger.warning(f"Insufficient valid data after cleaning NaN/Inf for MC Dropout (element: {element}), skipping visualization.")
        return
    uncertainties_valid = uncertainties[valid_mask]
    abs_errors_valid = abs_errors[valid_mask]

    # 1. Plot Uncertainty vs. Absolute Error Scatter Plot
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(uncertainties_valid, abs_errors_valid, alpha=0.4, label='Samples')

        # Add trend line
        z = np.polyfit(uncertainties_valid, abs_errors_valid, 1)
        p = np.poly1d(z)
        sorted_uncertainties = np.sort(uncertainties_valid)
        plt.plot(sorted_uncertainties, p(sorted_uncertainties), "r--", label=f'Trend (y={z[0]:.2f}x+{z[1]:.2f})')

        # Calculate Pearson correlation
        correlation, p_value = pearsonr(uncertainties_valid, abs_errors_valid)
        # English Title
        plt.title(f'{element} - MC Dropout Uncertainty vs. Absolute Error\nCorrelation: {correlation:.3f} (p={p_value:.3g})') 

        plt.xlabel('Prediction Uncertainty (Std Dev)')
        plt.ylabel('Absolute Error (|True - Mean Prediction|)')
        plt.legend()
        plt.grid(False) # <--- Explicitly disable grid
        scatter_plot_path = os.path.join(output_dir, f'{element}_mc_uncertainty_vs_error_scatter.png')
        plt.savefig(scatter_plot_path)
        plt.close()
        logger.info(f"Uncertainty vs. Error scatter plot saved: {scatter_plot_path}")
    except Exception as e:
        logger.error(f"Error plotting MC Dropout scatter plot: {e}", exc_info=True)
        plt.close() 

    # 2. Plot Error Box Plot by Uncertainty Bins
    try:
        num_bins = 4
        uncertainty_bins = pd.qcut(uncertainties_valid, q=num_bins, labels=False, duplicates='drop')
        actual_num_bins = len(np.unique(uncertainty_bins))
        if actual_num_bins == 0: raise ValueError("No valid bins after qcut")
        # English Labels for bins
        bin_labels = [f'{i*100/actual_num_bins:.0f}-{(i+1)*100/actual_num_bins:.0f}%' for i in range(actual_num_bins)]

        error_data_in_bins = []
        actual_bin_labels = []
        unique_bins = sorted(np.unique(uncertainty_bins))
        for i, bin_idx in enumerate(unique_bins):
             mask = (uncertainty_bins == bin_idx)
             if np.sum(mask) > 0:
                 error_data_in_bins.append(abs_errors_valid[mask])
                 actual_bin_labels.append(bin_labels[i])

        if not error_data_in_bins:
            raise ValueError("Could not create data for uncertainty bins")

        plt.figure(figsize=(10, 6))
        box = plt.boxplot(error_data_in_bins, labels=actual_bin_labels, patch_artist=True, showfliers=True)

        colors = plt.cm.viridis(np.linspace(0, 1, len(error_data_in_bins)))
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in box['medians']:
            median.set_color('black')
            median.set_linewidth(2)

        # English Title and Labels
        plt.title(f'{element} - Absolute Error Distribution by Uncertainty Quantile')
        plt.xlabel('Prediction Uncertainty Percentile Range')
        plt.ylabel('Absolute Error')
        plt.grid(False) # <--- Explicitly disable grid
        boxplot_path = os.path.join(output_dir, f'{element}_mc_error_boxplot_by_uncertainty.png')
        plt.savefig(boxplot_path)
        plt.close()
        logger.info(f"Error box plot by uncertainty bins saved: {boxplot_path}")

    except ValueError as ve:
        logger.warning(f"Could not create error box plot for {element} (maybe too few points or distribution issue): {ve}")
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting error box plot: {e}", exc_info=True)
        plt.close()

    logger.info(f"MC Dropout uncertainty visualization finished for {element}.")

def train_and_evaluate_model(model, train_loader, val_loader, test_loader, element, device, config, augment_fn=None):
    """
    训练和评估模型 (恢复到原始状态，仅训练和标准评估)
    """
    logger = logging.getLogger('train_eval')
    try:
        # +++ 将 device 字符串转换为 torch.device 对象 +++
        if isinstance(device, str):
            logger.info(f"[{element}] Converting device string '{device}' to torch.device object.")
            try:
                # 尝试直接用字符串创建，如果失败用 determine_device (确保它可用)
                device = torch.device(device)
            except Exception as e_conv:
                logger.warning(f"[{element}] Failed to create torch.device directly: {e_conv}. Trying determine_device.")
                # 假设 determine_device 在全局可用或已导入
                try: 
                    device = determine_device(device) 
                except NameError:
                    logger.error("determine_device function not found, falling back to CPU.")
                    device = torch.device('cpu')
        elif not isinstance(device, torch.device):
             logger.warning(f"[{element}] Received unexpected type for device: {type(device)}. Defaulting to CPU.")
             device = torch.device('cpu')
        # +++ 结束转换 +++

        # 1. 训练模型
        logger.info(f"[{element}] 开始训练 (使用设备: {device})...") # Log the actual device object type
        # 假设 train 返回: model(最佳状态), best_val_loss, final_val_r2, history
        model, best_val_loss_from_train, final_val_r2_from_train, history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            element=element,
            device=device, # <--- 传递转换后的 torch.device 对象
            config=config,
            augment_fn=augment_fn
        )
        logger.info(f"[{element}] 训练完成。最佳验证损失: {best_val_loss_from_train:.6f}, 最终验证 R²: {final_val_r2_from_train:.4f}")
        model.eval() # 确保模型处于评估模式

        # 2. 标准测试集评估
        logger.info(f"[{element}] 开始标准测试集评估...")
        loss_type = config.training_config.get('loss_function', 'MSE'); actual_loss_used = loss_type
        loss_params = config.training_config.get('loss_params', {})
        if loss_type.upper() == 'MSE': loss_fn = torch.nn.MSELoss()
        elif loss_type.upper() == 'MAE': loss_fn = torch.nn.L1Loss()
        elif loss_type.upper() == 'HUBER': delta = loss_params.get('delta', 1.0); loss_fn = torch.nn.HuberLoss(delta=delta)
        else: loss_fn = torch.nn.MSELoss(); actual_loss_used = 'MSE'
        logger.info(f"[{element}] 测试评估使用损失函数: {actual_loss_used}")
        
        # 执行评估，获取指标、预测和目标
        avg_loss, metrics, predictions, targets = evaluate_model(model, test_loader, device, loss_fn)
        
        # +++ 计算并添加系统性偏差 +++
        if predictions.size > 0 and targets.size > 0 and predictions.size == targets.size:
            valid_mask = np.isfinite(predictions) & np.isfinite(targets)
            if np.sum(valid_mask) > 0:
                 bias = np.mean(predictions[valid_mask] - targets[valid_mask])
                 metrics['bias'] = bias # 添加到指标字典
                 logger.info(f"[{element}] 计算的系统性偏差 (Mean Error): {bias:.6f}")
            else:
                 logger.warning(f"[{element}] 测试集评估后无有效数据计算偏差。")
                 metrics['bias'] = np.nan
        else:
             logger.warning(f"[{element}] 测试集预测或目标为空或长度不匹配，无法计算偏差。")
             metrics['bias'] = np.nan
        # +++ 结束计算偏差 +++
            
        logger.info(f"[{element}] 测试指标: {metrics}") # 现在会包含 bias
        
        # 保存测试指标 (现在会自动包含 bias)
        try:
            results_dir = config.output_config.get('results_dir', 'results'); element_eval_dir = os.path.join(results_dir, 'evaluation', element); os.makedirs(element_eval_dir, exist_ok=True)
            metrics_file_path = os.path.join(element_eval_dir, f'{element}_test_metrics.json')
            # 确保 bias 也能被正确序列化
            serializable_metrics = {k: (float(f'{v:.6g}') if isinstance(v, (np.floating, np.float64, np.float32)) else float(v)) if isinstance(v, (np.number, np.bool_)) else v for k, v in metrics.items()}
            with open(metrics_file_path, 'w') as f: json.dump(serializable_metrics, f, indent=4)
            logger.info(f"[{element}] 测试指标已保存到: {metrics_file_path}")
        except Exception as save_err: 
            logger.error(f"[{element}] 保存测试指标时出错: {save_err}")
            logger.error(traceback.format_exc()) # 添加 traceback 以便调试

        # 3. 返回 process_element 需要的值 (metrics 现在包含 bias)
        logger.info(f"[{element}] train_and_evaluate_model 即将返回。")
        return model, history, predictions, targets, metrics

    except Exception as e:
        logger.error(f"[{element}] 在 train_and_evaluate_model 中发生严重错误: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # 在返回的 metrics 中也添加 NaN bias
        error_metrics = {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'mse': np.nan, 'bias': np.nan}
        return None, {}, np.array([]), np.array([]), error_metrics

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

def visualize_training_progress(element, history, output_dir):
    """
    可视化训练过程中的损失和指标

    Args:
        element (str): 元素名称
        history (dict): 包含训练/验证指标历史的字典
                       (e.g., {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_r2': [], 'lr': []})
        output_dir (str): 保存绘图的目录
    """
    epochs = range(1, len(history['train_loss']) + 1)

    plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style

    # 1. Plot Loss (Train vs Val)
    plt.figure(figsize=(12, 7))
    plt.plot(epochs, history['train_loss'], 'o-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'o-', label='Validation Loss')
    plt.title(f'{element} - Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.tight_layout()
    loss_plot_path = os.path.join(output_dir, f'{element}_loss_curve.png')
    plt.savefig(loss_plot_path)
    plt.close()
    logger.info(f"训练/验证损失曲线已保存: {loss_plot_path}")

    # 2. Plot Validation R2
    if 'val_r2' in history and any(not np.isnan(r2) for r2 in history['val_r2']):
        plt.figure(figsize=(12, 7))
        plt.plot(epochs, history['val_r2'], 'o-', label='Validation R²', color='green')
        plt.title(f'{element} - Validation R² Score')
        plt.xlabel('Epoch')
        plt.ylabel('R² Score')
        # Adjust y-axis limits to focus on meaningful range, handling potential NaNs
        valid_r2 = [r2 for r2 in history['val_r2'] if not np.isnan(r2)]
        if valid_r2:
             min_r2 = min(valid_r2)
             plt.ylim(bottom=max(-1, min_r2 - 0.1)) # Adjust y-axis bottom limit, ensure not lower than -1
        else:
             plt.ylim(bottom=-1.1) # Default if no valid R2
        plt.legend()
        plt.tight_layout()
        r2_plot_path = os.path.join(output_dir, f'{element}_val_r2_curve.png')
        plt.savefig(r2_plot_path)
        plt.close()
        logger.info(f"验证 R² 曲线已保存: {r2_plot_path}")
    else:
        logger.warning(f"元素 {element} 缺少有效的验证 R² 数据，跳过绘图。")

    # 3. Plot Validation MAE
    if 'val_mae' in history and any(not np.isnan(mae) for mae in history['val_mae']):
        plt.figure(figsize=(12, 7))
        plt.plot(epochs, history['val_mae'], 'o-', label='Validation MAE', color='orange')
        plt.title(f'{element} - Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        plt.tight_layout()
        mae_plot_path = os.path.join(output_dir, f'{element}_val_mae_curve.png')
        plt.savefig(mae_plot_path)
        plt.close()
        logger.info(f"验证 MAE 曲线已保存: {mae_plot_path}")
    else:
         logger.warning(f"元素 {element} 缺少有效的验证 MAE 数据，跳过绘图。")

    # 4. Plot Learning Rate
    if 'lr' in history and history['lr']:
        plt.figure(figsize=(12, 7))
        plt.plot(epochs, history['lr'], 'o-', label='Learning Rate', color='purple')
        plt.title(f'{element} - Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log') # Use log scale for LR typically
        plt.legend()
        plt.tight_layout()
        lr_plot_path = os.path.join(output_dir, f'{element}_lr_curve.png')
        plt.savefig(lr_plot_path)
        plt.close()
        logger.info(f"学习率曲线已保存: {lr_plot_path}")

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

# --- 修改 process_element 函数 --- 
def process_element(element, config, architecture_params={}):
    # ... (保留函数开始部分：获取日志、配置、加载数据、创建模型、加载器等) ...
    logger = logging.getLogger('process_element')
    # ... (配置获取和日志记录) ...
    use_gru = getattr(config, 'use_gru', True); use_gcn = False # GCN is removed
    device = config.training_config['device']
    state_manager = TrainingStateManager(element=element, model_dir=config.model_config.get('model_dir', 'models'))
    
    # 加载数据
    # ... (加载 X_train, y_train, X_val, y_val, X_test, y_test) ...
    train_path = os.path.join('processed_data', 'train_dataset.npz'); X_train, y_train, _ = load_data(train_path, element)
    val_path = os.path.join('processed_data', 'val_dataset.npz'); X_val, y_val, _ = load_data(val_path, element)
    test_path = os.path.join('processed_data', 'test_dataset.npz'); X_test, y_test, _ = load_data(test_path, element)
    
    # 创建模型
    # ... (创建模型实例 model) ...
    model_type = config.model_config.get('model_type', 'SpectralResCNN_GCN'); 
    if model_type == 'SpectralResCNN_GCN': model = SpectralResCNN_GCN(device=device, use_gru=use_gru, **architecture_params)
    else: raise ValueError(f"未知的模型类型: {model_type}")
    
    # 创建数据加载器
    # ... (创建 train_loader, val_loader, test_loader) ...
    batch_size = config.training_config.get('batch_size', 32)
    train_loader = create_data_loaders(X_train, y_train, batch_size=batch_size)
    val_loader = create_data_loaders(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = create_data_loaders(X_test, y_test, batch_size=batch_size, shuffle=False)
    augment_fn = None # ... (增强逻辑) ...
    if config.data_config.get('augmentation_enabled', False):
        noise_level = config.data_config.get('augmentation_params', {}).get('noise_level', 0.01)
        augment_fn = lambda data: add_noise(data, noise_level)
        logger.info(f"数据增强已启用，噪声水平: {noise_level}")

    # 恢复训练状态
    # ... (state_manager.load_state() 和 checkpoint 加载逻辑) ...
    state = state_manager.load_state()
    if state and not state.get('training_completed', False):
        # ... (创建 optimizer, scheduler, 加载 checkpoint) ...
        pass

    # --- 调用 train_and_evaluate_model --- 
    logger.info(f"开始调用 train_and_evaluate_model 处理 {element}")
    # 假设它返回: model, history, original_predictions, original_targets, original_metrics
    returned_model, training_history, original_predictions, original_targets, original_metrics = train_and_evaluate_model(
        model, train_loader, val_loader, test_loader, element, device, config, augment_fn=augment_fn
    )

    if returned_model is None: # 检查 train_and_evaluate_model 是否成功
        logger.error(f"元素 {element} 的 train_and_evaluate_model 执行失败，跳过后续处理。")
        return None, None # 或者适合的错误返回值
    else:
        model = returned_model # 使用返回的最佳模型

    # --- 标准可视化 (现在在这里进行) --- 
    logger.info(f"为 {element} 生成标准可视化图表")
    plots_dir = config.output_config.get('plots_dir', 'plots')
    element_plot_dir = os.path.join(plots_dir, 'evaluation', element)
    os.makedirs(element_plot_dir, exist_ok=True)
    try:
        # 1. 训练过程可视化
        if training_history:
             visualize_training_progress(element, training_history, element_plot_dir)
        else: logger.warning(f"[{element}] 缺少训练历史，无法绘制训练过程图表。")
        
        # 2. 评估结果可视化 (使用返回的原始预测和目标)
        if original_predictions.size > 0 and original_targets.size > 0:
            # --- 修改调用点，使用新的 visualize_simple_predictions --- 
            logger.info(f"调用 visualize_simple_predictions 进行标准评估可视化...")
            visualize_simple_predictions( # <--- 调用新函数
                element=element,
                targets=original_targets, # 直接传递数组
                predictions=original_predictions, # 直接传递数组
                output_dir=element_plot_dir
            )
            # --- 结束修改调用点 ---
        else: logger.warning(f"[{element}] 缺少原始预测或目标数据，无法绘制评估结果图表。")
            
    except NameError as ne:
         if 'visualize_training_progress' in str(ne):
              logger.error("函数 visualize_training_progress 未定义或未导入!")
         elif 'visualize_simple_predictions' in str(ne): # <--- 更新检查的函数名
              logger.error("函数 visualize_simple_predictions 未定义或未导入!")
         else: logger.error(f"标准可视化时发生 NameError: {ne}")
    except Exception as viz_err:
         logger.error(f"生成标准可视化时出错: {viz_err}", exc_info=True)

    # --- MC Dropout 评估和可视化 (新添加的部分) --- 
    logger.info(f"为 {element} 进行 MC Dropout 不确定性评估...")
    try:
        # --- 移除内部导入，使用顶层导入 --- 
        # from model import predict_with_mc_dropout 
        mc_samples = config.training_config.get('mc_samples', 50)
        mc_mean_predictions, mc_uncertainties, mc_targets = predict_with_mc_dropout(
            model=model, 
            data_loader=test_loader, 
            device=device, # 确保这里的 device 是 torch.device 对象
            mc_samples=mc_samples
        )
        # ... (后续 MC 代码) ...
    except NameError as ne: # <--- 修改：只捕获 NameError
         if 'predict_with_mc_dropout' in str(ne):
              logger.error(f"[{element}] 函数 predict_with_mc_dropout 未定义，跳过 MC Dropout。确保已在顶部导入。")
         elif 'visualize_mc_uncertainty' in str(ne):
              logger.error(f"[{element}] 函数 visualize_mc_uncertainty 未定义，跳过 MC 可视化。")
         else: logger.error(f"[{element}] MC Dropout 过程中发生 NameError: {ne}")
    except Exception as mc_err:
         logger.error(f"[{element}] 执行 MC Dropout 时出错: {mc_err}", exc_info=True)

    # --- 处理完成 --- 
    # 训练完成，标记状态 (如果需要)
    state_manager.save_state(2, config.training_config.get('num_epochs', 100), 
                            original_metrics.get('loss', np.nan), # 使用原始评估损失？或最佳验证损失？
                            0, stage1_completed=True, training_completed=True)
    logger.info(f"元素 {element} 的处理完成")
    logger.info(f"最终测试指标 (原始): {original_metrics}")
    
    # 返回模型和原始指标给 main 函数 (如果需要)
    return model, original_metrics

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
            best_model, test_metrics = process_element(element, config=config, architecture_params=config.model_config.get('architecture_params', {}))
            
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

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='恒星光谱丰度预测模型')
    
    # 基本参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子，用于结果复现')
    parser.add_argument('--mode', type=str, choices=['train', 'tune', 'test', 'predict', 'all', 'show_results', 'analyze', 'preprocess', 'predict_new'], # 添加 predict_new 模式
                       default='train', help='运行模式')
    parser.add_argument('--data_path', type=str, default=None,
                       help='数据文件路径，可以是.npz格式的预处理数据或CSV格式的原始数据')
    parser.add_argument('--train_data_path', type=str, default=None,
                       help='训练数据文件路径，优先级高于data_path')
    parser.add_argument('--val_data_path', type=str, default=None,
                       help='验证数据文件路径，优先级高于data_path')
    parser.add_argument('--test_data_path', type=str, default=None,
                       help='测试数据文件路径，优先级高于data_path')
    # +++ 添加新参数 +++
    parser.add_argument('--predict_data_path', type=str, default=None,
                       help='要进行预测的新数据集路径 (只包含光谱)')
    # +++ 结束添加 +++
    
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

def main(args): # <--- 接收 args
    """程序主入口函数"""
    # 解析命令行参数 - 不再需要，已在外部调用
    # args = parse_args()
    
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
    # GCN已被移除，强制设为False以避免混淆
    config.use_gcn = False 
    
    logger.info(f"模型配置: 使用GRU={use_gru}, 使用GCN={config.use_gcn}") # Log the forced value
    
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
    
    # --- 主要处理循环 --- 
    processed_elements_count = 0
    if args.mode in ['train', 'tune', 'all', 'test']: # 这些模式会涉及 process_element
        logger.info(f"开始在 {args.mode} 模式下处理元素: {elements}")
        for i, element in enumerate(elements):
             logger.info(f"--- 处理元素 {i+1}/{len(elements)}: {element} ---")
             try:
                 # --- 确定架构参数 (结合 config 和 best_params) ---
                 base_arch_params = config.model_config.get('architecture_params', {})
                 final_arch_params = base_arch_params.copy()
                 best_params_file = os.path.join(config.output_config['results_dir'], 'hyperopt', element, 'best_params.json')
                 if os.path.exists(best_params_file):
                     try:
                         with open(best_params_file, 'r') as f: best_params = json.load(f)
                         if best_params:
                             logger.info(f"为 {element} 加载已保存的最佳超参数。")
                             # 更新架构参数 (示例键，根据你的实际情况调整)
                             arch_keys = ['initial_channels', 'res_channels', 'gru_hidden_size', 'fc_hidden_layers', 'num_res_blocks']
                             for key in arch_keys:
                                 if key in best_params: final_arch_params[key] = best_params[key]
                     except Exception as e:
                          logger.warning(f"加载或应用 {element} 的最佳参数失败: {e}")
                 else:
                      logger.info(f"未找到 {element} 的最佳超参数文件，使用配置默认值。")
                 # --- 结束架构参数确定 ---
                 
                 # 调用 process_element (假设它处理训练/评估/测试)
                 process_element(element, config=config, architecture_params=final_arch_params)
                 processed_elements_count += 1
                 
             except Exception as e:
                 logger.error(f"处理元素 {element} 时发生严重错误: {str(e)}")
                 logger.error(f"Traceback: {traceback.format_exc()}")
                 continue # 继续处理下一个元素
        logger.info(f"元素处理循环完成。成功处理 {processed_elements_count}/{len(elements)} 个元素。")

    # --- 新增：处理 predict_new 模式或在其他模式后进行预测 ---
    if args.mode == 'predict_new' or (args.predict_data_path is not None and processed_elements_count > 0):
        if args.predict_data_path is None:
             logger.error("需要在 predict_new 模式下或希望进行新数据预测时，通过 --predict_data_path 指定文件路径。")
        else:
             logger.info(f"--- 开始对新数据进行预测和分析: {args.predict_data_path} ---")
             # 确定要预测哪些元素 (可以与训练/测试的元素相同，或通过参数指定)
             elements_to_predict = elements # 复用之前定义的列表
             if not elements_to_predict:
                 logger.error("没有指定要预测的元素 (通过 --element 或 --elements)。")
             else:
                 predict_success_count = 0
                 for element in elements_to_predict:
                     logger.info(f"--- 预测新数据的元素: {element} ---")
                     try:
                         # *** 确保 predict_and_analyze_new_data 函数已定义在 main 之前 ***
                         predict_and_analyze_new_data(element, config, args.predict_data_path)
                         predict_success_count += 1
                     except Exception as e:
                         logger.error(f"预测和分析元素 {element} 的新数据时出错: {str(e)}")
                         logger.error(f"Traceback: {traceback.format_exc()}")
                         continue
                 logger.info(f"新数据预测分析完成。成功处理 {predict_success_count}/{len(elements_to_predict)} 个元素。")
    elif args.predict_data_path is not None and processed_elements_count == 0 and args.mode not in ['predict_new']:
         logger.warning("指定了 --predict_data_path 但之前的模式未能成功处理任何元素，跳过新数据预测。")

    # --- 处理其他模式 (show_results, analyze, preprocess, predict - 如果需要的话) --- 
    elif args.mode == 'show_results':
         # ... (show_results 的逻辑) ...
         logger.info("执行 show_results 模式...") # 示例
         pass
    elif args.mode == 'analyze':
         # ... (analyze 的逻辑) ...
         logger.info("执行 analyze 模式...") # 示例
         pass
    elif args.mode == 'preprocess':
         # ... (preprocess 的逻辑) ...
         logger.info("执行 preprocess 模式...") # 示例
         pass
    elif args.mode == 'predict': 
         # ... (原始 predict 模式的逻辑，如果它不同于 predict_new) ...
         logger.warning("原始 'predict' 模式的逻辑未在此处实现，如果需要请添加。")
         pass 

    logger.info("脚本执行完毕。")

# +++ 新增绘图函数 1: 预测值分布 +++
def plot_prediction_distribution(element, predictions, output_dir):
    """Plots the distribution of new predictions (KDE)."""
    logger = logging.getLogger('plot_dist')
    logger.info(f"[{element}] Plotting distribution of new predictions...")
    os.makedirs(output_dir, exist_ok=True)

    valid_predictions = predictions[np.isfinite(predictions)]
    if len(valid_predictions) == 0:
        logger.warning(f"[{element}] No finite predictions to plot distribution.")
        return

    plt.figure(figsize=(10, 6))
    sns.kdeplot(valid_predictions, fill=True, color='skyblue')
    
    mean_pred = np.mean(valid_predictions)
    median_pred = np.median(valid_predictions)
    std_pred = np.std(valid_predictions)
    
    stats_text = f'Mean: {mean_pred:.3f}\nMedian: {median_pred:.3f}\nStd Dev: {std_pred:.3f}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title(f'{element} - Distribution of Predicted Abundances (New Data)')
    plt.xlabel(f'Predicted [{element}] Abundance')
    plt.ylabel('Density')
    plt.grid(False)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'{element}_new_prediction_distribution.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"[{element}] New prediction distribution plot saved to: {plot_path}")

# +++ 新增绘图函数 2: 分布对比 +++
def plot_distribution_comparison(element, new_predictions, reference_labels, output_dir):
    """Compares the distribution of new predictions and reference true labels."""
    logger = logging.getLogger('plot_compare')
    logger.info(f"[{element}] Plotting comparison of prediction distribution vs reference labels...")
    os.makedirs(output_dir, exist_ok=True)

    valid_new_predictions = new_predictions[np.isfinite(new_predictions)]
    valid_reference_labels = reference_labels[np.isfinite(reference_labels)]

    if len(valid_new_predictions) == 0 or len(valid_reference_labels) == 0:
        logger.warning(f"[{element}] No finite data for distribution comparison plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.kdeplot(valid_reference_labels, fill=True, color='coral', label='Reference True Labels (Train/Val/Test)', alpha=0.5)
    sns.kdeplot(valid_new_predictions, fill=True, color='skyblue', label='New Predictions', alpha=0.5)

    plt.title(f'{element} - Distribution Comparison: New Predictions vs. Reference Labels')
    plt.xlabel(f'[{element}] Abundance')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f'{element}_prediction_vs_reference_distribution.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"[{element}] Distribution comparison plot saved to: {plot_path}")

# +++ 新增处理函数: 预测和分析新数据 +++
def predict_and_analyze_new_data(element, config, predict_data_path):
    """Loads model, predicts on new data, generates distribution plots, prioritizing best_model."""
    logger = logging.getLogger('predict_new')
    logger.info(f"[{element}] Starting prediction and analysis for new data: {predict_data_path}")
    
    device = determine_device(config.training_config.get('device')) 
    model_dir = config.model_config.get('model_dir', 'models')
    output_base_dir = config.output_config.get('plots_dir', 'plots')
    analysis_plot_dir = os.path.join(output_base_dir, 'prediction_analysis', element)
    os.makedirs(analysis_plot_dir, exist_ok=True)

    # 1. 加载训练好的模型 (优先 best_model.pth)
    logger.info(f"[{element}] Loading trained model...")
    use_gru = getattr(config, 'use_gru', True)
    architecture_params = config.model_config.get('architecture_params', {})
    # ... (加载 best_params 文件并更新 architecture_params 的逻辑保持不变) ...
    best_params_file = os.path.join(config.output_config['results_dir'], 'hyperopt', element, 'best_params.json')
    if os.path.exists(best_params_file):
        try:
            with open(best_params_file, 'r') as f: best_params = json.load(f)
            if best_params:
                logger.info(f"[{element}] Found best params, overriding architecture params from file: {best_params_file}")
                arch_keys = [k for k in best_params if k in architecture_params or k in ['gru_hidden_size', 'num_res_blocks']] 
                for key in arch_keys:
                     if key in best_params: architecture_params[key] = best_params[key]
        except Exception as e:
            logger.warning(f"[{element}] Failed to load or apply best params from {best_params_file}: {e}")

    # --- 确保使用正确的模型类 --- 
    model_type = config.model_config.get('model_type', 'SpectralResCNN_GCN') 
    if model_type == 'SpectralResCNN_GCN': 
        model = SpectralResCNN_GCN(device=device, use_gru=use_gru, **architecture_params)
    else: 
        logger.error(f"Unsupported model type '{model_type}' for loading in predict_and_analyze.")
        return

    # --- 修改加载逻辑: 优先 best_model.pth --- 
    best_model_path = os.path.join(model_dir, f'{element}_best_model.pth')
    checkpoint_path = os.path.join(model_dir, f'{element}_checkpoint.pth') # 使用训练时保存的检查点文件名
    loaded_successfully = False

    if os.path.exists(best_model_path):
        logger.info(f"[{element}] Attempting to load best model: {best_model_path}")
        try:
            # load_trained_model_core (或直接 torch.load) 应该处理状态字典
            checkpoint = torch.load(best_model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"[{element}] Best model state loaded successfully from {best_model_path} (Epoch {checkpoint.get('epoch', 'N/A')})")
                loaded_successfully = True
            else:
                logger.warning(f"[{element}] Best model file {best_model_path} does not contain 'model_state_dict'.")
        except Exception as e:
            logger.warning(f"[{element}] Failed to load best model from {best_model_path}: {e}")
    
    if not loaded_successfully and os.path.exists(checkpoint_path):
        logger.warning(f"[{element}] Best model not found or failed to load. Attempting to load checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"[{element}] Checkpoint model state loaded successfully from {checkpoint_path} (Epoch {checkpoint.get('epoch', 'N/A')})")
                loaded_successfully = True
            else:
                logger.warning(f"[{element}] Checkpoint file {checkpoint_path} does not contain 'model_state_dict'.")
        except Exception as e:
            logger.warning(f"[{element}] Failed to load checkpoint model from {checkpoint_path}: {e}")

    if not loaded_successfully:
         logger.error(f"[{element}] Failed to load model weights from both best model and checkpoint. Cannot proceed with prediction.")
         return
    # --- 结束加载逻辑修改 --- 

    model.to(device)
    model.eval()

    # ... (加载新数据 X_pred 的逻辑不变) ...
    logger.info(f"[{element}] Loading new prediction data (spectra only) from: {predict_data_path}")
    try:
        data = np.load(predict_data_path, allow_pickle=True)
        if 'spectra' in data:
            X_pred = data['spectra']
        elif 'X' in data:
            X_pred = data['X']
        else:
            keys = list(data.keys())
            logger.warning(f"Cannot find 'spectra' or 'X' key in {predict_data_path}. Attempting to use the first key '{keys[0]}' as spectra.")
            if keys: X_pred = data[keys[0]]
            else: raise ValueError("No data found in the prediction file.")
        if not isinstance(X_pred, np.ndarray):
            raise TypeError(f"Loaded prediction spectra data is not a numpy array (type: {type(X_pred)}). Check the .npz file structure.")
        logger.info(f"[{element}] Loaded {len(X_pred)} spectra for prediction.")
    except Exception as e:
        logger.error(f"[{element}] Failed to load prediction data: {e}")
        logger.error(traceback.format_exc())
        return

    # ... (创建 pred_loader 的逻辑不变) ...
    try:
        pred_loader = create_data_loaders(X_pred, np.zeros(len(X_pred)), 
                                          batch_size=config.training_config.get('batch_size', 128), 
                                          shuffle=False)
    except Exception as e:
         logger.error(f"[{element}] Failed to create prediction DataLoader: {e}")
         logger.error(traceback.format_exc())
         return

    # ... (执行预测的逻辑不变) ...
    logger.info(f"[{element}] Predicting abundances for new data...")
    all_predictions = []
    try:
        with torch.no_grad():
            for batch_spectra, _ in pred_loader: 
                batch_spectra = batch_spectra.to(device)
                batch_spectra, _, _ = handle_nan_values(batch_spectra, replacement_strategy='mean', name="预测输入") 
                with torch.cuda.amp.autocast(enabled=(str(device) == 'cuda')): # Use string comparison
                     outputs = model(batch_spectra)
                outputs, _, _ = handle_nan_values(outputs, replacement_strategy='zero', name="预测输出")
                all_predictions.append(outputs.cpu().numpy())
    except Exception as pred_err:
         logger.error(f"[{element}] Error during prediction loop: {pred_err}")
         logger.error(traceback.format_exc())
         return
    if not all_predictions: logger.error(f"[{element}] Prediction failed or produced no results."); return
    try: new_predictions = np.vstack(all_predictions).flatten()
    except ValueError as stack_err: logger.error(f"[{element}] Error stacking prediction results: {stack_err}"); return
    logger.info(f"[{element}] Prediction complete. Generated {len(new_predictions)} predictions.")

    # ... (加载参考标签 reference_labels 的逻辑不变) ...
    logger.info(f"[{element}] Loading reference true labels from Train/Val/Test sets...")
    reference_labels = []
    data_dir = config.data_config.get('processed_data_dir', 'processed_data')
    try:
        for phase in ['train', 'val', 'test']:
            data_path = os.path.join(data_dir, f'{phase}_dataset.npz')
            if os.path.exists(data_path):
                _, y, elements_info = load_data(data_path, element)
                if y is not None:
                    if len(y.shape) > 1 and y.shape[1] > 1:
                        if elements_info and isinstance(elements_info, dict) and element in elements_info:
                            element_key_list = list(elements_info.keys())
                            if element in element_key_list:
                                 element_idx = element_key_list.index(element)
                                 reference_labels.append(y[:, element_idx])
                            else:
                                 logger.warning(f"Element '{element}' not found in keys of elements_info dict in {phase}_dataset.npz. Keys: {element_key_list}")
                                 reference_labels.append(y[:, 0]) 
                        else:
                             logger.warning(f"Cannot determine column for {element} in {phase} labels (no/invalid elements_info), using first column.")
                             reference_labels.append(y[:, 0])
                    else:
                        reference_labels.append(y.flatten())
            else: logger.warning(f"Reference data file not found: {data_path}")
        if not reference_labels: logger.error(f"[{element}] Could not load any reference labels."); return
        reference_labels = np.concatenate(reference_labels)
        logger.info(f"[{element}] Loaded {len(reference_labels)} reference labels.")
    except Exception as e:
        logger.error(f"[{element}] Failed to load reference labels: {e}"); logger.error(traceback.format_exc()); return

    # ... (调用绘图函数的逻辑不变) ...
    plot_prediction_distribution(element, new_predictions, analysis_plot_dir)
    plot_distribution_comparison(element, new_predictions, reference_labels, analysis_plot_dir)
    logger.info(f"[{element}] Prediction analysis plots generated in: {analysis_plot_dir}")

if __name__ == '__main__':
    setup_logging()
    args = parse_args()
    # --- 确保函数定义在调用之前 --- 
    # (函数定义现在应该在 main 之前了)
    main(args)