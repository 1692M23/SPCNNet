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
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import pandas as pd
import glob
import traceback

# 导入自定义模块
import config
from model import SpectralResCNN, SpectralResCNN_GCN, SpectralResCNNEnsemble, train, evaluate_model, load_trained_model
from evaluation import evaluate_all_elements, plot_predictions_vs_true, plot_metrics_comparison
from utils import CacheManager, ProgressManager, ask_clear_cache, setup_analysis_directories
from multi_element_processor import MultiElementProcessor
from fits_cache import FITSCache
from hyperparameter_tuning_replacement import hyperparameter_tuning
from model_analysis import analyze_model_performance, show_batch_results, analyze_feature_importance, analyze_residuals

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
            if elements is not None:
                if isinstance(elements, dict) and element in elements:
                    logger.info(f"找到元素 {element} 的索引")
                elif isinstance(elements, dict):
                    logger.warning(f"元素 {element} 不在索引字典中")
                else:
                    logger.warning(f"elements不是字典类型，无法查找特定元素索引")
                    # 如果elements是数组，表示我们可能有每个样本的ID，而不是元素索引
                    # 在这种情况下，返回mock字典，将特定元素映射到0，表示我们的abundance是单元素的
                    elements = {element: 0}  # 创建模拟字典，用于单元素数据
            else:
                # 如果没有elements信息，创建一个简单的映射
                logger.warning(f"数据中没有元素索引信息，假设abundance对应 {element}")
                elements = {element: 0}  # 创建模拟字典，用于单元素数据
                
        logger.info(f"成功加载数据: {X.shape}, {y.shape}")
        return X, y, elements
    except Exception as e:
        logger.error(f"加载数据时出错: {e}")
        raise

def create_data_loaders(spectra, labels, batch_size=32, shuffle=True):
    """
    创建PyTorch数据加载器，支持多种格式的输入数据
    
    参数:
        spectra (numpy.ndarray): 光谱数据，形状可能为:
            - [n_samples, n_wavelengths] (2D)
            - [n_samples, channels, n_wavelengths] (3D)
            - [n_samples, 1, 1, n_wavelengths] (4D)
        labels (numpy.ndarray): 标签数据，形状为 [n_samples] 或 [n_samples, n_labels]
        batch_size (int): 批次大小
        shuffle (bool): 是否打乱数据
        
    返回:
        torch.utils.data.DataLoader: 数据加载器
    """
    logger.info(f"输入数据形状: {spectra.shape}")
    
    # 数据增强：添加随机噪声
    def add_noise(data, snr_min=10, snr_max=30):
        signal_power = np.mean(data ** 2)
        for i in range(len(data)):
            snr = np.random.uniform(snr_min, snr_max)
            noise_power = signal_power / (10 ** (snr / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), data[i].shape)
            data[i] = data[i] + noise
        return data
    
    # 对训练数据进行增强
    if shuffle:  # 只对训练集进行增强
        spectra = add_noise(spectra.copy())
    
    # 如果标签是一维的但需要二维，进行reshape
    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)
        logger.info(f"将标签reshape为二维: {labels.shape}")
    
    # 处理不同形状的光谱数据
    if len(spectra.shape) == 2:
        # 2D数据 [n_samples, n_wavelengths] -> 添加通道维度 [n_samples, 1, n_wavelengths]
        spectra_tensor = torch.FloatTensor(spectra).unsqueeze(1)
        logger.info(f"2D数据添加通道维度后形状: {spectra_tensor.shape}")
    elif len(spectra.shape) == 3:
        # 检查是否已经是正确的形状 [n_samples, channels, n_wavelengths]
        # 或者是 [n_samples, n_wavelengths, features]
        if spectra.shape[2] > spectra.shape[1]:
            # 可能是 [n_samples, features, n_wavelengths] 格式，需要转置
            spectra_tensor = torch.FloatTensor(spectra).transpose(1, 2)
            logger.info(f"3D数据转置后形状: {spectra_tensor.shape}")
        else:
            # 已经是正确的形状
            spectra_tensor = torch.FloatTensor(spectra)
            logger.info(f"保持3D数据原有形状: {spectra_tensor.shape}")
    elif len(spectra.shape) == 4:
        # 4D数据 [n_samples, 1, 1, n_wavelengths] -> 去掉多余维度 [n_samples, 1, n_wavelengths]
        spectra_tensor = torch.FloatTensor(spectra).squeeze(2)
        logger.info(f"4D数据压缩后形状: {spectra_tensor.shape}")
    else:
        # 不支持的形状
        logger.error(f"不支持的数据形状: {spectra.shape}")
        raise ValueError(f"不支持的数据形状: {spectra.shape}")
    
    # 确保标签是torch tensor
    labels_tensor = torch.FloatTensor(labels)
    
    # 确保数据维度正确 [batch_size, channels, length]
    if len(spectra_tensor.shape) != 3:
        logger.error(f"处理后的数据维度不正确: {spectra_tensor.shape}")
        raise ValueError(f"处理后的数据维度不正确: {spectra_tensor.shape}")
    
    # 创建数据集
    dataset = TensorDataset(spectra_tensor, labels_tensor)
    
    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=0  # 避免多进程问题
    )
    
    # 检查第一个批次的数据形状
    for batch in data_loader:
        spectra_batch, labels_batch = batch
        logger.info(f"批次数据形状: 特征={spectra_batch.shape}, 标签={labels_batch.shape}")
        break
    
    return data_loader

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

def train_and_evaluate_model(train_loader, val_loader, test_loader, element, config):
    """
    训练和评估模型的主函数
    """
    # 检查数据加载器
    logger = logging.getLogger('main')
    logger.info(f"检查{element}的数据加载器...")
    
    # 检查训练集
    train_data = next(iter(train_loader))
    logger.info(f"训练集批次形状: 数据={train_data[0].shape}, 标签={train_data[1].shape}")
    logger.info(f"训练集数据范围: [{train_data[0].min():.6f}, {train_data[0].max():.6f}]")
    logger.info(f"训练集标签范围: [{train_data[1].min():.6f}, {train_data[1].max():.6f}]")
    
    # 检查验证集
    val_data = next(iter(val_loader))
    logger.info(f"验证集批次形状: 数据={val_data[0].shape}, 标签={val_data[1].shape}")
    logger.info(f"验证集数据范围: [{val_data[0].min():.6f}, {val_data[0].max():.6f}]")
    logger.info(f"验证集标签范围: [{val_data[1].min():.6f}, {val_data[1].max():.6f}]")
    
    # 在创建模型前，读取数据确定input_size
    X, y, _ = load_data(os.path.join('processed_data', 'train_dataset.npz'), element)
    actual_input_size = X.shape[1] if len(X.shape) == 2 else X.shape[2]

    # 创建模型时传入实际尺寸
    model = SpectralResCNN_GCN(actual_input_size).to(config.training_config['device'])
    logger.info(f"模型结构:\n{model}")
    
    # 设置超参数
    hyperparams = {
        'lr': config.training_config['lr'],
        'weight_decay': config.training_config['weight_decay'],
        'num_epochs': config.training_config['num_epochs'],
        'patience': config.training_config['early_stopping_patience']
    }
    
    # 训练模型
    train_losses, val_losses = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config={
            'training': {
                'lr': hyperparams['lr'],
                'weight_decay': hyperparams['weight_decay'],
                'num_epochs': hyperparams['num_epochs'],
                'early_stopping_patience': hyperparams['patience'],
                'device': config.training_config['device']
            },
            'model_config': {
                'model_dir': config.model_config['model_dir']
            }
        },
        device=config.training_config['device'],
        element=element
    )
    
    # 获取最佳验证损失
    best_val_loss = min(val_losses) if val_losses else float('inf')
    
    # 加载最佳模型
    best_model = load_trained_model(config.model_config['input_size'], element, config)
    
    # 在测试集上评估
    test_metrics = evaluate_model(best_model, test_loader, config.training_config['device'])
    
    # 分析模型性能（特征重要性和残差）
    if hasattr(config, 'analysis_config') and config.analysis_config.get('enabled', False):
        logger.info(f"开始对{element}模型进行性能分析...")
        
        # 获取分析配置
        batch_size = config.analysis_config.get('batch_size', 32)
        save_batch_results = config.analysis_config.get('batch_results', {}).get('save_batch_results', True)
        
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
    
    return best_model, best_val_loss, test_metrics

def hyperparameter_tuning(element, train_loader, val_loader, grid=None, device=None):
    """
    超参数调优
    
    参数:
        element (str): 元素名称
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        grid (dict): 网格搜索参数
        device (str): 计算设备
        
    返回:
        dict: 最佳超参数
    """
    if device is None:
        device = config.training_config['device']
    
    # 尝试使用批量超参数优化
    try:
        import pickle
        import argparse
        from batch_hyperopt import run_element_hyperopt
        
        logger.info(f"使用批量超参数优化方法为 {element} 获取2组最优超参数")
        
        # 首先尝试读取已有结果
        results_dir = os.path.join(config.output_config['results_dir'], 'hyperopt')
        os.makedirs(results_dir, exist_ok=True)
        
        final_results_file = os.path.join(results_dir, f'{element}_best_params.pkl')
        
        # 如果存在最终结果文件并且不再需要更新，直接加载并返回结果
        if os.path.exists(final_results_file):
            try:
                with open(final_results_file, 'rb') as f:
                    results = pickle.load(f)
                    
                # 检查是否已完成足够批次的处理
                if len(results.get('processed_batches', [])) >= 5:  # 假设5批次足够
                    logger.info(f"从缓存加载 {element} 的最佳超参数: set1={results['best_params_set1']}, set2={results['best_params_set2']}")
                    
                    # 返回第一组参数（后续可以根据需要选择使用哪组参数）
                    return results['best_params_set1']
            except Exception as e:
                logger.warning(f"读取缓存文件失败: {e}，将重新进行超参数优化")
        
        # 获取命令行参数中的批量大小和每轮批次数量
        # 尝试解析已经存在的命令行参数
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size_hyperopt', type=int, default=1000)
        parser.add_argument('--batches_per_round', type=int, default=2)
        
        try:
            import sys
            args, _ = parser.parse_known_args()
            batch_size = args.batch_size_hyperopt
            batches_per_round = args.batches_per_round
        except:
            # 如果解析失败，使用默认值
            batch_size = 1000
            batches_per_round = 2
        
        logger.info(f"超参数优化批量大小: {batch_size}, 每轮批次数: {batches_per_round}")
        
        # 运行批量超参数优化
        results = run_element_hyperopt(
            element=element,
            batch_size=batch_size,
            batches_per_round=batches_per_round
        )
        
        if results:
            logger.info(f"{element} 批量超参数优化完成:")
            logger.info(f"最佳参数组1: {results['best_params_set1']}")
            logger.info(f"最佳参数组2: {results['best_params_set2']}")
            
            # 默认返回第一组参数
            return results['best_params_set1']
            
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning(f"无法使用批量超参数优化模块: {e}")
    
    # 检查是否可以使用高级优化方法（hyperopt）
    try:
        from hyperopt_tuning import run_hyperopt_tuning, load_best_params
        
        # 检查是否已有保存的超参数
        saved_params = load_best_params(element)
        if saved_params:
            logger.info(f"从缓存加载 {element} 的最佳超参数: {saved_params}")
            return saved_params
        
        # 使用两阶段超参数优化
        logger.info(f"使用两阶段hyperopt优化 {element} 的超参数")
        best_params = run_hyperopt_tuning(
            element=element,
            train_loader=train_loader,
            val_loader=val_loader,
            max_evals_stage1=20,  # 减少评估次数以节省时间
            max_evals_stage2=10,
            device=device
        )
        
        return best_params
        
    except (ImportError, ModuleNotFoundError):
        logger.warning("无法导入hyperopt_tuning模块，回退到网格搜索方法")
    
    # 回退到简单的网格搜索
    # 如果没有提供网格参数，使用默认值
    if grid is None:
        grid = {
            'lr': [0.001, 0.0005, 0.0001],
            'weight_decay': [1e-4, 1e-5, 1e-6],
            'num_blocks': [2, 3, 4],
            'num_filters': [32, 64]
        }
    
    logger.info(f"开始 {element} 的超参数调优，参数网格:")
    for param, values in grid.items():
        logger.info(f"  {param}: {values}")
    
    # 生成所有参数组合
    param_combinations = list(ParameterGrid(grid))
    logger.info(f"共 {len(param_combinations)} 种参数组合")
    
    # 记录最佳结果
    best_val_loss = float('inf')
    best_params = None
    
    # 使用进度管理器
    with ProgressManager(len(param_combinations), desc=f"{element} 超参数调优") as progress:
        # 遍历所有参数组合
        for i, params in enumerate(param_combinations):
            logger.info(f"参数组合 {i+1}/{len(param_combinations)}: {params}")
            
            # 为当前超参数添加固定参数
            current_params = {
                **params,
                'num_epochs': min(config.training_config['num_epochs'], 50),  # 调优时使用较少的时代数
                'patience': config.training_config['early_stopping_patience']
            }
            
            # 训练模型
            try:
                _, val_loss, _ = train_and_evaluate_model(
                    f"{element}_tune_{i}",
                    train_loader,
                    val_loader,
                    hyperparams=current_params,
                    device=device
                )
                
                # 更新最佳参数
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = current_params
                    logger.info(f"找到新的最佳参数，验证损失: {best_val_loss:.6f}")
                    
            except Exception as e:
                logger.error(f"参数组合 {params} 训练失败: {str(e)}")
            
            # 更新进度
            progress.update(1)
    
    logger.info(f"超参数调优完成")
    logger.info(f"最佳参数: {best_params}")
    logger.info(f"最佳验证损失: {best_val_loss:.6f}")
    
    return best_params

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

def process_element(element, config=None, tune_hyperparams=False):
    """
    处理单个元素的训练过程
    """
    try:
        from model import train, SpectralResCNN_GCN  # 导入需要的模块
        
        # 设置设备
        device = config.training_config['device'] if config else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {device}")
        
        # 加载数据
        if config and hasattr(config, 'data_paths'):
            train_data = load_data(config.data_paths['train_data'], element)
            val_data = load_data(config.data_paths['val_data'], element)
            test_data = load_data(config.data_paths['test_data'], element)
        else:
            # 使用默认路径
            train_data = load_data(os.path.join('processed_data', 'train_dataset.npz'), element)
            val_data = load_data(os.path.join('processed_data', 'val_dataset.npz'), element)
            test_data = load_data(os.path.join('processed_data', 'test_dataset.npz'), element)
    
        if train_data[0] is None or train_data[1] is None:
            logger.error("加载训练数据失败")
            return None, None
            
        if val_data[0] is None:
            logger.warning("加载验证数据失败，使用训练数据代替")
            val_data = train_data
            
        if test_data[0] is None:
            logger.warning("加载测试数据失败，使用验证数据代替")
            test_data = val_data
        
        # 创建数据加载器
        train_loader = create_data_loaders(train_data[0], train_data[1], batch_size=config.training_config['batch_size'])
        val_loader = create_data_loaders(val_data[0], val_data[1], batch_size=config.training_config['batch_size'])
        test_loader = create_data_loaders(test_data[0], test_data[1], batch_size=config.training_config['batch_size'])
        
        # 获取输入大小
        input_size = config.model_config.get('input_size')
        # 如果输入大小为None，且有数据，则从数据中获取
        if input_size is None and train_data[0] is not None:
            if len(train_data[0].shape) > 1:  # 确保数据不是空的
                input_size = train_data[0].shape[1]  # 获取特征维度
                logger.info(f"从训练数据自动确定输入大小: {input_size}")
        
        # 创建模型
        model = SpectralResCNN_GCN(input_size=input_size, device=device)
        logger.info(f"创建模型: SpectralResCNN_GCN, 输入大小: {input_size}")
        
        # 尝试恢复训练
        model_dir = config.model_config['model_dir'] if config else 'models'
        model_path = os.path.join(model_dir, f"{element}_model.pth")

        if config.training_config.get('resume_training', False) and os.path.exists(model_path):
            logger.info(f"尝试从 {model_path} 恢复训练")
            try:
                from model import load_trained_model
                model = load_trained_model(model_path, device)
                logger.info(f"成功恢复模型，继续训练")
            except Exception as e:
                logger.warning(f"恢复训练失败: {str(e)}，将重新开始训练")
        
        # 是否进行超参数调优
        if tune_hyperparams:
            from hyperopt_tuning import hyperparameter_tuning
            logger.info(f"开始 {element} 的超参数调优")
            best_params, best_model = hyperparameter_tuning(
                element,
                train_loader,
                val_loader,
                device=device,
                config=config
            )
            logger.info(f"{element} 超参数调优完成，最佳参数: {best_params}")
            return best_model, best_params
        
        # 正常训练流程
        else:
            from model import train_and_evaluate_model
            logger.info(f"开始 {element} 的训练和评估")
            best_model, best_val_loss, test_metrics = train_and_evaluate_model(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                element=element,
                config=config
            )
            
            logger.info(f"{element} 训练和评估完成，最佳验证损失: {best_val_loss:.6f}")
            logger.info(f"测试指标: {test_metrics}")
            
            return best_model, test_metrics
            
    except Exception as e:
        logger.error(f"处理元素 {element} 时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

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
            best_model, test_metrics = process_element(element, config, tune_hyperparams=tune_hyperparams)
            
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

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='恒星光谱元素丰度预测系统')
    parser.add_argument('--mode', type=str, choices=['train', 'tune', 'test', 'predict', 'all', 'show_results', 'analyze', 'preprocess'], 
                        default='train', help='运行模式')
    parser.add_argument('--data_path', type=str, action='append', default=[],
                       help='数据路径，可以指定多个。例如：--data_path train.npz --data_path val.npz')
    parser.add_argument('--train_data_path', type=str, default=None, 
                       help='训练数据路径')
    parser.add_argument('--val_data_path', type=str, default=None,
                       help='验证数据路径')
    parser.add_argument('--test_data_path', type=str, default=None,
                       help='测试数据路径')
    parser.add_argument('--elements', nargs='+', default=None,
                       help='要处理的元素列表，默认为所有配置的元素')
    parser.add_argument('--element', type=str, default=None,
                       help='要处理的单个元素，与--elements互斥')
    parser.add_argument('--batch_size', type=int, default=None,
                      help='批次大小，默认使用配置文件中的设置')
    parser.add_argument('--learning_rate', type=float, default=None,
                      help='学习率，默认使用配置文件中的设置')
    parser.add_argument('--epochs', type=int, default=None,
                      help='训练轮数，默认使用配置文件中的设置')
    parser.add_argument('--early_stopping', type=int, default=None,
                      help='早停耐心值，默认使用配置文件中的设置')
    parser.add_argument('--device', type=str, default=None,
                      help='计算设备，默认使用配置文件中的设置')
    parser.add_argument('--batch_size_hyperopt', type=int, default=1000,
                       help='超参数优化的批量数据大小')
    parser.add_argument('--batches_per_round', type=int, default=2,
                       help='超参数优化每轮处理的批次数')
    parser.add_argument('--tune_hyperparams', action='store_true',
                       help='是否进行超参数调优')
    parser.add_argument('--result_type', type=str, choices=['training', 'evaluation', 'prediction', 'analysis'], 
                        default='training', help='要显示的结果类型')
    parser.add_argument('--perform_analysis', action='store_true',
                       help='是否执行模型性能分析（特征重要性和残差分析）')
    parser.add_argument('--analysis_type', type=str, choices=['feature_importance', 'residual_analysis', 'both'], 
                        default='both', help='要执行的分析类型')
    parser.add_argument('--analysis_batch_size', type=int, default=None,
                       help='模型分析批处理大小')
    parser.add_argument('--save_batch_results', action='store_true',
                       help='是否保存批处理结果')
    # 添加preprocessdata7相关参数
    parser.add_argument('--use_preprocessor', action='store_true',
                       help='使用preprocessdata7.py作为预处理器')
    parser.add_argument('--csv_files', nargs='+', default=['C_FE.csv', 'MG_FE.csv', 'CA_FE.csv'],
                      help='preprocessdata7使用的CSV文件列表')
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
    
    args = parser.parse_args()
    
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
    
    if args.mode in ['train', 'tune', 'test', 'all']:
        # 处理数据路径
        data_paths = args.data_path
        
        # 优先使用专门的路径参数
        if args.train_data_path:
            train_path = args.train_data_path
        elif data_paths and len(data_paths) > 0:
            train_path = data_paths[0]
        else:
            train_path = os.path.join('processed_data', 'train_dataset.npz')
            
        if args.val_data_path:
            val_path = args.val_data_path
        elif data_paths and len(data_paths) > 1:
            val_path = data_paths[1]
        else:
            val_path = os.path.join('processed_data', 'val_dataset.npz')
            
        if args.test_data_path:
            test_path = args.test_data_path
        elif data_paths and len(data_paths) > 2:
            test_path = data_paths[2]
        else:
            test_path = os.path.join('processed_data', 'test_dataset.npz')
        
        # 加载训练数据
        logger.info(f"加载训练数据: {train_path}")
        train_data = load_data(train_path, elements[0] if len(elements) == 1 else elements)
        
        if train_data[0] is None or train_data[1] is None:
            logger.error("加载训练数据失败，退出程序")
            return
        
        # 加载验证数据
        logger.info(f"加载验证数据: {val_path}")
        val_data = load_data(val_path, elements[0] if len(elements) == 1 else elements)
        if val_data[0] is None:
            logger.warning("加载验证数据失败，使用训练数据代替")
            val_data = train_data
        
        # 加载测试数据
        logger.info(f"加载测试数据: {test_path}")
        test_data = load_data(test_path, elements[0] if len(elements) == 1 else elements)
        if test_data[0] is None:
            logger.warning("加载测试数据失败，使用验证数据代替")
            test_data = val_data
        
        # 创建数据加载器
        train_loader = create_data_loaders(
            train_data[0], train_data[1], 
            batch_size=config.training_config['batch_size'],
            shuffle=True
        )
        
        val_loader = create_data_loaders(
            val_data[0], val_data[1], 
            batch_size=config.training_config['batch_size'],
            shuffle=False
        )
        
        test_loader = create_data_loaders(
            test_data[0], test_data[1], 
            batch_size=config.training_config['batch_size'],
            shuffle=False
        )
    
    if args.mode == 'train' or args.mode == 'all':
        # 训练模型
        for i, element in enumerate(elements):
            logger.info(f"训练 {element} 元素丰度预测模型")
            
            # 检查是否需要从元素索引中提取特定元素的标签
            element_indices = train_data[2]  # 从load_data返回的结果中获取
            if element_indices is not None:
                if isinstance(element_indices, dict) and element in element_indices:
                # 从标签中提取特定元素的值
                    element_idx = element_indices[element]
        
                    # 修改标签提取逻辑，处理1D和2D数组
                    if len(train_data[1].shape) == 1:
                    # 如果是1D数组，直接使用原始标签
                        element_label = train_data[1]
                        val_element_label = val_data[1]
                        test_element_label = test_data[1]
                    else:
                        # 如果是2D数组，则使用索引
                        element_label = train_data[1][:, element_idx]
                        val_element_label = val_data[1][:, element_idx]
                        test_element_label = test_data[1][:, element_idx]
                    
                    # 创建特定元素的数据加载器
                        train_loader_element = create_data_loaders(train_data[0], element_label,
                                                            batch_size=config.training_config['batch_size'])
                        val_loader_element = create_data_loaders(val_data[0], val_element_label,
                                                          batch_size=config.training_config['batch_size'], shuffle=False)
                        test_loader_element = create_data_loaders(test_data[0], test_element_label,
                                                           batch_size=config.training_config['batch_size'], shuffle=False)
                    
                        logger.info(f"为元素 {element} 创建特定数据加载器")
                else:
                    # 使用原始数据加载器
                    train_loader_element = train_loader
                    val_loader_element = val_loader
                    test_loader_element = test_loader
                    logger.warning(f"无法找到元素 {element} 的索引，使用原始数据加载器")
            else:
                # 没有元素索引信息，使用原始数据加载器
                train_loader_element = train_loader
                val_loader_element = val_loader
                test_loader_element = test_loader
            
            # 训练和评估模型
            try:
                # 根据命令行参数决定是否进行超参数调优
                tune_hyperparams = args.tune_hyperparams or args.mode == 'tune'
                process_element(element, config, tune_hyperparams=tune_hyperparams)
            except Exception as e:
                logger.error(f"训练元素 {element} 时出错: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
    
    if args.mode == 'tune' or args.mode == 'all':
        # 超参数调优
        for i, element in enumerate(elements):
            logger.info(f"为 {element} 元素进行超参数调优")
            # 实现超参数调优逻辑
    
    if args.mode == 'test' or args.mode == 'all':
        # 测试模型
        for i, element in enumerate(elements):
            logger.info(f"测试 {element} 元素丰度预测模型")
            # 实现测试逻辑
    
    if args.mode == 'predict':
        # 预测模式
        if args.use_preprocessor:
            # 使用preprocessdata7进行预测
            if len(args.data_path) == 0:
                logger.error("预测模式需要指定输入文件路径")
                return
                
            for element in elements:
                logger.info(f"使用preprocessdata7预测元素 {element}")
                result = use_preprocessor(
                    task='predict',
                    element=element,
                    input_file=args.data_path[0],
                    **preprocessor_kwargs
                )
                
                if result['success']:
                    logger.info(f"预测完成: {result['result']}")
                else:
                    logger.error(f"预测失败: {result.get('error', '未知错误')}")
        else:
            # 使用原系统进行预测
            data_path = args.data_path[0] if args.data_path else config.data_paths.get('reference_data')
            logger.info(f"加载预测数据: {data_path}")
        spectra, _, _ = load_data(data_path)
        
        if spectra is None:
            logger.error("加载数据失败，退出程序")
            return
        
        # 使用训练好的模型进行预测
        for element in elements:
            logger.info(f"使用 {element} 模型进行预测")

if __name__ == '__main__':
    main() 