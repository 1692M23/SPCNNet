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

# 导入自定义模块
import config
from model import SpectralResCNN, SpectralResCNNEnsemble, train_model, evaluate_model, load_trained_model
from evaluation import evaluate_all_elements, plot_predictions_vs_true, plot_metrics_comparison
from utils import CacheManager, ProgressManager, ask_clear_cache
from multi_element_processor import MultiElementProcessor
from fits_cache import FITSCache

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
    """加载数据，支持从缓存加载FITS信息"""
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    
    if 'obsid' in data:
        obsids = data['obsid']
        # 如果需要额外的FITS信息
        fits_cache = FITSCache(os.path.join('processed_data', 'fits_cache'))
        additional_data = []
        for obsid in obsids:
            cached_data = fits_cache.get_fits_data(obsid)
            if cached_data is not None:
                additional_data.append(cached_data)
    
    return X, y, additional_data if 'obsid' in data else None

def create_data_loaders(spectra, labels, batch_size=32, shuffle=True):
    """
    创建PyTorch数据加载器
    
    参数:
        spectra (numpy.ndarray): 光谱数据，形状为 [n_samples, n_wavelengths] 或 [n_samples, channels, n_wavelengths]
        labels (numpy.ndarray): 标签数据，形状为 [n_samples]
        batch_size (int): 批次大小
        shuffle (bool): 是否打乱数据
        
    返回:
        torch.utils.data.DataLoader: 数据加载器
    """
    logger.info(f"输入数据形状: {spectra.shape}")
    
    # 检查数据形状，如果还没有通道维度，则添加
    if len(spectra.shape) == 2:
        # 添加通道维度 [n_samples, 1, n_wavelengths]
        spectra_tensor = torch.FloatTensor(spectra).unsqueeze(1)
        logger.info(f"2D数据添加通道维度后形状: {spectra_tensor.shape}")
    elif len(spectra.shape) == 4:
        # 如果是4D数据 [n_samples, 1, 1, n_wavelengths]，去掉多余的维度
        spectra_tensor = torch.FloatTensor(spectra).squeeze(2)
        logger.info(f"4D数据压缩后形状: {spectra_tensor.shape}")
    else:
        # 已经有正确的通道维度，直接转换为张量
        spectra_tensor = torch.FloatTensor(spectra)
        logger.info(f"保持原有形状: {spectra_tensor.shape}")
    
    # 确保数据维度正确 [batch_size, channels, length]
    if len(spectra_tensor.shape) != 3:
        logger.error(f"数据维度不正确: {spectra_tensor.shape}")
        raise ValueError(f"数据维度不正确: {spectra_tensor.shape}")
    
    labels_tensor = torch.FloatTensor(labels)
    
    # 创建数据集
    dataset = TensorDataset(spectra_tensor, labels_tensor)
    
    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True
    )
    
    # 检查第一个批次的数据形状
    for batch in data_loader:
        spectra_batch, _ = batch
        logger.info(f"批次数据形状: {spectra_batch.shape}")
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
        'epochs': config.training_config['epochs'],
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
    plt.plot(train_metrics['loss'], label='训练损失')
    plt.plot(val_metrics['loss'], label='验证损失')
    plt.title(f'{element} 训练过程')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, f'{element}_training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制MAE曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics['mae'], label='训练MAE')
    plt.plot(val_metrics['mae'], label='验证MAE')
    plt.title(f'{element} 训练过程')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, f'{element}_training_mae.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"已保存训练过程可视化图表")

def process_element(element, config=None):
    """处理单个元素的训练和评估"""
    if config is None:
        config = config.CONFIG
        
    # 加载数据集
    train_data = load_data(os.path.join('processed_data', 'train_dataset.npz'), element)
    val_data = load_data(os.path.join('processed_data', 'val_dataset.npz'), element)
    test_data = load_data(os.path.join('processed_data', 'test_dataset.npz'), element)
    
    if train_data[0] is None or val_data[0] is None:
        logger.error(f"加载{element}的数据集失败")
        return None
        
    # 创建数据加载器
    train_loader = create_data_loaders(train_data[0], train_data[1], 
                                     batch_size=config['training']['batch_size'])
    val_loader = create_data_loaders(val_data[0], val_data[1], 
                                   batch_size=config['training']['batch_size'])
    test_loader = None
    if test_data[0] is not None:
        test_loader = create_data_loaders(test_data[0], test_data[1], 
                                        batch_size=config['training']['batch_size'])
    
    # 超参数调优（如果启用）
    hyperparams = None
    if config['training']['tune_hyperparams']:
        logger.info(f"开始 {element} 的超参数调优")
        # 使用新的两阶段调优方法
        hyperparams = hyperparameter_tuning(element, train_loader, val_loader, device=config['training']['device'])
        
        if hyperparams:
            logger.info(f"{element} 的最佳超参数: {hyperparams}")
            
            # 更新训练配置，应用优化后的超参数
            training_config = config['training'].copy()
            
            # 应用关键超参数
            if 'lr' in hyperparams:
                training_config['lr'] = hyperparams['lr']
            if 'weight_decay' in hyperparams:
                training_config['weight_decay'] = hyperparams['weight_decay']
            if 'batch_size' in hyperparams:
                # 更新数据加载器的batch_size
                batch_size = hyperparams['batch_size']
                train_loader = create_data_loaders(train_data[0], train_data[1], batch_size=batch_size)
                val_loader = create_data_loaders(val_data[0], val_data[1], batch_size=batch_size)
                if test_data[0] is not None:
                    test_loader = create_data_loaders(test_data[0], test_data[1], batch_size=batch_size)
                training_config['batch_size'] = batch_size
            if 'patience' in hyperparams:
                training_config['early_stopping_patience'] = hyperparams['patience']
                
            # 更新配置
            config['training'] = training_config
    
    # 使用最佳超参数训练模型
    logger.info(f"开始训练 {element} 模型" + 
               (f" (使用优化超参数)" if hyperparams else ""))
    
    # 如果有dropout_rate超参数，传递给模型
    dropout_rate = hyperparams.get('dropout_rate', 0.5) if hyperparams else 0.5
    
    # 创建模型，将dropout_rate传递给模型构造函数
    model = SpectralResCNN(
        input_size=config['model_config']['input_size'],
        dropout_rate=dropout_rate
    ).to(config['training']['device'])
    
    # 训练模型
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        element=element,
        config=config
    )
    
    # 评估模型（如果有测试集）
    test_metrics = None
    if test_loader:
        test_metrics = evaluate_model(model, test_loader, config['training']['device'])
        logger.info(f"{element} 测试集评估结果: {test_metrics}")
    
    # 保存到缓存
    cache_key = f"model_{element}"
    cache_manager.set_cache(cache_key, {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'metrics': test_metrics,
        'hyperparams': hyperparams
    })
    
    return model, test_metrics

def process_multiple_elements(csv_file, fits_dir, element_columns=None, 
                             test_size=0.2, val_size=0.1, batch_size=32, 
                             tune_hyperparams=False, device=None, batch_size_hyperopt=1000, batches_per_round=2):
    """
    处理多个元素丰度的完整流程：数据加载、训练、评估
    
    参数:
        csv_file (str): 包含obsid和元素丰度的CSV文件
        fits_dir (str): FITS文件目录
        element_columns (list): 元素丰度列名列表，如['C_FE', 'MG_FE', 'CA_FE']
        test_size (float): 测试集比例
        val_size (float): 验证集比例
        batch_size (int): 批次大小
        tune_hyperparams (bool): 是否进行超参数调优
        device (str): 计算设备
        batch_size_hyperopt (int): 批量超参数优化批量大小
        batches_per_round (int): 批量超参数优化每轮处理的批次数
        
    返回:
        dict: 包含各元素模型和评估指标的字典
    """
    # 检查缓存
    cache_key = f"multi_elements_{os.path.basename(csv_file)}"
    cached_data = cache_manager.get_cache(cache_key)
    if cached_data is not None:
        logger.info(f"从缓存加载多元素处理结果")
        return cached_data
    
    if device is None:
        device = config.training_config['device']
    
    # 初始化数据处理器
    processor = MultiElementProcessor(fits_dir=fits_dir)
    
    # 准备数据集
    logger.info(f"开始处理CSV文件 {csv_file} 中的多元素丰度数据")
    datasets = processor.prepare_datasets(
        csv_file, 
        element_columns=element_columns,
        test_size=test_size,
        val_size=val_size
    )
    
    # 处理每个元素
    results = {}
    with ProgressManager(len(datasets), desc="处理元素丰度") as progress:
        for element, dataset in datasets.items():
            logger.info(f"开始处理元素: {element}")
            
            # 创建数据加载器
            train_loader = create_data_loaders(
                dataset['train'][0], dataset['train'][1], 
                batch_size=batch_size
            )
            val_loader = create_data_loaders(
                dataset['val'][0], dataset['val'][1], 
                batch_size=batch_size, shuffle=False
            )
            test_loader = create_data_loaders(
                dataset['test'][0], dataset['test'][1], 
                batch_size=batch_size, shuffle=False
            )
            
            # 超参数调优（如果启用）
            hyperparams = None
            if tune_hyperparams:
                logger.info(f"开始 {element} 的超参数调优")
                hyperparams = hyperparameter_tuning(
                    element, train_loader, val_loader, device=device
                )
            
            # 训练集成模型
            logger.info(f"开始训练 {element} 模型")
            ensemble = SpectralResCNNEnsemble(
                input_size=config.model_config['input_size'],
                num_models=config.model_config.get('num_ensemble_models', 3),
                dropout_rate=config.model_config.get('dropout_rate', 0.5)
            )
            
            # 训练模型
            train_results = ensemble.train(train_loader, val_loader, element, config)
            
            # 评估模型
            logger.info(f"开始评估 {element} 模型")
            test_metrics = {}
            for i, model in enumerate(ensemble.models):
                model_metrics = evaluate_model(
                    model, test_loader, device=device
                )
                test_metrics[f"model_{i+1}"] = model_metrics
            
            # 评估集成模型
            ensemble_predictions = []
            ensemble_targets = []
            
            for spectra, targets in test_loader:
                spectra = spectra.to(device)
                predictions, _ = ensemble.predict(spectra)
                ensemble_predictions.extend(predictions)
                ensemble_targets.extend(targets.cpu().numpy())
            
            # 计算集成模型指标
            ensemble_predictions = np.array(ensemble_predictions)
            ensemble_targets = np.array(ensemble_targets)
            
            ensemble_metrics = {
                'mae': np.mean(np.abs(ensemble_predictions - ensemble_targets)),
                'mse': np.mean((ensemble_predictions - ensemble_targets) ** 2),
                'rmse': np.sqrt(np.mean((ensemble_predictions - ensemble_targets) ** 2)),
                'r2': 1 - (np.sum((ensemble_targets - ensemble_predictions) ** 2) / 
                          np.sum((ensemble_targets - np.mean(ensemble_targets)) ** 2)),
                'bias': np.mean(ensemble_predictions - ensemble_targets),
                'scatter': np.std(ensemble_predictions - ensemble_targets),
                'predictions': ensemble_predictions,
                'targets': ensemble_targets
            }
            
            test_metrics['ensemble'] = ensemble_metrics
            
            # 保存结果
            results[element] = {
                'ensemble': ensemble,
                'train_results': train_results,
                'test_metrics': test_metrics
            }
            
            # 保存预测结果
            pred_df = pd.DataFrame({
                'true': ensemble_targets,
                'predicted': ensemble_predictions,
                'error': ensemble_predictions - ensemble_targets
            })
            
            os.makedirs(config.output_config['predictions_dir'], exist_ok=True)
            pred_file = os.path.join(
                config.output_config['predictions_dir'], 
                f"{element}_predictions.csv"
            )
            pred_df.to_csv(pred_file, index=False)
            logger.info(f"已保存 {element} 预测结果到: {pred_file}")
            
            # 更新进度
            progress.update(1)
    
    # 保存到缓存
    cache_manager.set_cache(cache_key, results)
    
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

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='恒星光谱元素丰度预测系统')
    parser.add_argument('--mode', type=str, choices=['train', 'tune', 'test', 'predict', 'all', 'show_results'], 
                        default='train', help='运行模式')
    parser.add_argument('--data_path', type=str, default=None,
                       help='数据路径，默认使用配置文件中的设置')
    parser.add_argument('--elements', nargs='+', default=None,
                       help='要处理的元素列表，默认为所有配置的元素')
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
    parser.add_argument('--result_type', type=str, choices=['training', 'evaluation', 'prediction'], 
                        default='training', help='要显示的结果类型')
    
    args = parser.parse_args()
    
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
    
    # 确定要处理的元素
    elements = args.elements if args.elements else config.training_config['elements']
    
    # 创建必要的目录
    setup_training_directories()
    
    # 根据模式执行不同操作
    if args.mode == 'show_results':
        # 显示批次结果
        if args.elements is None or len(args.elements) == 0:
            logger.error("显示结果时必须指定元素名称")
            return
        
        for element in elements:
            show_batch_results(element, args.result_type, config)
        return
    
    if args.mode in ['train', 'tune', 'test', 'all']:
        # 加载数据
        data_path = args.data_path if args.data_path else config.data_paths['train_data']
        spectra, labels, element_indices = load_data(data_path, elements)
        
        if spectra is None or labels is None:
            logger.error("加载数据失败，退出程序")
            return
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_data_loaders(
            spectra, labels, 
            batch_size=config.training_config['batch_size'],
            shuffle=True
        )
    
    if args.mode == 'train' or args.mode == 'all':
        # 训练模型
        for i, element in enumerate(elements):
            logger.info(f"训练 {element} 元素丰度预测模型")
            element_label = labels[:, element_indices[i]] if element_indices else None
            train_loader_element = DataLoader(
                TensorDataset(spectra, element_label),
                batch_size=config.training_config['batch_size'],
                shuffle=True
            )
            val_loader_element = DataLoader(
                TensorDataset(spectra, element_label),
                batch_size=config.training_config['batch_size'],
                shuffle=False
            )
            train_and_evaluate_model(train_loader_element, val_loader_element, element, CONFIG)
    
    if args.mode == 'tune' or args.mode == 'all':
        # 超参数调优
        for i, element in enumerate(elements):
            logger.info(f"为 {element} 元素进行超参数调优")
            element_label = labels[:, element_indices[i]] if element_indices else None
            train_loader_element = DataLoader(
                TensorDataset(spectra, element_label),
                batch_size=config.training_config['batch_size'],
                shuffle=True
            )
            val_loader_element = DataLoader(
                TensorDataset(spectra, element_label),
                batch_size=config.training_config['batch_size'],
                shuffle=False
            )
            hyperparameter_tuning(
                element, 
                train_loader_element, 
                val_loader_element, 
                device=config.training_config['device'],
                batch_size_hyperopt=args.batch_size_hyperopt,
                batches_per_round=args.batches_per_round
            )
    
    if args.mode == 'test' or args.mode == 'all':
        # 测试模型
        for i, element in enumerate(elements):
            logger.info(f"测试 {element} 元素丰度预测模型")
            # 实现测试逻辑
    
    if args.mode == 'predict':
        # 预测模式
        data_path = args.data_path if args.data_path else config.data_paths['reference_data']
        spectra, _, _ = load_data(data_path)
        
        if spectra is None:
            logger.error("加载数据失败，退出程序")
            return
        
        # 使用训练好的模型进行预测
        for element in elements:
            logger.info(f"使用 {element} 模型进行预测")

if __name__ == '__main__':
    main() 