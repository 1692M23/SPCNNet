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
        hyperparams = hyperparameter_tuning(element, train_loader, val_loader, device=config['training']['device'])
    
    # 使用最佳超参数训练模型
    logger.info(f"开始训练 {element} 模型")
    model, _, test_metrics = train_and_evaluate_model(
        element=element,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config
    )
    
    # 保存到缓存
    cache_key = f"model_{element}"
    cache_manager.set_cache(cache_key, {
        'model': model,
        'metrics': test_metrics
    })
    
    return model, test_metrics

def process_multiple_elements(csv_file, fits_dir, element_columns=None, 
                             test_size=0.2, val_size=0.1, batch_size=32, 
                             tune_hyperparams=False, device=None):
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

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='恒星光谱元素丰度预测模型训练与评估')
    parser.add_argument('--csv_file', type=str, required=True,
                       help='输入CSV文件，包含obsid和元素丰度')
    parser.add_argument('--fits_dir', type=str, default='fits',
                       help='FITS文件目录')
    parser.add_argument('--elements', nargs='+', default=None,
                       help='要处理的元素列表，格式如C_FE MG_FE CA_FE，默认处理CSV中所有元素丰度列')
    parser.add_argument('--tune', action='store_true',
                       help='是否进行超参数调优')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--visualize', action='store_true',
                       help='是否生成可视化')
    parser.add_argument('--cpu', action='store_true',
                       help='强制使用CPU进行计算')
    parser.add_argument('--clear_cache', action='store_true',
                       help='清除所有缓存')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.cpu:
        device = 'cpu'
    else:
        device = config.training_config['device']
    logger.info(f"使用设备: {device}")
    
    # 创建必要的目录
    setup_training_directories()
    
    # 处理缓存
    if args.clear_cache:
        cache_manager.clear_cache()
        logger.info("已清除所有缓存")
    else:
        ask_clear_cache(cache_manager)
    
    # 处理多元素丰度
    results = process_multiple_elements(
        csv_file=args.csv_file,
        fits_dir=args.fits_dir,
        element_columns=args.elements,
        test_size=args.test_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        tune_hyperparams=args.tune,
        device=device
    )
    
    # 如果需要进行可视化评估
    if args.visualize:
        logger.info("生成评估可视化")
        
        # 生成每个元素的评估可视化
        for element, element_results in results.items():
            metrics = element_results['test_metrics']['ensemble']
            
            # 绘制预测对比图
            plt.figure(figsize=(10, 8))
            plt.scatter(metrics['targets'], metrics['predictions'], alpha=0.6)
            plt.plot([min(metrics['targets']), max(metrics['targets'])], 
                    [min(metrics['targets']), max(metrics['targets'])], 'r--')
            plt.title(f'{element} 预测对比')
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            plt.grid(True, alpha=0.3)
            plt.annotate(f'MAE: {metrics["mae"]:.4f}\nRMSE: {metrics["rmse"]:.4f}\n'
                        f'散度: {metrics["scatter"]:.4f}',
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # 保存图像
            plot_dir = config.output_config['plots_dir']
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f'{element}_predictions.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 生成元素间比较图
        if len(results) > 1:
            # 收集各元素评估指标
            elements = list(results.keys())
            mae_values = [results[elem]['test_metrics']['ensemble']['mae'] for elem in elements]
            rmse_values = [results[elem]['test_metrics']['ensemble']['rmse'] for elem in elements]
            scatter_values = [results[elem]['test_metrics']['ensemble']['scatter'] for elem in elements]
            
            # 绘制比较图
            plt.figure(figsize=(12, 8))
            
            # 绘制MAE比较
            plt.subplot(311)
            plt.bar(elements, mae_values, color='skyblue')
            plt.title('各元素MAE比较')
            plt.ylabel('MAE')
            plt.grid(True, alpha=0.3)
            
            # 绘制RMSE比较
            plt.subplot(312)
            plt.bar(elements, rmse_values, color='salmon')
            plt.title('各元素RMSE比较')
            plt.ylabel('RMSE')
            plt.grid(True, alpha=0.3)
            
            # 绘制散度比较
            plt.subplot(313)
            plt.bar(elements, scatter_values, color='lightgreen')
            plt.title('各元素散度比较')
            plt.ylabel('散度')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'elements_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    logger.info("处理完成")

if __name__ == '__main__':
    main() 