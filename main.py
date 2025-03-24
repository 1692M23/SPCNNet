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

# 导入自定义模块
import config
from model import SpectralResCNN, train_model, evaluate_model, load_trained_model
from evaluation import evaluate_all_elements, plot_predictions_vs_true, plot_metrics_comparison

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

def load_dataset(data_path, element):
    """
    加载数据集
    
    参数:
        data_path (str): 数据文件路径
        element (str): 元素名称
        
    返回:
        tuple: (spectra, labels) 光谱数据和标签
    """
    try:
        data = np.load(data_path)
        
        # 检查'X'字段（光谱数据）是否存在
        if 'X' not in data:
            logger.error(f"数据文件缺少'X'字段")
            return None, None
        
        # 获取光谱数据
        spectra = data['X']
        
        # 检查'y'字段是否存在
        if 'y' not in data:
            logger.error(f"数据文件缺少'y'字段")
            return None, None
            
        # 获取标签数据
        labels = data['y']
        
        # 检查元素是否匹配
        if 'elements' in data:
            # 获取第一个元素名称（因为所有元素名称都是相同的）
            dataset_element = str(data['elements'][0])
            if dataset_element != element:
                logger.error(f"数据集元素 {dataset_element} 与请求元素 {element} 不匹配")
                return None, None
            else:
                logger.info(f"找到元素 {element} 的数据")
        
        logger.info(f"已加载数据集: {data_path}, 光谱数量: {len(spectra)}")
        return spectra, labels
        
    except Exception as e:
        logger.error(f"加载数据集失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

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
    # 检查数据形状，如果还没有通道维度，则添加
    if len(spectra.shape) == 2:
        # 添加通道维度 [n_samples, 1, n_wavelengths]
        spectra_tensor = torch.FloatTensor(spectra).unsqueeze(1)
    elif len(spectra.shape) == 4:
        # 如果是4D数据 [n_samples, 1, 1, n_wavelengths]，去掉多余的维度
        spectra_tensor = torch.FloatTensor(spectra).squeeze(2)
    else:
        # 已经有正确的通道维度，直接转换为张量
        spectra_tensor = torch.FloatTensor(spectra)
    
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
    
    return data_loader

def setup_training_directories():
    """
    创建训练所需的目录
    """
    os.makedirs(config.model_config['model_dir'], exist_ok=True)
    os.makedirs(config.output_config['log_dir'], exist_ok=True)
    os.makedirs(config.output_config['results_dir'], exist_ok=True)
    os.makedirs(config.output_config['plots_dir'], exist_ok=True)
    os.makedirs(config.output_config['predictions_dir'], exist_ok=True)
    
    logger.info(f"已创建输出目录")

def train_and_evaluate_model(element, train_loader, val_loader, test_loader=None, 
                         hyperparams=None, device=None):
    """
    训练并评估模型
    
    参数:
        element (str): 元素名称
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        test_loader (DataLoader): 测试数据加载器
        hyperparams (dict): 超参数
        device (str): 计算设备
        
    返回:
        tuple: (model, best_val_loss, test_metrics) 模型，最佳验证损失和测试指标
    """
    if device is None:
        device = config.training_config['device']
    
    # 如果没有提供超参数，使用默认配置
    if hyperparams is None:
        hyperparams = {
            'lr': config.training_config['lr'],
            'weight_decay': config.training_config['weight_decay'],
            'epochs': config.training_config['num_epochs'],
            'patience': config.training_config['early_stopping_patience']
        }
    
    # 创建模型
    model = SpectralResCNN(
        input_size=config.model_config['input_dim']
    ).to(device)
    
    # 记录超参数
    logger.info(f"训练 {element} 模型，超参数:")
    for param, value in hyperparams.items():
        logger.info(f"  {param}: {value}")
    
    # 训练模型
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        element=element,
        config={
            'training': {
                'learning_rate': hyperparams['lr'],
                'weight_decay': hyperparams['weight_decay'],
                'epochs': hyperparams['epochs'],
                'early_stopping_patience': hyperparams['patience'],
                'device': device
            },
            'output': {
                'model_dir': config.model_config['model_dir']
            }
        }
    )
    
    # 获取最佳验证损失
    best_val_loss = min(val_losses)
    
    # 保存模型
    model_path = os.path.join(config.model_config['model_dir'], f"{element}_model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"模型已保存到: {model_path}")
    
    # 如果提供了测试数据加载器，进行测试评估
    test_metrics = None
    if test_loader is not None:
        test_metrics = evaluate_model(model, test_loader, device)
        logger.info(f"{element} 测试集评估结果:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.6f}")
    
    return model, best_val_loss, test_metrics

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

def process_element(element, train_data_path, val_data_path, test_data_path=None,
                   tune_hyperparams=False, batch_size=32, device=None):
    """
    处理单个元素的完整流程：数据加载、超参数调优、训练、评估
    
    参数:
        element (str): 元素名称
        train_data_path (str): 训练数据路径
        val_data_path (str): 验证数据路径
        test_data_path (str): 测试数据路径
        tune_hyperparams (bool): 是否进行超参数调优
        batch_size (int): 批次大小
        device (str): 计算设备
        
    返回:
        tuple: (model, test_metrics) 模型和测试指标
    """
    # 加载训练数据
    train_spectra, train_labels = load_dataset(train_data_path, element)
    if train_spectra is None or train_labels is None:
        logger.error(f"加载训练数据失败，退出处理 {element}")
        return None, None
    
    # 加载验证数据
    val_spectra, val_labels = load_dataset(val_data_path, element)
    if val_spectra is None or val_labels is None:
        logger.error(f"加载验证数据失败，退出处理 {element}")
        return None, None
    
    # 创建数据加载器
    train_loader = create_data_loaders(train_spectra, train_labels, batch_size)
    val_loader = create_data_loaders(val_spectra, val_labels, batch_size, shuffle=False)
    
    # 加载测试数据（如果提供）
    test_loader = None
    if test_data_path:
        test_spectra, test_labels = load_dataset(test_data_path, element)
        if test_spectra is not None and test_labels is not None:
            test_loader = create_data_loaders(test_spectra, test_labels, batch_size, shuffle=False)
    
    # 超参数调优（如果启用）
    hyperparams = None
    if tune_hyperparams:
        logger.info(f"开始 {element} 的超参数调优")
        hyperparams = hyperparameter_tuning(element, train_loader, val_loader, device=device)
    
    # 使用最佳超参数训练模型
    logger.info(f"开始训练 {element} 模型")
    model, _, test_metrics = train_and_evaluate_model(
        element=element,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        hyperparams=hyperparams,
        device=device
    )
    
    return model, test_metrics

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='恒星光谱元素丰度预测模型训练与评估')
    parser.add_argument('--elements', nargs='+', default=None,
                        help='要处理的元素列表，默认为所有配置的元素')
    parser.add_argument('--tune', action='store_true',
                        help='是否进行超参数调优')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--no_test', action='store_true',
                        help='跳过测试评估')
    parser.add_argument('--visualize', action='store_true',
                        help='是否生成可视化')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用CPU进行计算')
    
    args = parser.parse_args()
    
    # 确定要处理的元素
    elements = args.elements if args.elements else config.training_config['elements']
    logger.info(f"将处理以下元素: {', '.join(elements)}")
    
    # 设置设备
    if args.cpu:
        device = 'cpu'
    else:
        device = config.training_config['device']
    logger.info(f"使用设备: {device}")
    
    # 创建必要的目录
    setup_training_directories()
    
    # 获取数据路径
    train_data_path = config.data_paths['train_data']
    val_data_path = config.data_paths['val_data']
    test_data_path = None if args.no_test else config.data_paths['test_data']
    
    # 处理每个元素
    for element in elements:
        logger.info(f"开始处理元素: {element}")
        
        # 运行完整处理流程
        model, metrics = process_element(
            element=element,
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            test_data_path=test_data_path,
            tune_hyperparams=args.tune,
            batch_size=args.batch_size,
            device=device
        )
        
        if model is None:
            logger.error(f"处理 {element} 失败，跳过")
            continue
    
    # 如果需要进行可视化评估
    if args.visualize and not args.no_test:
        logger.info("生成评估可视化")
        
        # 评估所有元素模型
        evaluate_all_elements(elements)
        
        # 生成预测对比图
        plot_predictions_vs_true(elements)
        
        # 生成评估指标比较图
        plot_metrics_comparison(elements)
    
    logger.info("处理完成")

if __name__ == '__main__':
    main() 