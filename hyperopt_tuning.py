#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
超参数优化模块：使用hyperopt实现两阶段超参数优化
"""

import os
import logging
import numpy as np
import torch
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pickle
from tqdm import tqdm
import time

# 导入自定义模块
import config
from model import SpectralResCNN, train, evaluate_model

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.output_config['log_dir'], 'hyperopt.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('hyperopt')

def objective(params, element, train_loader, val_loader, device=None):
    """
    优化目标函数：根据超参数创建并训练模型，返回验证集上的损失
    
    参数:
        params (dict): 超参数字典
        element (str): 元素名称
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 计算设备
    
    返回:
        dict: 包含损失值和状态的字典
    """
    if device is None:
        device = config.training_config['device']
    
    try:
        logger.info(f"评估超参数: {params}")
        start_time = time.time()
        
        # 创建模型
        dropout_rate = params.get('dropout_rate', 0.5)
        model = SpectralResCNN(
            input_size=config.model_config['input_size'],
            dropout_rate=dropout_rate
        ).to(device)
        
        # 设置训练配置
        train_config = config.CONFIG.copy()
        train_config['training_config'] = {
            'device': device,
            'batch_size': int(params.get('batch_size', 32)),
            'num_epochs': int(params.get('num_epochs', 50)),  # 调优时使用较少的epochs
            'lr': float(params.get('lr', 0.001)),
            'weight_decay': float(params.get('weight_decay', 1e-4)),
            'early_stopping_patience': int(params.get('patience', 10))
        }
        
        # 训练模型
        train_losses, val_losses = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            element=f"{element}_tune",
            config=train_config
        )
        
        # 获取验证集最佳损失
        best_val_loss = min(val_losses) if val_losses else float('inf')
        
        elapsed_time = time.time() - start_time
        logger.info(f"超参数评估完成: 验证损失={best_val_loss:.6f}, 耗时={elapsed_time:.2f}秒")
        
        return {
            'loss': best_val_loss,  # hyperopt寻找最小值
            'status': STATUS_OK,
            'model': model,
            'val_losses': val_losses,
            'train_losses': train_losses,
            'elapsed_time': elapsed_time
        }
        
    except Exception as e:
        logger.error(f"超参数评估失败: {str(e)}")
        import traceback
        traceback.print_exc()
        # 返回一个很大的损失值，确保这个参数组合不会被选中
        return {
            'loss': float('inf'),
            'status': STATUS_OK,
            'error': str(e)
        }

def run_hyperopt_tuning(element, train_loader, val_loader, max_evals_stage1=30, max_evals_stage2=20, device=None):
    """
    运行两阶段超参数优化
    
    参数:
        element (str): 元素名称
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        max_evals_stage1 (int): 第一阶段最大评估次数
        max_evals_stage2 (int): 第二阶段最大评估次数
        device: 计算设备
        
    返回:
        dict: 最佳超参数字典
    """
    if device is None:
        device = config.training_config['device']
    
    # 创建结果目录
    results_dir = os.path.join(config.output_config['results_dir'], 'hyperopt')
    os.makedirs(results_dir, exist_ok=True)
    
    # 第一阶段：广泛搜索
    logger.info(f"开始第一阶段超参数优化 ({element})")
    space_stage1 = {
        'lr': hp.loguniform('lr', np.log(0.00001), np.log(0.01)),
        'weight_decay': hp.loguniform('weight_decay', np.log(1e-6), np.log(1e-3)),
        'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
        'dropout_rate': hp.uniform('dropout_rate', 0.2, 0.7),
        'patience': hp.choice('patience', [5, 10, 15, 20]),
        'num_epochs': 50,  # 固定epoch数量用于调参
    }
    
    trials_stage1 = Trials()
    best_stage1 = fmin(
        fn=lambda params: objective(params, element, train_loader, val_loader, device)['loss'],
        space=space_stage1,
        algo=tpe.suggest,
        max_evals=max_evals_stage1,
        trials=trials_stage1,
        show_progressbar=False
    )
    
    # 解析最佳参数
    best_params_stage1 = {
        'lr': float(np.exp(best_stage1['lr'])),
        'weight_decay': float(np.exp(best_stage1['weight_decay'])),
        'batch_size': [16, 32, 64, 128][best_stage1['batch_size']],
        'dropout_rate': float(best_stage1['dropout_rate']),
        'patience': [5, 10, 15, 20][best_stage1['patience']],
    }
    
    logger.info(f"第一阶段最佳参数 ({element}): {best_params_stage1}")
    
    # 保存第一阶段结果
    with open(os.path.join(results_dir, f'{element}_hyperopt_stage1.pkl'), 'wb') as f:
        pickle.dump({
            'best_params': best_params_stage1,
            'trials': trials_stage1
        }, f)
    
    # 第二阶段：精细调整
    logger.info(f"开始第二阶段超参数优化 ({element})")
    # 在第一阶段最佳参数周围搜索
    space_stage2 = {
        'lr': hp.loguniform('lr', 
                           np.log(best_params_stage1['lr'] * 0.5), 
                           np.log(best_params_stage1['lr'] * 2.0)),
        'weight_decay': hp.loguniform('weight_decay', 
                                     np.log(best_params_stage1['weight_decay'] * 0.5), 
                                     np.log(best_params_stage1['weight_decay'] * 2.0)),
        'batch_size': hp.choice('batch_size', [best_params_stage1['batch_size']]),  # 固定batch_size
        'dropout_rate': hp.uniform('dropout_rate', 
                                  max(0.1, best_params_stage1['dropout_rate'] - 0.1), 
                                  min(0.9, best_params_stage1['dropout_rate'] + 0.1)),
        'patience': hp.choice('patience', [best_params_stage1['patience']]),  # 固定patience
        'num_epochs': 100,  # 第二阶段使用更多epoch进行精细调优
    }
    
    trials_stage2 = Trials()
    best_stage2 = fmin(
        fn=lambda params: objective(params, element, train_loader, val_loader, device)['loss'],
        space=space_stage2,
        algo=tpe.suggest,
        max_evals=max_evals_stage2,
        trials=trials_stage2,
        show_progressbar=False
    )
    
    # 解析最终最佳参数
    best_params_final = {
        'lr': float(np.exp(best_stage2['lr'])),
        'weight_decay': float(np.exp(best_stage2['weight_decay'])),
        'batch_size': best_params_stage1['batch_size'],  # 第二阶段固定
        'dropout_rate': float(best_stage2['dropout_rate']),
        'patience': best_params_stage1['patience'],  # 第二阶段固定
    }
    
    logger.info(f"第二阶段最佳参数 ({element}): {best_params_final}")
    
    # 保存最终结果
    with open(os.path.join(results_dir, f'{element}_hyperopt_final.pkl'), 'wb') as f:
        pickle.dump({
            'best_params_stage1': best_params_stage1,
            'best_params_final': best_params_final,
            'trials_stage1': trials_stage1,
            'trials_stage2': trials_stage2
        }, f)
    
    return best_params_final

def load_best_params(element):
    """
    加载保存的最佳超参数
    
    参数:
        element (str): 元素名称
        
    返回:
        dict: 最佳超参数字典，如果不存在则返回None
    """
    results_dir = os.path.join(config.output_config['results_dir'], 'hyperopt')
    if not os.path.exists(results_dir):
        return None
    
    final_path = os.path.join(results_dir, f'{element}_hyperopt_final.pkl')
    stage1_path = os.path.join(results_dir, f'{element}_hyperopt_stage1.pkl')
    
    if os.path.exists(final_path):
        try:
            with open(final_path, 'rb') as f:
                data = pickle.load(f)
            return data['best_params_final']
        except Exception as e:
            logger.error(f"加载 {element} 的最终超参数失败: {e}")
    
    if os.path.exists(stage1_path):
        try:
            with open(stage1_path, 'rb') as f:
                data = pickle.load(f)
            return data['best_params']
        except Exception as e:
            logger.error(f"加载 {element} 的第一阶段超参数失败: {e}")
    
    return None

if __name__ == "__main__":
    # 简单测试
    print("超参数优化模块") 