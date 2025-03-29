#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量超参数优化模块：主要提供获取多组最优超参数的功能
基于hyperopt库和自定义网格搜索的组合方法
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pickle
from tqdm import tqdm
import time
import random
from collections import defaultdict

# 导入自定义模块
import config
from model import SpectralResCNN, SpectralResCNN_GCN, train_and_evaluate_model

# 尝试导入hyperopt_tuning中的方法
try:
    from hyperopt_tuning import objective, run_hyperopt_tuning
except ImportError:
    # 如果导入失败，提供简单的兼容函数
    def objective(params, element, train_loader, val_loader, device=None):
        """与hyperopt_tuning中的objective函数兼容"""
        if device is None:
            device = config.training_config['device']
        
        logger.info(f"评估超参数: {params}")
        
        # 设置训练配置
        train_config = config.CONFIG.copy()
        train_config['training_config'] = {
            'device': device,
            'batch_size': int(params.get('batch_size', 32)),
            'num_epochs': int(params.get('num_epochs', 50)),
            'lr': float(params.get('lr', 0.001)),
            'weight_decay': float(params.get('weight_decay', 1e-4)),
            'early_stopping_patience': int(params.get('patience', 10))
        }
        train_config['model_config'] = {
            'use_gcn': params.get('use_gcn', True)
        }
        
        # 选择模型类型
        use_gcn = params.get('use_gcn', True)
        if use_gcn:
            model = SpectralResCNN_GCN(
                input_size=config.model_config['input_size'],
                device=device
            )
        else:
            model = SpectralResCNN(
                input_size=config.model_config['input_size']
            ).to(device)
        
        # 训练和评估模型
        _, val_loss, _ = train_and_evaluate_model(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=val_loader,
            element=f"{element}_tune",
            config=train_config
        )
        
        return {
            'loss': val_loss,
            'status': STATUS_OK,
            'params': params
        }

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.output_config['log_dir'], 'batch_hyperopt.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('batch_hyperopt')

def run_element_hyperopt(element, train_loader=None, val_loader=None, batch_size=1000, batches_per_round=5, device=None):
    """
    运行元素超参数优化，获取两组最优超参数
    
    参数:
        element (str): 元素名称
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        batch_size: 批次大小
        batches_per_round: 每轮处理的批次数
        device: 计算设备
        
    返回:
        dict: 包含两组最优超参数的字典
    """
    if device is None:
        device = config.training_config['device']
    
    logger.info(f"开始为元素 {element} 获取两组最优超参数")
    
    # 创建结果目录
    results_dir = os.path.join(config.output_config['results_dir'], 'hyperopt')
    os.makedirs(results_dir, exist_ok=True)
    
    # 检查是否已有保存的结果
    final_results_file = os.path.join(results_dir, f'{element}_best_params.pkl')
    if os.path.exists(final_results_file):
        try:
            with open(final_results_file, 'rb') as f:
                results = pickle.load(f)
                
            # 检查结果是否完整
            if 'best_params_set1' in results and 'best_params_set2' in results:
                logger.info(f"从缓存加载 {element} 的两组最优超参数")
                logger.info(f"参数组1: {results['best_params_set1']}")
                logger.info(f"参数组2: {results['best_params_set2']}")
                return results
        except Exception as e:
            logger.warning(f"读取缓存结果失败: {e}，将重新进行优化")
    
    # 如果没有数据加载器，需要创建（这部分可能需要根据实际情况修改）
    if train_loader is None or val_loader is None:
        try:
            from main import load_data
            logger.info(f"加载 {element} 的数据")
            data = load_data(element)
            if data:
                X_train, y_train, X_val, y_val, X_test, y_test = data
                
                # 创建数据加载器
                from torch.utils.data import TensorDataset, DataLoader
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_train), 
                    torch.FloatTensor(y_train)
                )
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val), 
                    torch.FloatTensor(y_val)
                )
                
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=batch_size, 
                    shuffle=True
                )
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=batch_size
                )
            else:
                logger.error(f"无法加载 {element} 的数据")
                return None
        except Exception as e:
            logger.error(f"准备数据加载器失败: {e}")
            return None
    
    # 初始化结果存储
    results = {
        'element': element,
        'processed_batches': [],
        'all_trials': []
    }
    
    # 策略1：使用不同随机种子的两次优化
    logger.info("策略1: 使用两种不同随机种子的hyperopt优化")
    
    # 第一次优化：使用默认随机种子
    logger.info("运行第一次超参数优化")
    try:
        best_params_set1 = run_hyperopt_tuning(
            element=f"{element}_set1",
            train_loader=train_loader,
            val_loader=val_loader,
            max_evals_stage1=15,
            max_evals_stage2=10,
            device=device
        )
        
        results['best_params_set1'] = best_params_set1
        logger.info(f"第一组最优参数: {best_params_set1}")
    except Exception as e:
        logger.error(f"第一次优化失败: {e}")
        # 使用备用方法：网格搜索
        best_params_set1 = _fallback_grid_search(
            element, train_loader, val_loader, device, "set1"
        )
        results['best_params_set1'] = best_params_set1
    
    # 第二次优化：使用不同的随机种子和变化的搜索空间
    logger.info("运行第二次超参数优化，使用不同随机种子")
    random.seed(int(time.time()))  # 使用当前时间设置随机种子
    np.random.seed(int(time.time()) % 10000)
    
    # 定义不同的搜索空间（反向思维，避免与第一组过于相似）
    try:
        # 定义反向搜索空间
        if 'use_gcn' in best_params_set1:
            use_gcn_opposite = not best_params_set1['use_gcn']
        else:
            use_gcn_opposite = False
        
        # 第二次优化，使用不同的随机种子和倾向于与第一组不同的搜索空间
        best_params_set2 = _run_second_optimization(
            element, train_loader, val_loader, best_params_set1, use_gcn_opposite, device
        )
        
        results['best_params_set2'] = best_params_set2
        logger.info(f"第二组最优参数: {best_params_set2}")
    except Exception as e:
        logger.error(f"第二次优化失败: {e}")
        # 使用备用方法：使用与第一组相反的gcn设置
        opposite_params = best_params_set1.copy()
        if 'use_gcn' in opposite_params:
            opposite_params['use_gcn'] = not opposite_params['use_gcn']
        else:
            opposite_params['use_gcn'] = False
        
        results['best_params_set2'] = opposite_params
    
    # 保存结果
    try:
        with open(final_results_file, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"已保存两组最优超参数至 {final_results_file}")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")
    
    return results

def _run_second_optimization(element, train_loader, val_loader, best_params_set1, use_gcn_opposite, device):
    """
    运行第二次优化，尝试获取与第一组不同的参数
    """
    # 修改搜索空间以尝试找到与第一组不同的参数
    space_stage1 = {
        'lr': hp.loguniform('lr', np.log(0.00001), np.log(0.01)),
        'weight_decay': hp.loguniform('weight_decay', np.log(1e-6), np.log(1e-3)),
        'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
        'dropout_rate': hp.uniform('dropout_rate', 0.2, 0.7),
        'patience': hp.choice('patience', [5, 10, 15, 20]),
        'use_gcn': hp.choice('use_gcn', [use_gcn_opposite, not use_gcn_opposite]),  # 更倾向于与第一组不同
        'num_epochs': 50,  # 固定epoch数量用于调参
    }
    
    trials = Trials()
    best = fmin(
        fn=lambda params: objective(params, element, train_loader, val_loader, device)['loss'],
        space=space_stage1,
        algo=tpe.suggest,
        max_evals=15,
        trials=trials,
        show_progressbar=False
    )
    
    # 解析最佳参数
    best_params = {
        'lr': float(np.exp(best['lr'])),
        'weight_decay': float(np.exp(best['weight_decay'])),
        'batch_size': [16, 32, 64, 128][best['batch_size']],
        'dropout_rate': float(best['dropout_rate']),
        'patience': [5, 10, 15, 20][best['patience']],
        'use_gcn': [use_gcn_opposite, not use_gcn_opposite][best['use_gcn']]
    }
    
    return best_params

def _fallback_grid_search(element, train_loader, val_loader, device, param_set="set1"):
    """
    当hyperopt优化失败时的备用网格搜索方法
    """
    logger.info(f"使用网格搜索获取 {param_set} 的超参数")
    
    # 定义搜索空间
    if param_set == "set1":
        # 第一组默认偏好使用GCN
        param_grid = {
            'lr': [0.001, 0.0001],
            'weight_decay': [1e-4, 1e-5],
            'batch_size': [32, 64],
            'use_gcn': [True],
            'dropout_rate': [0.3, 0.5]
        }
    else:
        # 第二组默认偏好不使用GCN
        param_grid = {
            'lr': [0.0005, 0.0002],
            'weight_decay': [1e-5, 1e-6],
            'batch_size': [16, 32],
            'use_gcn': [False],
            'dropout_rate': [0.4, 0.6]
        }
    
    # 计算所有参数组合
    from itertools import product
    param_keys = list(param_grid.keys())
    param_values = [param_grid[key] for key in param_keys]
    
    best_params = None
    best_val_loss = float('inf')
    
    # 遍历所有参数组合
    for values in product(*param_values):
        params = dict(zip(param_keys, values))
        params['num_epochs'] = 30  # 网格搜索使用较短的epoch
        params['patience'] = 10
        
        logger.info(f"评估参数: {params}")
        
        # 选择模型类型
        use_gcn = params.get('use_gcn', True)
        if use_gcn:
            model = SpectralResCNN_GCN(
                input_size=config.model_config['input_size'],
                device=device
            )
        else:
            model = SpectralResCNN(
                input_size=config.model_config['input_size']
            ).to(device)
        
        # 设置训练配置
        train_config = config.CONFIG.copy()
        train_config['training_config'] = {
            'device': device,
            'batch_size': int(params.get('batch_size', 32)),
            'num_epochs': int(params.get('num_epochs', 30)),
            'lr': float(params.get('lr', 0.001)),
            'weight_decay': float(params.get('weight_decay', 1e-4)),
            'early_stopping_patience': int(params.get('patience', 10))
        }
        train_config['model_config'] = {
            'use_gcn': params.get('use_gcn', True)
        }
        
        try:
            # 训练和评估模型
            _, val_loss, _ = train_and_evaluate_model(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=val_loader,
                element=f"{element}_{param_set}",
                config=train_config
            )
            
            # 更新最佳参数
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params
                logger.info(f"发现新的最佳参数: {best_params}, 验证损失: {best_val_loss:.6f}")
                
        except Exception as e:
            logger.error(f"参数评估失败: {e}")
    
    if best_params is None:
        # 如果所有参数组合都失败，使用默认参数
        logger.warning("所有参数组合评估失败，使用默认参数")
        best_params = {
            'lr': 0.001,
            'weight_decay': 1e-4,
            'batch_size': 32,
            'use_gcn': param_set == "set1",  # set1使用GCN，set2不使用
            'dropout_rate': 0.5,
            'patience': 10,
            'num_epochs': 100
        }
    
    return best_params

if __name__ == "__main__":
    # 测试代码
    print("批量超参数优化模块") 