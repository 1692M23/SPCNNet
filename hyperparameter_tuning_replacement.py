import os
import pickle
import logging
import torch
from sklearn.model_selection import ParameterGrid
from utils import ProgressManager
import config
from model import train_and_evaluate_model, SpectralResCNN_GCN
import json
import time
import numpy as np
import pandas as pd

# 设置logger
logger = logging.getLogger(__name__)

def hyperparameter_tuning(element, X_train, y_train, X_val, y_val, param_grid=None, 
                         device=None, batch_size=None, batches_per_round=None):
    """
    网格搜索超参数调优
    
    参数:
        element (str): 元素名称
        X_train (ndarray): 训练特征
        y_train (ndarray): 训练标签
        X_val (ndarray): 验证特征
        y_val (ndarray): 验证标签
        param_grid (dict): 参数网格
        device (torch.device): 计算设备
        batch_size (int): 超参数调优每批次的样本数量
        batches_per_round (int): 每轮处理的批次数量
        
    返回:
        dict: 最佳超参数
    """
    logger = logging.getLogger('hyperopt')
    
    # 如果未指定设备，使用CPU
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 如果未指定参数网格，使用默认值
    if param_grid is None:
        param_grid = {
            'lr': [0.001, 0.0005, 0.0001],
            'batch_size': [32, 64, 128],
            'dropout_rate': [0.3, 0.5, 0.7],
            'weight_decay': [1e-4, 1e-5, 1e-6],
            'use_gcn': [True, False]
        }
    
    # 添加固定参数
    param_grid.update({
        'num_epochs': [50],  # 调优时使用较少的时代数
        'patience': [15]
    })
    
    # 创建结果目录
    results_dir = os.path.join('results', 'hyperopt', element)
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建超参数调优状态文件路径
    hyperopt_state_file = os.path.join(results_dir, 'hyperopt_state.json')
    
    # 初始化参数组合和进度
    param_combinations = list(ParameterGrid(param_grid))
    total_combinations = len(param_combinations)
    processed_combinations = []
    best_val_loss = float('inf')
    best_params = None
    current_combination_idx = 0
    
    # 检查是否存在超参数调优状态文件
    if os.path.exists(hyperopt_state_file):
        try:
            with open(hyperopt_state_file, 'r') as f:
                hyperopt_state = json.load(f)
                
            # 恢复状态
            processed_combinations = hyperopt_state.get('processed_combinations', [])
            best_val_loss = hyperopt_state.get('best_val_loss', float('inf'))
            best_params = hyperopt_state.get('best_params', None)
            current_combination_idx = hyperopt_state.get('current_combination_idx', 0)
            
            logger.info(f"从中断点恢复超参数调优: 已处理 {len(processed_combinations)}/{total_combinations} 组合")
            
            # 如果上次已经完成，重新开始
            if hyperopt_state.get('completed', False):
                logger.info("超参数调优已完成，将从头开始")
                processed_combinations = []
                best_val_loss = float('inf')
                best_params = None
                current_combination_idx = 0
                
        except Exception as e:
            logger.warning(f"读取超参数调优状态文件失败: {str(e)}，将重新开始")
    
    # 更新超参数调优状态的辅助函数
    def update_hyperopt_state(processed, best_loss, best_p, current_idx, completed=False):
        """更新超参数调优状态"""
        hyperopt_state = {
            'element': element,
            'processed_combinations': processed,
            'best_val_loss': best_loss,
            'best_params': best_p,
            'current_combination_idx': current_idx,
            'total_combinations': total_combinations,
            'completed': completed,
            'timestamp': time.time()
        }
        
        with open(hyperopt_state_file, 'w') as f:
            json.dump(hyperopt_state, f, indent=4)
    
    # 创建训练和验证数据加载器
    from torch.utils.data import TensorDataset, DataLoader
    
    # 确保数据是torch.Tensor
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.FloatTensor(X_train)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.FloatTensor(y_train)
    if not isinstance(X_val, torch.Tensor):
        X_val = torch.FloatTensor(X_val)
    if not isinstance(y_val, torch.Tensor):
        y_val = torch.FloatTensor(y_val)
    
    # 对于一维y，添加一个维度
    if len(y_train.shape) == 1:
        y_train = y_train.unsqueeze(1)
    if len(y_val.shape) == 1:
        y_val = y_val.unsqueeze(1)
    
    # 对于二维X，添加通道维度
    if len(X_train.shape) == 2:
        X_train = X_train.unsqueeze(1)
    if len(X_val.shape) == 2:
        X_val = X_val.unsqueeze(1)
    
    # 循环处理参数组合
    logger.info(f"开始超参数调优: 共{total_combinations}组参数组合")
    
    # 从当前索引开始处理
    for i in range(current_combination_idx, total_combinations):
        params = param_combinations[i]
        
        # 如果已经处理过，跳过
        param_key = str(params)
        if param_key in [str(p) for p in processed_combinations]:
            logger.info(f"跳过已处理的参数组合 {i+1}/{total_combinations}: {params}")
            continue
        
        logger.info(f"评估超参数 {i+1}/{total_combinations}: {params}")
        
        # 创建数据加载器
        current_batch_size = params.get('batch_size', 32)
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=current_batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=current_batch_size,
            shuffle=False
        )
        
        # 创建模型
        from model import SpectralResCNN, SpectralResCNN_GCN
        
        input_size = X_train.shape[2]  # 假设形状为 [batch, channel, length]
        
        use_gcn = params.get('use_gcn', True)
        dropout_rate = params.get('dropout_rate', 0.5)
        
        # 创建模型
        if use_gcn:
            model = SpectralResCNN_GCN(input_size=input_size, device=device)
            # 设置dropout率
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = dropout_rate
        else:
            model = SpectralResCNN(input_size=input_size)
            # 设置dropout率
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = dropout_rate
        
        model = model.to(device)
        
        logger.info(f"已将模型中的dropout层设置为 {dropout_rate}")
        
        # 训练模型
        start_time = time.time()
        
        try:
            from model import train, train_and_evaluate_model
            
            # 准备配置
            config = {
                'training': {
                    'lr': params['lr'],
                    'weight_decay': params['weight_decay'],
                    'num_epochs': params['num_epochs'],
                    'early_stopping_patience': params['patience'],
                    'device': device
                },
                'model_config': {
                    'model_dir': os.path.join(results_dir, 'models'),
                    'use_gcn': use_gcn
                }
            }
            
            # 确保模型目录存在
            os.makedirs(config['model_config']['model_dir'], exist_ok=True)
            
            # 训练模型并获取验证损失
            train_losses, val_losses = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device=device,
                element=f"{element}_hyperparam_{i}"
            )
            
            # 获取最佳验证损失
            val_loss = min(val_losses) if val_losses else float('inf')
            
            # 记录训练时间
            training_time = time.time() - start_time
            logger.info(f"超参数评估完成: 验证损失={val_loss:.6f}, 耗时={training_time:.2f}秒")
            
            # 更新最佳参数
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params.copy()
                logger.info(f"找到新的最佳参数: {best_params}, 验证损失: {best_val_loss:.6f}")
                
                # 保存最佳参数
                best_params_file = os.path.join(results_dir, 'best_params.pkl')
                with open(best_params_file, 'wb') as f:
                    pickle.dump(best_params, f)
                
                # 也保存为JSON格式方便查看
                best_params_json = os.path.join(results_dir, 'best_params.json')
                with open(best_params_json, 'w') as f:
                    json.dump(best_params, f, indent=4)
            
            # 记录参数和结果
            result = {
                'params': params,
                'val_loss': val_loss,
                'training_time': training_time
            }
            
            # 添加到已处理列表
            processed_combinations.append(params)
            
            # 更新当前进度
            current_combination_idx = i + 1
            
            # 保存超参数调优状态
            update_hyperopt_state(processed_combinations, best_val_loss, best_params, current_combination_idx)
            
            # 保存结果到CSV
            results_csv = os.path.join(results_dir, 'hyperopt_results.csv')
            result_df = pd.DataFrame([{**params, 'val_loss': val_loss, 'training_time': training_time}])
            
            if os.path.exists(results_csv):
                # 追加到现有文件
                result_df.to_csv(results_csv, mode='a', header=False, index=False)
            else:
                # 创建新文件
                result_df.to_csv(results_csv, index=False)
                
        except Exception as e:
            logger.error(f"参数组合 {params} 评估失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 记录失败，但仍然更新当前进度
            current_combination_idx = i + 1
            update_hyperopt_state(processed_combinations, best_val_loss, best_params, current_combination_idx)
    
    # 全部完成，更新状态
    update_hyperopt_state(processed_combinations, best_val_loss, best_params, total_combinations, True)
    
    logger.info(f"超参数调优完成，共评估 {len(processed_combinations)}/{total_combinations} 组参数")
    logger.info(f"最佳参数: {best_params}")
    logger.info(f"最佳验证损失: {best_val_loss:.6f}")
    
    return best_params 