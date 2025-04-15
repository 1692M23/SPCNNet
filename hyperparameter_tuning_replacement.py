import os
import pickle
import logging
import torch
from sklearn.model_selection import ParameterGrid
from utils import ProgressManager
import config
from model import SpectralResCNN_GCN
import json
import time
import numpy as np
import pandas as pd

# 设置logger
logger = logging.getLogger(__name__)

def hyperparameter_tuning(element, train_loader, val_loader, param_grid=None, 
                         device=None, batch_size=None, batches_per_round=None, config_module=None):
    """
    网格搜索超参数调优 (修改为接收 DataLoader)
    
    参数:
        element (str): 元素名称
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        param_grid (dict): 参数网格
        device (torch.device): 计算设备
        batch_size (int): (此实现中未使用，保留兼容性)
        batches_per_round (int): (此实现中未使用，保留兼容性)
        config_module (module): 传递过来的config模块
        
    返回:
        dict: 最佳超参数
    """
    logger = logging.getLogger('hyperopt')
    
    # 如果未指定设备，使用CPU
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 如果未指定参数网格，使用默认值 (从config模块获取)
    if param_grid is None:
        if config_module and hasattr(config_module, 'tuning_config'):
             param_grid = config_module.tuning_config.get('param_grid')
        if not param_grid: # 如果还是没有，用硬编码的默认值
            param_grid = {
                'lr': [0.001, 0.0005, 0.0001],
                'batch_size': [32, 64, 128],
                'dropout_rate': [0.3, 0.5, 0.7],
                'weight_decay': [1e-4, 1e-5, 1e-6],
                'use_gru': [True, False],
                # 'use_gcn': [True, False] # Removed as GCN is no longer used
            }

    # 添加固定参数 (从config模块获取)
    fixed_params = {}
    if config_module and hasattr(config_module, 'tuning_config'):
        fixed_params['num_epochs'] = [config_module.tuning_config.get('stage1_epochs', 50)]
        fixed_params['patience'] = [config_module.tuning_config.get('early_stopping_patience', 15)] # 使用调优特定的patience
    else:
        fixed_params['num_epochs'] = [50]
        fixed_params['patience'] = [15]
    param_grid.update(fixed_params)
    
    # 创建结果目录
    results_dir = os.path.join('results', 'hyperopt', element)
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建超参数调优状态文件路径
    hyperopt_state_file = os.path.join(results_dir, 'hyperopt_state.json')
    
    # 初始化参数组合和进度
    param_combinations = list(ParameterGrid(param_grid))
    total_combinations = len(param_combinations)
    processed_combinations = []
    best_val_metric = -float('inf') # 追踪最佳 R²，初始化为负无穷
    best_params = None
    current_combination_idx = 0
    
    # 检查是否存在超参数调优状态文件
    if os.path.exists(hyperopt_state_file):
        try:
            with open(hyperopt_state_file, 'r') as f:
                hyperopt_state = json.load(f)
                
            # 恢复状态
            processed_combinations = hyperopt_state.get('processed_combinations', [])
            best_val_metric = hyperopt_state.get('best_val_metric', -float('inf'))
            best_params = hyperopt_state.get('best_params', None)
            current_combination_idx = hyperopt_state.get('current_combination_idx', 0)
            
            logger.info(f"从中断点恢复超参数调优: 已处理 {len(processed_combinations)}/{total_combinations} 组合")
            
            # 如果上次已经完成，重新开始
            if hyperopt_state.get('completed', False):
                logger.info("超参数调优已完成，将从头开始")
                processed_combinations = []
                best_val_metric = -float('inf')
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
            'best_val_metric': best_loss,
            'best_params': best_p,
            'current_combination_idx': current_idx,
            'total_combinations': total_combinations,
            'completed': completed,
            'timestamp': time.time()
        }
        
        with open(hyperopt_state_file, 'w') as f:
            json.dump(hyperopt_state, f, indent=4)
    
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
        
        # 创建模型 (需要 input_size)
        # !!! 如何获取 input_size？需要从数据加载器或config获取 !!!
        input_size = None
        if config_module and hasattr(config_module, 'model_config'):
             input_size = config_module.model_config.get('input_size') # 尝试从config获取
        if input_size is None:
             # 尝试从DataLoader的第一个batch获取
             try:
                 first_batch_data, _ = next(iter(train_loader))
                 input_size = first_batch_data.shape[2] # 假设 [batch, channel, length]
                 logger.info(f"从数据加载器推断输入大小: {input_size}")
             except Exception as data_err:
                 logger.error(f"无法从数据加载器推断输入大小: {data_err}")
                 logger.error("请在 config.py 中设置 model_config['input_size']")
                 continue # 跳过此参数组合

        # 导入模型
        try:
            # 尝试从项目根目录下的 model.py 导入
            from model import SpectralResCNN_GCN
        except ImportError:
            # 如果失败，尝试从当前目录导入（可能作为独立脚本运行）
            pass

        use_gru = params.get('use_gru', True)
        # use_gcn = params.get('use_gcn', True) # Removed, GCN is no longer part of the model
        dropout_rate = params.get('dropout_rate', 0.5)
        
        # 创建模型实例 (GCN is removed, always create SpectralResCNN_GCN)
        model = SpectralResCNN_GCN(input_size=input_size, device=device, use_gru=use_gru)
        
        # 设置dropout率 (需要确保模型创建后设置)
        for module in model.modules():
             if isinstance(module, torch.nn.Dropout):
                 module.p = dropout_rate
        model = model.to(device)
        
        logger.info(f"已将模型中的dropout层设置为 {dropout_rate}")
        
        # 训练模型
        start_time = time.time()
        
        try:
            from model import train
            
            # 创建临时的 config 字典给 train 函数
            # !!! 注意：train 函数内部也需要能处理 config 模块 !!!
            # (最好是让 train 函数也接收 config_module)
            temp_config = {
                'training': {
                    # Robust replacement: Use 'lr' if present, otherwise try 'learning_rate'
                    'lr': params.get('lr', params.get('learning_rate')), 
                    'weight_decay': params['weight_decay'],
                    'num_epochs': params['num_epochs'],
                    'early_stopping_patience': params['patience'],
                    'device': device,
                     # 传递调优特定的配置给train
                    'resume_training': False, # 调优时不恢复检查点
                    'save_checkpoints': False # 调优时不保存检查点
                },
                'model_config': {
                    'model_dir': os.path.join(results_dir, 'models'),
                    'use_gru': use_gru,
                    'use_gcn': False
                }
            }
            # Ensure that the lr value is not None if neither key was found
            if temp_config['training']['lr'] is None:
                 logger.error(f"参数组合 {params} 中缺少 'lr' 或 'learning_rate' 键。跳过此组合。")
                 processed_combinations.append(params) # Mark as processed to avoid retrying
                 update_hyperopt_state(processed_combinations, best_val_metric, best_params, current_combination_idx)
                 continue # Skip to the next parameter combination
            # !!! 更好的方法是修改 train 函数接收 config_module !!!
            # 如果 model.py 的 train 已修改为接收模块，则直接传递
            train_config_arg = config_module if config_module else temp_config

            # 记录传递给 train 函数的配置信息
            logger.info(f"  [Tune Detail] 调用 model.train 使用配置: epochs={temp_config['training']['num_epochs']}, patience={temp_config['training']['early_stopping_patience']}, lr={temp_config['training']['lr']:.2e}, wd={temp_config['training']['weight_decay']:.2e}, use_gru={temp_config['model_config']['use_gru']}, use_gcn={temp_config['model_config']['use_gcn']}")

            # 确保模型目录存在
            os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)
            
            # 训练模型并获取验证损失和 R²
            # 注意：train 返回 (model, best_val_loss, best_val_r2)
            trained_model, best_run_val_loss, best_run_val_r2 = train(
                model=model,
                train_loader=train_loader, # 确保 train_loader 在此作用域内有效
                val_loader=val_loader,   # 确保 val_loader 在此作用域内有效
                config=train_config_arg, # 传递模块或字典
                device=device,           # 确保 device 在此作用域内有效
                element=f"{element}_hyperparam_{i}"
            )
            
            # 获取最佳验证 R² (train 函数直接返回)
            val_metric = best_run_val_r2 # 使用 R² 作为评估指标
            val_loss = best_run_val_loss   # 同时记录损失用于日志
            
            # 记录训练时间
            training_time = time.time() - start_time
            logger.info(f"超参数评估完成: 验证损失={val_loss:.6f}, 验证 R²={val_metric:.4f}, 耗时={training_time:.2f}秒")
            # 记录传递给 train 函数的配置信息
            logger.info(f"  [Tune Result] 参数: {params} -> Val Loss: {val_loss:.6f}, Val R2: {val_metric:.4f}")
            
            # 更新最佳参数 (基于 R²)
            if not np.isnan(val_metric) and val_metric > best_val_metric:
                best_val_metric = val_metric
                best_params = params.copy()
                logger.info(f"找到新的最佳参数(基于R²): {best_params}, 验证 R²: {best_val_metric:.4f}")
                
                # 记录传递给 train 函数的配置信息 (记录最佳R²)
                logger.info(f"    [Tune Best Update] 新最佳参数: {best_params}, 新最佳验证 R²: {best_val_metric:.4f}")
            
                # 保存最佳参数 (仍然保存最佳参数，即使标准变了)
                best_params_file = os.path.join(results_dir, 'best_params.pkl')
                with open(best_params_file, 'wb') as f:
                    pickle.dump(best_params, f)
                
                # 也保存为JSON格式方便查看
                best_params_path = os.path.join(results_dir, 'best_params.json')
                if best_params is not None:
                    best_params.pop('use_gcn', None)
                    with open(best_params_path, 'w') as f:
                        json.dump(best_params, f, indent=4)
                    logger.info(f"最佳超参数已保存到: {best_params_path}")
                    logger.info(f"最佳参数: {best_params}")
                    logger.info(f"对应的最低验证损失: {val_loss:.6f}")
                else:
                    logger.warning("最佳参数为空，无法保存。")
            
            # 记录参数和结果
            result = {
                'params': params,
                'val_loss': val_loss,
                'val_r2': val_metric,
                'training_time': training_time
            }
            
            # 添加到已处理列表
            processed_combinations.append(params)
            
            # 更新当前进度
            current_combination_idx = i + 1
            
            # 保存超参数调优状态
            update_hyperopt_state(processed_combinations, best_val_metric, best_params, current_combination_idx)
            
            # 保存结果到CSV
            results_csv = os.path.join(results_dir, 'hyperopt_results.csv')
            result_df = pd.DataFrame([{**params, 'val_loss': val_loss, 'val_r2': val_metric, 'training_time': training_time}])
            
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
            
            # 在每次循环结束（无论成功或失败）后更新状态
            # 这样即使中途中断，下次也能从下一个组合开始
            update_hyperopt_state(processed_combinations, best_val_metric, best_params, current_combination_idx)
    
    # 全部完成，更新状态
    update_hyperopt_state(processed_combinations, best_val_metric, best_params, total_combinations, True)
    
    logger.info(f"超参数调优循环结束，共处理 {current_combination_idx}/{total_combinations} 组参数")
    logger.info(f"最佳参数: {best_params}")
    logger.info(f"最佳验证 R²: {best_val_metric:.4f}")
    
    # 确保返回的字典也不包含 use_gcn
    if best_params is not None:
        best_params.pop('use_gcn', None)
        
    return best_params 