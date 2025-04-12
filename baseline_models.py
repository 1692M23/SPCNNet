#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基线模型模块：使用XGBoost和LightGBM实现基线回归模型
使用标准的训练-验证-测试流程
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import pickle
import gc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import json

# 导入自定义模块
import config
from utils import CacheManager, ProgressManager, ask_clear_cache
from config import Config

# 确保输出目录配置完整
if 'log_dir' not in config.output_config:
    config.output_config['log_dir'] = os.path.join('logs')
    os.makedirs(config.output_config['log_dir'], exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.output_config['log_dir'], 'baseline_models.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('baseline_models')

# 确保其他输出目录配置完整
if 'model_dir' not in config.output_config:
    config.output_config['model_dir'] = os.path.join('models', 'baseline')
    logger.info(f"添加默认model_dir配置: {config.output_config['model_dir']}")

if 'results_dir' not in config.output_config:
    config.output_config['results_dir'] = os.path.join('results', 'baseline')
    logger.info(f"添加默认results_dir配置: {config.output_config['results_dir']}")

if 'cache_dir' not in config.output_config:
    config.output_config['cache_dir'] = os.path.join('cache', 'baseline')
    logger.info(f"添加默认cache_dir配置: {config.output_config['cache_dir']}")

# 创建缓存管理器
cache_manager = CacheManager(cache_dir=os.path.join(config.output_config['cache_dir'], 'baseline_models'))

class XGBoostModel:
    """
    XGBoost模型 (使用标准训练流程)
    """
    def __init__(self, config_obj=None, params_update=None):
        if config_obj is None:
            config_obj = Config
            
        self.params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.01,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 1000, # 总轮数
            'early_stopping_rounds': 50, # 早停轮数
            'seed': 42
        }
        if params_update:
            self.params.update(params_update)
            logger.info(f"XGBoostModel 使用更新后的参数初始化: {params_update}")
            
        self.model = None
        # self.trained_batches = [] # 不再需要

    def train(self, X_train, y_train, X_val, y_val):
        """
        训练 XGBoost 模型 (单次完整训练)

        参数:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
        """
        logger.info("训练 XGBoost 模型")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        evals = [(dtrain, 'train'), (dval, 'val')]

        # 从参数中提取训练控制参数
        num_boost_round = self.params.get('n_estimators', 1000)
        early_stopping_rounds = self.params.get('early_stopping_rounds', 50)

        # 准备传递给 xgb.train 的参数字典 (移除训练控制参数)
        train_params = self.params.copy()
        train_params.pop('n_estimators', None)
        train_params.pop('early_stopping_rounds', None)

        start_time = time.time()
        evals_result = {}
        self.model = xgb.train(
            train_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result, # 用于记录历史
            verbose_eval=100 # 每 100 轮打印一次日志
        )
        elapsed_time = time.time() - start_time
        
        best_iteration = self.model.best_iteration if hasattr(self.model, 'best_iteration') else 'N/A'
        best_score = self.model.best_score if hasattr(self.model, 'best_score') else 'N/A'

        logger.info(f"XGBoost 训练完成, 耗时: {elapsed_time:.2f} 秒")
        logger.info(f"最佳迭代轮数: {best_iteration}, 最佳分数 (验证集RMSE): {best_score}")

        # 可选：保存训练历史
        # history_path = os.path.join(config.output_config['results_dir'], 'xgb_train_history.json')
        # with open(history_path, 'w') as f:
        #    json.dump(evals_result, f)

    def predict(self, X):
        """
        预测
        """
        if self.model is None:
            raise ValueError("模型未训练")
            
        dtest = xgb.DMatrix(X)
        # 使用最佳迭代次数进行预测 (如果早停被触发)
        best_iteration = 0 # 默认值为0，表示使用所有树
        if hasattr(self.model, 'best_iteration') and self.model.best_iteration > 0:
            best_iteration = self.model.best_iteration
        return self.model.predict(dtest, iteration_range=(0, best_iteration + 1))

    def predict_in_batches(self, X, batch_size=1000):
        """
        分批预测 (适用于大型测试集)
        """
        if self.model is None:
            raise ValueError("模型未训练")
        
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        predictions = np.zeros(n_samples)
        # 使用最佳迭代次数进行预测
        best_iteration = 0
        if hasattr(self.model, 'best_iteration') and self.model.best_iteration > 0:
             best_iteration = self.model.best_iteration

        with ProgressManager(n_batches, desc="XGBoost预测") as progress:
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X[start_idx:end_idx]
                
                dtest = xgb.DMatrix(X_batch)
                predictions[start_idx:end_idx] = self.model.predict(dtest, iteration_range=(0, best_iteration + 1))
                
                progress.update(1)
                del X_batch, dtest
                gc.collect()
        
        return predictions
    
    def save(self, element):
        """
        保存模型
        """
        if self.model is None:
            raise ValueError("模型未训练")
            
        os.makedirs(config.output_config['model_dir'], exist_ok=True)
        model_path = os.path.join(config.output_config['model_dir'], f'xgboost_{element}.json')
        self.model.save_model(model_path)
        
        # 不再需要保存训练状态
        # state_path = os.path.join(config.output_config['model_dir'], f'xgboost_{element}_state.pkl')
        # with open(state_path, 'wb') as f:
        #     pickle.dump({}, f) # 保存空字典或完全移除
        
        logger.info(f"XGBoost模型已保存: {model_path}")
    
    def load(self, element):
        """
        加载模型
        """
        model_path = os.path.join(config.output_config['model_dir'], f'xgboost_{element}.json')
        if not os.path.exists(model_path):
            logger.warning(f"找不到模型文件: {model_path}")
            return False
        
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        # 不再需要加载训练状态
        # state_path = os.path.join(config.output_config['model_dir'], f'xgboost_{element}_state.pkl')
        # if os.path.exists(state_path):
        #     try:
        #         with open(state_path, 'rb') as f:
        #            state = pickle.load(f)
        #     except: # 处理可能的空文件或格式问题
        #         pass 
        
        logger.info(f"XGBoost模型已加载: {model_path}")
        return True

class LightGBMModel:
    """
    LightGBM模型 (使用标准训练流程)
    """
    def __init__(self, config_obj=None, params_update=None):
        if config_obj is None:
            config_obj = Config
            
        self.params = {
            'objective': 'regression',
            'metric': 'l2', # l2 是 MSE
            'learning_rate': 0.01,
            'num_leaves': 31,
            'max_depth': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 1000, # 总轮数
            'early_stopping_rounds': 50, # 早停轮数 (通过回调实现)
            'seed': 42
        }
        if params_update:
            self.params.update(params_update)
            logger.info(f"LightGBMModel 使用更新后的参数初始化: {params_update}")
            
        self.model = None
        # self.trained_batches = [] # 不再需要

    def train(self, X_train, y_train, X_val, y_val):
        """
        训练 LightGBM 模型 (单次完整训练)

        参数:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
        """
        logger.info("训练 LightGBM 模型")
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 从参数中提取训练控制参数
        num_boost_round = self.params.get('n_estimators', 1000)
        early_stopping_rounds = self.params.get('early_stopping_rounds', 50)

        # 准备传递给 lgb.train 的参数字典 (移除训练控制参数)
        train_params = self.params.copy()
        train_params.pop('n_estimators', None)
        train_params.pop('early_stopping_rounds', None)
        # 确保 metric 是列表形式
        metric = train_params.get('metric', 'l2')
        if not isinstance(metric, list):
             train_params['metric'] = [metric]


        # 准备回调
        callbacks = [
            lgb.log_evaluation(period=100) # 每 100 轮打印一次日志
        ]
        if early_stopping_rounds > 0:
            # 使用 lgb.early_stopping 回调
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds,
                                                first_metric_only=False, # 基于所有验证指标
                                                verbose=True))

        start_time = time.time()
        # evals_result = {} # 不再直接传递给 train，如果需要历史记录，应使用 lgb.record_evaluation 回调
        self.model = lgb.train(
            train_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data], # 提供验证集
            valid_names=['train', 'val'],
            callbacks=callbacks, # 使用回调进行早停和日志记录
            # evals_result=evals_result # 移除此错误参数
        )
        elapsed_time = time.time() - start_time

        best_iteration = self.model.best_iteration if hasattr(self.model, 'best_iteration') and self.model.best_iteration > 0 else num_boost_round
        # 获取最佳分数 - lgb.early_stopping 会将最佳分数存储在 model.best_score 中
        best_score_dict = self.model.best_score
        best_score = 'N/A'
        metric_key = train_params['metric'][0] # 取第一个指标用于报告
        if best_score_dict and 'val' in best_score_dict and metric_key in best_score_dict['val']:
            best_score = best_score_dict['val'][metric_key]


        logger.info(f"LightGBM 训练完成, 耗时: {elapsed_time:.2f} 秒")
        logger.info(f"最佳迭代轮数: {best_iteration}, 最佳分数 (验证集 {metric_key}): {best_score}")

        # 可选：保存训练历史
        # history_path = os.path.join(config.output_config['results_dir'], 'lgb_train_history.json')
        # with open(history_path, 'w') as f:
        #    json.dump(evals_result, f)

    def predict(self, X):
        """
        预测
        """
        if self.model is None:
            raise ValueError("模型未训练")
            
        # 使用 best_iteration 进行预测
        best_iteration = 0
        if hasattr(self.model, 'best_iteration') and self.model.best_iteration > 0:
            best_iteration = self.model.best_iteration
        return self.model.predict(X, num_iteration=best_iteration)
    
    def predict_in_batches(self, X, batch_size=1000):
        """
        分批预测 (适用于大型测试集)
        """
        if self.model is None:
            raise ValueError("模型未训练")
        
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        predictions = np.zeros(n_samples)
        # 使用 best_iteration 进行预测
        best_iteration = 0
        if hasattr(self.model, 'best_iteration') and self.model.best_iteration > 0:
             best_iteration = self.model.best_iteration

        with ProgressManager(n_batches, desc="LightGBM预测") as progress:
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X[start_idx:end_idx]
                
                predictions[start_idx:end_idx] = self.model.predict(X_batch, num_iteration=best_iteration)
                
                progress.update(1)
                del X_batch
                gc.collect()
        
        return predictions
    
    def save(self, element):
        """
        保存模型
        """
        if self.model is None:
            raise ValueError("模型未训练")
            
        os.makedirs(config.output_config['model_dir'], exist_ok=True)
        model_path = os.path.join(config.output_config['model_dir'], f'lightgbm_{element}.txt')
        self.model.save_model(model_path)
        
        # 不再需要保存训练状态
        # state_path = os.path.join(config.output_config['model_dir'], f'lightgbm_{element}_state.pkl')
        # with open(state_path, 'wb') as f:
        #     pickle.dump({}, f)

        logger.info(f"LightGBM模型已保存: {model_path}")
    
    def load(self, element):
        """
        加载模型
        """
        model_path = os.path.join(config.output_config['model_dir'], f'lightgbm_{element}.txt')
        if not os.path.exists(model_path):
            logger.warning(f"找不到模型文件: {model_path}")
            return False
        
        self.model = lgb.Booster(model_file=model_path)
        
        # 不再需要加载训练状态
        # state_path = os.path.join(config.output_config['model_dir'], f'lightgbm_{element}_state.pkl')
        # if os.path.exists(state_path):
        #     try:
        #         with open(state_path, 'rb') as f:
        #            state = pickle.load(f)
        #     except:
        #         pass
        
        logger.info(f"LightGBM模型已加载: {model_path}")
        return True

def evaluate_baseline_model(y_true, y_pred, model_name, element):
    """
    评估基线模型性能
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        model_name: 模型名称
        element: 元素名称
        
    返回:
        评估指标
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    scatter = np.std(y_pred - y_true)
    
    results = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'scatter': scatter
    }
    
    # 保存结果
    os.makedirs(config.output_config['results_dir'], exist_ok=True)
    results_path = os.path.join(config.output_config['results_dir'], f'{model_name}_{element}_results.csv')
    
    # 生成详细结果
    results_df = pd.DataFrame({
        'true': y_true,
        'pred': y_pred,
        'error': y_pred - y_true
    })
    results_df.to_csv(results_path, index=False)
    
    # 保存评估指标摘要
    summary_path = os.path.join(config.output_config['results_dir'], f'{model_name}_{element}_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"{model_name} 在 {element} 上的评估结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")
        f.write(f"R²: {r2:.6f}\n")
        f.write(f"散度: {scatter:.6f}\n")
    
    logger.info(f"{model_name} 在 {element} 上的评估结果:")
    logger.info(f"  MSE: {mse:.6f}")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  MAE: {mae:.6f}")
    logger.info(f"  R²: {r2:.6f}")
    logger.info(f"  散度: {scatter:.6f}")
    
    return results

def load_processed_data(element, data_type='train', use_main_dataset=False, dataset_path=None):
    """
    加载处理后的数据
    
    参数:
        element: 元素名称
        data_type: 数据类型 (train, test)
        use_main_dataset: 是否使用主模型数据集
        dataset_path: 数据集路径
        
    返回:
        X: 特征数据
        y: 标签数据
    """
    if use_main_dataset and dataset_path:
        try:
            # 使用主模型数据集
            data_file = dataset_path.replace('{type}', data_type)
            if not os.path.exists(data_file):
                logger.error(f"找不到主模型数据文件: {data_file}")
                return None, None
                
            data = np.load(data_file)
            
            # 提取特征和标签，需要适配主模型数据格式
            if 'spectra' in data and element in data:
                X = data['spectra'] 
                y = data[element]
            elif 'X' in data and 'y' in data:
                X = data['X']
                y = data['y']
            else:
                # 尝试使用主模型的数据格式
                from main import load_data
                X, y, _ = load_data(data_file, element)
                
            # 检查数据形状
            if len(X.shape) == 3:  # [samples, channels, features]
                # 将3D数据转换为2D
                n_samples, n_channels, n_features = X.shape
                X = X.reshape(n_samples, n_channels * n_features)
            
            logger.info(f"加载 {element} 的 {data_type} 数据，形状: {X.shape}, {y.shape}")
            return X, y
        
        except Exception as e:
            logger.error(f"加载 {element} 的 {data_type} 数据出错: {e}")
            return None, None
    else:
        try:
            # 尝试加载特定元素的数据
            data_file = os.path.join('processed_data', f'{data_type}_dataset.npz')
            
            if not os.path.exists(data_file):
                logger.error(f"找不到数据文件: {data_file}")
                return None, None
            
            data = np.load(data_file)
            
            # 提取特征和标签
            if 'spectra' in data:
                X = data['spectra']
                # 如果abundance字段存在
                if 'abundance' in data:
                    # 检查abundance是多维还是一维
                    if len(data['abundance'].shape) > 1:
                        # 如果是多维，尝试从elements中获取索引
                        if 'elements' in data and element in str(data['elements']):
                            elements_dict = {e: i for i, e in enumerate(data['elements'])}
                            if element in elements_dict:
                                y = data['abundance'][:, elements_dict[element]]
                            else:
                                # 尝试直接按元素名称查找列
                                try:
                                    y = data[element]
                                except:
                                    logger.error(f"找不到元素 {element} 的丰度数据")
                                    return None, None
                        else:
                            # 尝试直接使用abundance的第一列
                            y = data['abundance'][:, 0]
                            logger.warning(f"使用丰度数据的第一列作为 {element} 的丰度")
                    else:
                        # 一维，直接使用
                        y = data['abundance']
                elif element in data:
                    # 如果元素字段存在
                    y = data[element]
                else:
                    # 尝试使用 'y' 字段
                    y = data['y'] if 'y' in data else None
                    if y is None:
                        logger.error(f"找不到 {element} 的相关数据")
                        return None, None
            elif 'X' in data and 'y' in data:
                X = data['X']
                y = data['y']
            else:
                logger.error(f"找不到 {element} 的相关数据")
                return None, None
            
            # 检查数据形状
            if len(X.shape) == 3:  # [samples, channels, features]
                # 将3D数据转换为2D
                n_samples, n_channels, n_features = X.shape
                X = X.reshape(n_samples, n_channels * n_features)
            
            logger.info(f"加载 {element} 的 {data_type} 数据，形状: {X.shape}, {y.shape}")
            return X, y
        
        except Exception as e:
            logger.error(f"加载 {element} 的 {data_type} 数据出错: {e}")
            return None, None

def train_and_evaluate_baseline(element, model_type='xgboost', 
                               val_size=0.2, # 保留 val_size 用于划分验证集
                               force_retrain=False, evaluate_only=False, device=None,
                               xgb_params=None, lgb_params=None,
                               use_main_dataset=False, dataset_path=None # 保留数据集参数
                               ):
    """使用标准训练流程训练和评估基线模型"""
    # 创建模型实例，并传入参数更新
    model_params_update = None
    model_config = Config # 传递 Config 对象
    if model_type.lower() == 'xgboost':
        model_params_update = xgb_params
        model = XGBoostModel(config_obj=model_config, params_update=model_params_update) 
        model_name = 'xgboost'
    elif model_type.lower() == 'lightgbm':
        model_params_update = lgb_params
        model = LightGBMModel(config_obj=model_config, params_update=model_params_update)
        model_name = 'lightgbm'
    else:
        logger.error(f"不支持的模型类型: {model_type}")
        return None

    # 配置GPU/CPU (如果需要)
    current_params = model.params # 获取当前模型参数
    if device and 'cuda' in str(device):
        if model_type.lower() == 'xgboost':
            # 确保tree_method是hist，并设置device
            current_params['tree_method'] = 'hist'
            current_params['device'] = 'cuda'
            logger.info(f"为 XGBoost 配置 GPU 支持: {current_params}")
        elif model_type.lower() == 'lightgbm':
            current_params['device'] = 'cuda'
            logger.info(f"为 LightGBM 配置 GPU 支持: {current_params}")
    elif device and 'cpu' in str(device):
         if model_type.lower() == 'xgboost':
             # 移除GPU特定参数
             current_params.pop('device', None)
             # tree_method='hist' 在CPU上通常也是最佳选择，可以保留或移除让其自动选择
             # current_params.pop('tree_method', None)
             logger.info(f"为 XGBoost 配置 CPU 支持: {current_params}")
         elif model_type.lower() == 'lightgbm':
             current_params.pop('device', None)
             logger.info(f"为 LightGBM 配置 CPU 支持: {current_params}")
    # elif device and 'xla' in str(device): ... # TPU 逻辑保持不变

    # 尝试加载预训练模型
    model_loaded = False
    if not force_retrain:
        model_loaded = model.load(element)
        if model_loaded:
            logger.info(f"已加载预训练的 {model_name} 模型")
        else:
            logger.info(f"未找到预训练的 {model_name} 模型或需要强制重新训练")

    # 如果模型未加载或需要强制重训，则进行训练
    if not model_loaded or force_retrain:
        if evaluate_only:
             logger.error(f"指定了 evaluate_only 但无法加载模型 {model_name} for {element}")
             return None

        # 加载完整的训练数据
        logger.info("加载完整训练数据...")
        X_train_full, y_train_full = load_processed_data(element, 'train', use_main_dataset, dataset_path)
        if X_train_full is None or y_train_full is None:
            logger.error(f"加载 {element} 的完整训练数据失败")
            return None
        
        # 检查数据量是否足够划分
        if len(X_train_full) < 2 or (val_size > 0 and len(X_train_full) * (1 - val_size) < 1) or (val_size > 0 and len(X_train_full) * val_size < 1) :
             logger.error(f"数据量过少 ({len(X_train_full)})，无法进行有效的训练/验证划分 (val_size={val_size})")
             return None

        # 划分训练集和验证集
        logger.info(f"划分训练集和验证集 (验证集比例: {val_size})")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size, random_state=42,
            stratify=None # 回归任务通常不使用 stratify
        )
        del X_train_full, y_train_full # 释放内存
        gc.collect()

        logger.info(f"训练集形状: {X_train.shape}, {y_train.shape}")
        logger.info(f"验证集形状: {X_val.shape}, {y_val.shape}")

        # 训练模型
        logger.info(f"开始训练 {model_name} 模型...")
        try:
            model.train(X_train, y_train, X_val, y_val)
            # 保存训练好的模型
            model.save(element)
        except Exception as e:
            logger.error(f"训练 {model_name} 模型时出错: {e}", exc_info=True)
            return None
        finally:
            # 确保清理内存
             del X_train, y_train, X_val, y_val
             gc.collect()
    elif evaluate_only and not model_loaded:
         logger.error(f"指定了 evaluate_only 但无法加载模型 {model_name} for {element}")
         return None


    # 评估模型
    # 加载测试数据
    logger.info("加载测试数据...")
    X_test, y_test = load_processed_data(element, 'test', use_main_dataset, dataset_path)
    if X_test is None or y_test is None:
        logger.error(f"加载 {element} 的测试数据失败")
        # 即使测试数据加载失败，如果模型已训练或加载，仍然返回一个标记，避免下面比较时出错
        return {'error': 'Failed to load test data'} if model.model else None 
    
    logger.info(f"测试集形状: {X_test.shape}, {y_test.shape}")

    # 检查模型是否真的被训练或加载了
    if model.model is None:
        logger.error(f"模型 {model_name} for {element} 未被成功训练或加载，无法进行评估。")
        return {'error': 'Model not available for prediction'}


    # 预测测试集 (如果内存允许，可以用 model.predict 替代)
    logger.info(f"使用 {model_name} 预测测试集...")
    try:
        # y_pred = model.predict(X_test) # 如果测试集不大
        y_pred = model.predict_in_batches(X_test) # 更安全的选择
    except Exception as e:
         logger.error(f"使用 {model_name} 预测时出错: {e}", exc_info=True)
         # 返回错误标记
         return {'error': 'Prediction failed'}

    # 评估模型
    logger.info(f"评估 {model_name} 模型...")
    results = evaluate_baseline_model(y_test, y_pred, model_name, element)
    
    return results 

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='基线模型训练与评估')
    parser.add_argument('--element', type=str, required=True,
                       help='元素名称，如CA_FE')
    parser.add_argument('--model', type=str, default='xgboost',
                      choices=['xgboost', 'lightgbm', 'both'],
                      help='模型类型，默认为xgboost')
    parser.add_argument('--val_size', type=float, default=0.2,
                       help='验证集比例，默认为0.2')
    parser.add_argument('--force_retrain', action='store_true',
                       help='强制重新训练，默认为False')
    parser.add_argument('--evaluate_only', action='store_true',
                       help='仅评估，不训练，默认为False')
    parser.add_argument('--clear_cache', action='store_true',
                       help='清除所有缓存，默认为False')
    parser.add_argument('--use_optimal_params', action='store_true',
                      help='使用主模型的最优超参数 (不推荐用于基线对比)')
    parser.add_argument('--optimal_params_file', type=str,
                      default='results/hyperopt/{element}/best_params.json',
                      help='最优超参数文件路径，{element}会被替换为元素名称')
    parser.add_argument('--use_main_dataset', action='store_true',
                      help='使用主模型的数据集')
    parser.add_argument('--dataset_path', type=str,
                      default='processed_data/{type}_dataset.npz',
                      help='数据集路径，{type}会被替换为train/val/test')
    parser.add_argument('--device', type=str, default='cpu', # 默认使用 CPU
                       help='计算设备，可选值: cpu, cuda')
    
    # 添加直接设置超参数的命令行参数 (保持不变)
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=None, help='权重衰减 (L2正则)')
    parser.add_argument('--max_depth', type=int, default=None, help='XGBoost/LightGBM的最大树深度')
    parser.add_argument('--n_estimators', type=int, default=None, help='XGBoost/LightGBM的树数量 (总轮数)')
    parser.add_argument('--subsample', type=float, default=None, help='XGBoost/LightGBM的样本抽样比例')
    parser.add_argument('--colsample_bytree', type=float, default=None, help='XGBoost/LightGBM的特征抽样比例')
    parser.add_argument('--early_stopping_rounds', type=int, default=None, help='XGBoost/LightGBM的早停轮数')
    parser.add_argument('--min_child_weight', type=int, default=None, help='XGBoost的最小子节点权重')
    parser.add_argument('--num_leaves', type=int, default=None, help='LightGBM的叶子节点数')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    
    args = parser.parse_args()
    
    # 处理缓存 (保持不变)
    if args.clear_cache:
        cache_manager.clear_cache()
        logger.info("已清除所有缓存")
    else:
        ask_clear_cache(cache_manager)
    
    # 创建必要的目录 (保持不变)
    os.makedirs(config.output_config['model_dir'], exist_ok=True)
    os.makedirs(config.output_config['results_dir'], exist_ok=True)
    
    # 处理设备选择 (移除TPU逻辑，简化为cpu/cuda)
    device = args.device.lower() if args.device else 'cpu'
    if device not in ['cpu', 'cuda']:
        logger.warning(f"不支持的设备 '{args.device}', 回退到 CPU.")
        device = 'cpu'

    # 设置通过命令行指定的超参数 (逻辑保持不变, 但注意 weight_decay 映射)
    xgb_params_update = {}
    lgb_params_update = {}
    
    if args.lr is not None:
        xgb_params_update['learning_rate'] = args.lr
        lgb_params_update['learning_rate'] = args.lr
    
    if args.weight_decay is not None:
        # XGBoost: reg_lambda 是 L2 正则化权重
        # LightGBM: lambda_l2 是 L2 正则化权重
        xgb_params_update['reg_lambda'] = args.weight_decay 
        lgb_params_update['lambda_l2'] = args.weight_decay 
    
    if args.max_depth is not None:
        xgb_params_update['max_depth'] = args.max_depth
        lgb_params_update['max_depth'] = args.max_depth
    
    if args.n_estimators is not None:
        xgb_params_update['n_estimators'] = args.n_estimators
        lgb_params_update['n_estimators'] = args.n_estimators
    
    if args.subsample is not None:
        xgb_params_update['subsample'] = args.subsample
        lgb_params_update['subsample'] = args.subsample
    
    if args.colsample_bytree is not None:
        xgb_params_update['colsample_bytree'] = args.colsample_bytree
        lgb_params_update['colsample_bytree'] = args.colsample_bytree
    
    if args.early_stopping_rounds is not None:
        xgb_params_update['early_stopping_rounds'] = args.early_stopping_rounds
        lgb_params_update['early_stopping_rounds'] = args.early_stopping_rounds
    
    if args.min_child_weight is not None:
        xgb_params_update['min_child_weight'] = args.min_child_weight
    
    if args.num_leaves is not None:
        lgb_params_update['num_leaves'] = args.num_leaves
    
    if args.seed is not None:
        xgb_params_update['seed'] = args.seed
        lgb_params_update['seed'] = args.seed
            
    # 加载最优超参数 (如果指定了 use_optimal_params 且未通过命令行指定参数)
    if args.use_optimal_params and not any([args.lr, args.weight_decay, args.max_depth, args.n_estimators, args.subsample, args.colsample_bytree, args.early_stopping_rounds, args.min_child_weight, args.num_leaves]):
        params_file = args.optimal_params_file.replace('{element}', args.element)
        if os.path.exists(params_file):
            with open(params_file, 'r') as f:
                optimal_params = json.load(f)
                logger.info(f"加载最优超参数: {optimal_params}")
                
                # 转换并更新参数 (注意: 这里的转换逻辑可能需要适配基线模型)
                if args.model.lower() in ['xgboost', 'both']:
                    translated_xgb = translate_params_to_xgboost(optimal_params)
                    logger.info(f"转换后的XGBoost参数: {translated_xgb}")
                    xgb_params_update.update(translated_xgb) 
                
                if args.model.lower() in ['lightgbm', 'both']:
                    translated_lgb = translate_params_to_lightgbm(optimal_params)
                    logger.info(f"转换后的LightGBM参数: {translated_lgb}")
                    lgb_params_update.update(translated_lgb) 
        else:
            logger.warning(f"找不到最优超参数文件: {params_file}")
    
    # 训练和评估模型
    xgb_results = None
    lgb_results = None

    if args.model.lower() in ['xgboost', 'both']:
        logger.info(f"开始处理 XGBoost 模型 for {args.element}")
        xgb_results = train_and_evaluate_baseline(
            args.element, 'xgboost',
            val_size=args.val_size,
            force_retrain=args.force_retrain,
            evaluate_only=args.evaluate_only,
            device=device,
            xgb_params=xgb_params_update, # 传递更新字典
            use_main_dataset=args.use_main_dataset,
            dataset_path=args.dataset_path
        )
        
    if args.model.lower() in ['lightgbm', 'both']:
        logger.info(f"开始处理 LightGBM 模型 for {args.element}")
        lgb_results = train_and_evaluate_baseline(
            args.element, 'lightgbm',
            val_size=args.val_size,
            force_retrain=args.force_retrain,
            evaluate_only=args.evaluate_only,
            device=device,
            lgb_params=lgb_params_update, # 传递更新字典
            use_main_dataset=args.use_main_dataset,
            dataset_path=args.dataset_path
        )
        
    # 比较两种模型 (仅在两个模型都成功运行后)
    if xgb_results and lgb_results and 'error' not in xgb_results and 'error' not in lgb_results:
        logger.info(f"模型比较 ({args.element}):")
        logger.info(f"  XGBoost RMSE: {xgb_results['rmse']:.6f}, MAE: {xgb_results['mae']:.6f}, R²: {xgb_results['r2']:.6f}")
        logger.info(f"  LightGBM RMSE: {lgb_results['rmse']:.6f}, MAE: {lgb_results['mae']:.6f}, R²: {lgb_results['r2']:.6f}")
            
        # 确定胜者
        winner = "XGBoost" if xgb_results['rmse'] < lgb_results['rmse'] else "LightGBM"
        loser = "LightGBM" if winner == "XGBoost" else "XGBoost"
        winner_rmse = min(xgb_results['rmse'], lgb_results['rmse'])
        loser_rmse = max(xgb_results['rmse'], lgb_results['rmse'])

        improvement = 0
        if loser_rmse > 1e-9: # 避免除零或极小值错误
             improvement = abs(loser_rmse - winner_rmse) / loser_rmse * 100
        logger.info(f"  胜者 (基于RMSE): {winner}，提升了 {improvement:.2f}%")
            
        # 比较结果可视化
        compare_models(args.element, xgb_results, lgb_results)
    elif xgb_results and 'error' in xgb_results:
         logger.error(f"XGBoost 处理失败: {xgb_results['error']}")
    elif lgb_results and 'error' in lgb_results:
         logger.error(f"LightGBM 处理失败: {lgb_results['error']}")

def compare_models(element, xgb_results, lgb_results):
    """
    比较XGBoost和LightGBM模型的性能
    """
    try:
        # 创建比较目录
        compare_dir = os.path.join(config.output_config['results_dir'], f'model_comparison_{element}')
        os.makedirs(compare_dir, exist_ok=True)
        
        # 准备比较数据
        metrics = ['mse', 'rmse', 'mae', 'r2', 'scatter']
        xgb_values = [xgb_results.get(m, float('nan')) for m in metrics] # 使用 .get 以防万一
        lgb_values = [lgb_results.get(m, float('nan')) for m in metrics]
        
        # 创建性能对比图
        plt.figure(figsize=(12, 8))
        
        # RMSE和MAE对比
        plt.subplot(2, 2, 1)
        models = ['XGBoost', 'LightGBM']
        x = np.arange(len(models))
        width = 0.35
        plt.bar(x - width/2, [xgb_values[1], lgb_values[1]], width, label='RMSE')
        plt.bar(x + width/2, [xgb_values[2], lgb_values[2]], width, label='MAE')
        plt.xlabel('模型')
        plt.ylabel('误差')
        plt.title('RMSE和MAE对比')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(True, axis='y')
        
        # R²对比
        plt.subplot(2, 2, 2)
        plt.bar(models, [xgb_values[3], lgb_values[3]], color=['blue', 'orange'])
        plt.xlabel('模型')
        plt.ylabel('R²')
        plt.title('R²对比')
        # 动态调整Y轴范围以更好地显示R2，特别是当它是负数时
        min_r2 = min(xgb_values[3], lgb_values[3], 0) - 0.1
        max_r2 = max(xgb_values[3], lgb_values[3], 0) + 0.1
        plt.ylim(min_r2, max_r2)
        plt.grid(True, axis='y')
        
        # 散度对比
        plt.subplot(2, 2, 3)
        plt.bar(models, [xgb_values[4], lgb_values[4]], color=['blue', 'orange'])
        plt.xlabel('模型')
        plt.ylabel('散度')
        plt.title('散度对比')
        plt.grid(True, axis='y')
        
        # 雷达图对比
        plt.subplot(2, 2, 4, polar=True)
        # 指标: rmse, mae, scatter, 1-r2 (值越小越好)
        radar_metrics = ['RMSE', 'MAE', 'Scatter', '1-R²']
        # 获取数据，处理可能的 NaN 或 None
        xgb_r2 = xgb_values[3] if xgb_values[3] is not None and not np.isnan(xgb_values[3]) else -1 # 惩罚无效R2
        lgb_r2 = lgb_values[3] if lgb_values[3] is not None and not np.isnan(lgb_values[3]) else -1

        xgb_radar_raw = [
            xgb_values[1] if not np.isnan(xgb_values[1]) else float('inf'),
            xgb_values[2] if not np.isnan(xgb_values[2]) else float('inf'),
            xgb_values[4] if not np.isnan(xgb_values[4]) else float('inf'),
            1 - xgb_r2 # 值越小越好
        ]
        lgb_radar_raw = [
            lgb_values[1] if not np.isnan(lgb_values[1]) else float('inf'),
            lgb_values[2] if not np.isnan(lgb_values[2]) else float('inf'),
            lgb_values[4] if not np.isnan(lgb_values[4]) else float('inf'),
            1 - lgb_r2
        ]

        # 找出每个指标的最大值用于归一化 (确保非负)
        max_radar_values = [max(xgb_radar_raw[i], lgb_radar_raw[i], 0.00001) for i in range(len(radar_metrics))]

        # 归一化 (值越小越好，所以直接除以最大值)
        normalized_xgb = [xgb_radar_raw[i] / max_radar_values[i] for i in range(len(radar_metrics))]
        normalized_lgb = [lgb_radar_raw[i] / max_radar_values[i] for i in range(len(radar_metrics))]

        # 防止值超出1 (可能因为最大值计算方式)
        normalized_xgb = [min(v, 1.0) for v in normalized_xgb]
        normalized_lgb = [min(v, 1.0) for v in normalized_lgb]


        # 完成雷达图的圆形
        normalized_xgb += normalized_xgb[:1]
        normalized_lgb += normalized_lgb[:1]

        # 设置角度
        angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合多边形

        # 绘制雷达图
        ax = plt.gca()
        ax.plot(angles, normalized_xgb, 'o-', linewidth=2, label='XGBoost')
        ax.fill(angles, normalized_xgb, 'blue', alpha=0.25)
        ax.plot(angles, normalized_lgb, 'o-', linewidth=2, label='LightGBM')
        ax.fill(angles, normalized_lgb, 'red', alpha=0.25)

        # 设置雷达图标签和刻度
        ax.set_thetagrids(np.degrees(angles[:-1]), radar_metrics)
        ax.set_yticks(np.linspace(0, 1, 6)) # 0, 0.2, ..., 1.0
        ax.set_ylim(0, 1.0) # Y轴范围设为0到1.0
        plt.title('归一化性能雷达图 (值越接近0越好)')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout()

        # 保存比较图表
        compare_plot_path = os.path.join(compare_dir, f'model_comparison_{element}.png')
        plt.savefig(compare_plot_path)
        plt.close()
        
        # 保存比较摘要
        compare_summary_path = os.path.join(compare_dir, f'model_comparison_{element}_summary.txt')
        with open(compare_summary_path, 'w') as f:
            f.write(f"XGBoost和LightGBM在{element}上的性能比较\n")
            f.write("=" * 50 + "\n")
            f.write("\nXGBoost性能指标:\n")
            f.write(f"  MSE: {xgb_values[0]:.6f}\n")
            f.write(f"  RMSE: {xgb_values[1]:.6f}\n")
            f.write(f"  MAE: {xgb_values[2]:.6f}\n")
            f.write(f"  R²: {xgb_values[3]:.6f}\n")
            f.write(f"  散度: {xgb_values[4]:.6f}\n")
            
            f.write("\nLightGBM性能指标:\n")
            f.write(f"  MSE: {lgb_values[0]:.6f}\n")
            f.write(f"  RMSE: {lgb_values[1]:.6f}\n")
            f.write(f"  MAE: {lgb_values[2]:.6f}\n")
            f.write(f"  R²: {lgb_values[3]:.6f}\n")
            f.write(f"  散度: {lgb_values[4]:.6f}\n")
            
            # 计算性能提升 (基于RMSE)
            winner = "XGBoost" if xgb_values[1] < lgb_values[1] else "LightGBM"
            loser = "LightGBM" if winner == "XGBoost" else "XGBoost"
            winner_rmse = min(xgb_values[1], lgb_values[1])
            loser_rmse = max(xgb_values[1], lgb_values[1])
            improvement = 0
            if loser_rmse > 1e-9: # 避免除零或极小值错误
                 improvement = abs(loser_rmse - winner_rmse) / loser_rmse * 100

            f.write(f"\n总结: {winner}在RMSE指标上表现更好，提升了{improvement:.2f}%。\n")
        
        logger.info(f"已生成模型比较结果，保存在: {compare_dir}")
        
    except Exception as e:
        logger.error(f"生成模型比较结果时出错: {e}", exc_info=True)

def translate_params_to_xgboost(optimal_params):
    """
    将主模型参数转换为XGBoost参数 (需要根据实际情况调整映射关系)
    """
    params = {}
    params['learning_rate'] = optimal_params.get('lr', 0.01) # 直接映射学习率
    # params['max_depth'] = optimal_params.get('some_nn_param', 6) # 神经网络参数可能无法直接映射到树深度
    params['reg_lambda'] = optimal_params.get('weight_decay', 0.0) # 映射 L2 正则化
    # params['reg_alpha'] = optimal_params.get('l1_decay', 0.0) # 映射 L1 正则化 (如果需要)
    # 其他参数如 n_estimators, early_stopping_rounds 最好使用基线模型自己的默认值或单独调整
    logger.warning("translate_params_to_xgboost 使用的映射可能不准确")
    return params

def translate_params_to_lightgbm(optimal_params):
    """
    将主模型参数转换为LightGBM参数 (需要根据实际情况调整映射关系)
    """
    params = {}
    params['learning_rate'] = optimal_params.get('lr', 0.01) # 直接映射学习率
    # params['max_depth'] = optimal_params.get('some_nn_param', -1)
    # params['num_leaves'] = optimal_params.get('other_nn_param', 31)
    params['lambda_l2'] = optimal_params.get('weight_decay', 0.0) # 映射 L2 正则化
    # params['lambda_l1'] = optimal_params.get('l1_decay', 0.0) # 映射 L1 正则化
    logger.warning("translate_params_to_lightgbm 使用的映射可能不准确")
    return params

if __name__ == '__main__':
    main() 