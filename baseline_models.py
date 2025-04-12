#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基线模型模块：使用XGBoost和LightGBM实现基线回归模型
实现分批处理机制以处理大规模数据集
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
import joblib
import time
from tqdm import tqdm
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
    XGBoost模型
    特点：
    1. 优秀的特征处理能力
    2. 内置正则化，防止过拟合
    3. 支持并行计算
    4. 处理高维特征的能力强
    """
    def __init__(self, config=None, params_update=None):
        if config is None:
            config = Config  # 使用直接导入的Config而非config.CONFIG
            
        self.params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.01,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'seed': 42
        }
        if params_update:
            self.params.update(params_update)
            logger.info(f"XGBoostModel 使用更新后的参数初始化: {params_update}")
            
        self.model = None
        self.trained_batches = []
        
    def train_on_batch(self, X_train, y_train, X_val, y_val, batch_id=None):
        """
        在单个批次上训练XGBoost模型
        
        参数:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            batch_id: 批次ID
        
        返回:
            训练结果
        """
        logger.info(f"在批次 {batch_id} 上训练XGBoost模型")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # 使用早停机制训练
        evals_result = {}
        model = xgb.train(
            self.params,
            dtrain,
            evals=[(dtrain, 'train'), (dval, 'val')],
            evals_result=evals_result,
            verbose_eval=100
        )
        
        # 获取验证集上的最佳性能
        best_score = min(evals_result['val']['rmse'])
        
        # 记录已训练的批次
        if batch_id is not None and batch_id not in self.trained_batches:
            self.trained_batches.append(batch_id)
        
        return {
            'model': model,
            'batch_id': batch_id,
            'validation_score': best_score
        }
    
    def train_in_batches(self, X, y, batch_size=Config.BASELINE_BATCH_SIZE, batches_per_round=Config.BASELINE_BATCHES_PER_ROUND, val_size=0.2, element=None):
        """
        分批增量训练模型
        """
        # 计算批次索引
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # 确定要处理的批次
        remaining_batches = [i for i in range(n_batches) if i not in self.trained_batches]
        if not remaining_batches:
            logger.info("所有批次已训练完成")
            return self.model
        
        # 仅处理指定数量的批次
        batch_indices = remaining_batches[:batches_per_round]
        logger.info(f"将处理 {len(batch_indices)}/{n_batches} 个批次")
        
        # 使用进度管理器
        with ProgressManager(len(batch_indices), desc=f"{self.__class__.__name__}训练 ({element})") as progress:
            for batch_id in batch_indices:
                # 获取批次索引
                start_idx = batch_id * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                # 提取批次数据
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]
                
                # 划分训练集和验证集
                X_train, X_val, y_train, y_val = train_test_split(
                    X_batch, y_batch, test_size=val_size, random_state=42
                )
                
                # 增量训练
                if isinstance(self, XGBoostModel):
                    dtrain = xgb.DMatrix(X_train, label=y_train)
                    dval = xgb.DMatrix(X_val, label=y_val)
                    self.model = xgb.train(
                        self.params,
                        dtrain,
                        evals=[(dtrain, 'train'), (dval, 'val')],
                        xgb_model=self.model,  # 使用现有模型继续训练
                        evals_result={},
                        verbose_eval=100
                    )
                elif isinstance(self, LightGBMModel):
                    train_data = lgb.Dataset(X_train, label=y_train)
                    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                    self.model = lgb.train(
                        self.params,
                        train_data,
                        valid_sets=[train_data, val_data],
                        valid_names=['train', 'val'],
                        init_model=self.model,  # 使用现有模型继续训练
                        verbose_eval=100
                    )
                
                # 记录已训练的批次
                self.trained_batches.append(batch_id)
                
                # 更新进度
                progress.update(1)
                
                # 释放内存
                del X_batch, y_batch, X_train, X_val, y_train, y_val
                gc.collect()
        
        return self.model
    
    def _generate_batch_results(self, model, batch_id, X_val, y_val, element, batch_results_dir):
        """
        为每个批次生成评估结果和可视化
        
        参数:
            model: 当前批次训练的模型
            batch_id: 批次ID
            X_val: 验证特征
            y_val: 验证标签
            element: 元素名称
            batch_results_dir: 批次结果保存目录
        """
        try:
            # 使用当前批次模型进行预测
            dval = xgb.DMatrix(X_val)
            y_pred = model.predict(dval)
            
            # 计算评估指标
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            scatter = np.std(y_pred - y_val)
            
            # 保存批次评估指标
            batch_summary_path = os.path.join(batch_results_dir, f'batch_{batch_id}_summary.txt')
            with open(batch_summary_path, 'w') as f:
                f.write(f"XGBoost 批次 {batch_id} 在 {element} 上的评估结果\n")
                f.write("=" * 50 + "\n")
                f.write(f"MSE: {mse:.6f}\n")
                f.write(f"RMSE: {rmse:.6f}\n")
                f.write(f"MAE: {mae:.6f}\n")
                f.write(f"R²: {r2:.6f}\n")
                f.write(f"散度: {scatter:.6f}\n")
            
            # 保存预测结果
            batch_results_path = os.path.join(batch_results_dir, f'batch_{batch_id}_results.csv')
            pd.DataFrame({
                'true': y_val,
                'pred': y_pred,
                'error': y_pred - y_val
            }).to_csv(batch_results_path, index=False)
            
            # 生成散点图对比真实值和预测值
            plt.figure(figsize=(10, 6))
            plt.scatter(y_val, y_pred, alpha=0.5)
            plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            plt.title(f'XGBoost 批次 {batch_id} 预测 vs 真实值 (RMSE: {rmse:.4f})')
            plt.grid(True)
            plt.tight_layout()
            
            # 保存图表
            batch_plot_path = os.path.join(batch_results_dir, f'batch_{batch_id}_plot.png')
            plt.savefig(batch_plot_path)
            plt.close()
            
            # 保存批次信息到批次追踪文件
            batch_tracking_path = os.path.join(batch_results_dir, 'batch_tracking.csv')
            
            # 如果追踪文件不存在，创建一个新的
            if not os.path.exists(batch_tracking_path):
                batch_df = pd.DataFrame(columns=['batch_id', 'mse', 'rmse', 'mae', 'r2', 'scatter', 'timestamp'])
            else:
                batch_df = pd.read_csv(batch_tracking_path)
            
            # 添加新批次的结果
            new_row = pd.DataFrame({
                'batch_id': [batch_id],
                'mse': [mse],
                'rmse': [rmse],
                'mae': [mae],
                'r2': [r2],
                'scatter': [scatter],
                'timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')]
            })
            
            batch_df = pd.concat([batch_df, new_row], ignore_index=True)
            batch_df.to_csv(batch_tracking_path, index=False)
            
            # 生成批次进度趋势图
            if len(batch_df) > 1:
                plt.figure(figsize=(12, 8))
                
                plt.subplot(2, 2, 1)
                plt.plot(batch_df['batch_id'], batch_df['rmse'], 'o-')
                plt.xlabel('批次ID')
                plt.ylabel('RMSE')
                plt.title('RMSE趋势')
                plt.grid(True)
                
                plt.subplot(2, 2, 2)
                plt.plot(batch_df['batch_id'], batch_df['mae'], 'o-')
                plt.xlabel('批次ID')
                plt.ylabel('MAE')
                plt.title('MAE趋势')
                plt.grid(True)
                
                plt.subplot(2, 2, 3)
                plt.plot(batch_df['batch_id'], batch_df['r2'], 'o-')
                plt.xlabel('批次ID')
                plt.ylabel('R²')
                plt.title('R²趋势')
                plt.grid(True)
                
                plt.subplot(2, 2, 4)
                plt.plot(batch_df['batch_id'], batch_df['scatter'], 'o-')
                plt.xlabel('批次ID')
                plt.ylabel('散度')
                plt.title('散度趋势')
                plt.grid(True)
                
                plt.tight_layout()
                
                # 保存趋势图
                trend_plot_path = os.path.join(batch_results_dir, 'batch_trends.png')
                plt.savefig(trend_plot_path)
                plt.close()
            
            logger.info(f"成功生成批次 {batch_id} 的结果和可视化")
            
        except Exception as e:
            logger.error(f"生成批次 {batch_id} 结果时出错: {e}")
    
    def _generate_final_results(self, element):
        """
        生成并更新最终的评估结果
        
        参数:
            element: 元素名称
        """
        try:
            # 检查是否有批次结果
            batch_results_dir = os.path.join(config.output_config['results_dir'], f'xgboost_{element}_batch_results')
            batch_tracking_path = os.path.join(batch_results_dir, 'batch_tracking.csv')
            
            if not os.path.exists(batch_tracking_path):
                logger.warning("找不到批次追踪文件，无法生成最终结果")
                return
            
            # 加载批次追踪数据
            batch_df = pd.read_csv(batch_tracking_path)
            
            if len(batch_df) == 0:
                logger.warning("批次追踪文件为空，无法生成最终结果")
                return
            
            # 计算平均指标
            avg_mse = batch_df['mse'].mean()
            avg_rmse = batch_df['rmse'].mean()
            avg_mae = batch_df['mae'].mean()
            avg_r2 = batch_df['r2'].mean()
            avg_scatter = batch_df['scatter'].mean()
            
            # 找出最佳批次
            best_batch_id = batch_df.loc[batch_df['rmse'].idxmin(), 'batch_id']
            best_rmse = batch_df['rmse'].min()
            
            # 保存最终评估摘要
            final_summary_path = os.path.join(config.output_config['results_dir'], f'xgboost_{element}_final_summary.txt')
            with open(final_summary_path, 'w') as f:
                f.write(f"XGBoost 在 {element} 上的最终评估结果\n")
                f.write("=" * 50 + "\n")
                f.write(f"已处理批次数: {len(batch_df)}\n")
                f.write(f"平均 MSE: {avg_mse:.6f}\n")
                f.write(f"平均 RMSE: {avg_rmse:.6f}\n")
                f.write(f"平均 MAE: {avg_mae:.6f}\n")
                f.write(f"平均 R²: {avg_r2:.6f}\n")
                f.write(f"平均散度: {avg_scatter:.6f}\n")
                f.write("\n")
                f.write(f"最佳批次: {best_batch_id} (RMSE: {best_rmse:.6f})\n")
                f.write("\n")
                f.write("批次处理时间:\n")
                for _, row in batch_df.iterrows():
                    f.write(f"  批次 {int(row['batch_id'])}: {row['timestamp']}\n")
            
            logger.info(f"已成功生成并更新最终结果，平均RMSE: {avg_rmse:.6f}，最佳批次: {best_batch_id}")
            
        except Exception as e:
            logger.error(f"生成最终结果时出错: {e}")
    
    def predict(self, X):
        """
        预测
        """
        if self.model is None:
            raise ValueError("模型未训练")
            
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def predict_in_batches(self, X, batch_size=1000):
        """
        分批预测
        
        参数:
            X: 特征数据
            batch_size: 每批大小
            
        返回:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型未训练")
        
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        predictions = np.zeros(n_samples)
        
        # 使用进度管理器
        with ProgressManager(n_batches, desc="XGBoost预测") as progress:
            for i in range(n_batches):
                # 获取批次索引
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                # 提取批次数据
                X_batch = X[start_idx:end_idx]
                
                # 预测
                dtest = xgb.DMatrix(X_batch)
                predictions[start_idx:end_idx] = self.model.predict(dtest)
                
                # 更新进度
                progress.update(1)
                
                # 释放内存
                del X_batch
                gc.collect()
        
        return predictions
    
    def save(self, element):
        """
        保存模型
        
        参数:
            element: 元素名称
        """
        if self.model is None:
            raise ValueError("模型未训练")
            
        os.makedirs(config.output_config['model_dir'], exist_ok=True)
        model_path = os.path.join(config.output_config['model_dir'], f'xgboost_{element}.json')
        self.model.save_model(model_path)
        
        # 保存训练状态
        state_path = os.path.join(config.output_config['model_dir'], f'xgboost_{element}_state.pkl')
        with open(state_path, 'wb') as f:
            pickle.dump({
                'trained_batches': self.trained_batches
            }, f)
        
        logger.info(f"XGBoost模型已保存: {model_path}")
    
    def load(self, element):
        """
        加载模型
        
        参数:
            element: 元素名称
        """
        model_path = os.path.join(config.output_config['model_dir'], f'xgboost_{element}.json')
        if not os.path.exists(model_path):
            logger.warning(f"找不到模型文件: {model_path}")
            return False
        
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        # 加载训练状态
        state_path = os.path.join(config.output_config['model_dir'], f'xgboost_{element}_state.pkl')
        if os.path.exists(state_path):
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
                self.trained_batches = state.get('trained_batches', [])
        
        logger.info(f"XGBoost模型已加载: {model_path}")
        return True

class LightGBMModel:
    """
    LightGBM模型
    特点：
    1. 更快的训练速度
    2. 更低的内存消耗
    3. 更好的准确性
    4. 支持直接处理类别特征
    """
    def __init__(self, config=None, params_update=None):
        if config is None:
            config = Config  # 使用直接导入的Config而非config.CONFIG
            
        self.params = {
            'objective': 'regression',
            'metric': 'mse',
            'learning_rate': 0.01,
            'num_leaves': 31,
            'max_depth': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'seed': 42
        }
        if params_update:
            self.params.update(params_update)
            logger.info(f"LightGBMModel 使用更新后的参数初始化: {params_update}")
            
        self.model = None
        self.trained_batches = []
    
    def train_on_batch(self, X_train, y_train, X_val, y_val, batch_id=None):
        """
        在单个批次上训练LightGBM模型
        
        参数:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            batch_id: 批次ID
        
        返回:
            训练结果
        """
        logger.info(f"在批次 {batch_id} 上训练LightGBM模型")
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 使用早停机制训练
        model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            verbose_eval=100
        )
        
        # 获取验证集上的最佳性能
        best_score = min(evals_result['val']['l2'])
        
        # 记录已训练的批次
        if batch_id is not None and batch_id not in self.trained_batches:
            self.trained_batches.append(batch_id)
        
        return {
            'model': model,
            'batch_id': batch_id,
            'validation_score': best_score
        }
    
    def train_in_batches(self, X, y, batch_size=Config.BASELINE_BATCH_SIZE, batches_per_round=Config.BASELINE_BATCHES_PER_ROUND, val_size=0.2, element=None):
        """
        分批增量训练模型
        """
        # 计算批次索引
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # 确定要处理的批次
        remaining_batches = [i for i in range(n_batches) if i not in self.trained_batches]
        if not remaining_batches:
            logger.info("所有批次已训练完成")
            return self.model
        
        # 仅处理指定数量的批次
        batch_indices = remaining_batches[:batches_per_round]
        logger.info(f"将处理 {len(batch_indices)}/{n_batches} 个批次")
        
        # 使用进度管理器
        with ProgressManager(len(batch_indices), desc=f"{self.__class__.__name__}训练 ({element})") as progress:
            for batch_id in batch_indices:
                # 获取批次索引
                start_idx = batch_id * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                # 提取批次数据
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]
                
                # 划分训练集和验证集
                X_train, X_val, y_train, y_val = train_test_split(
                    X_batch, y_batch, test_size=val_size, random_state=42
                )
                
                # 增量训练
                if isinstance(self, XGBoostModel):
                    dtrain = xgb.DMatrix(X_train, label=y_train)
                    dval = xgb.DMatrix(X_val, label=y_val)
                    self.model = xgb.train(
                        self.params,
                        dtrain,
                        evals=[(dtrain, 'train'), (dval, 'val')],
                        xgb_model=self.model,  # 使用现有模型继续训练
                        evals_result={},
                        verbose_eval=100
                    )
                elif isinstance(self, LightGBMModel):
                    train_data = lgb.Dataset(X_train, label=y_train)
                    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                    self.model = lgb.train(
                        self.params,
                        train_data,
                        valid_sets=[train_data, val_data],
                        valid_names=['train', 'val'],
                        init_model=self.model,  # 使用现有模型继续训练
                        verbose_eval=100
                    )
                
                # 记录已训练的批次
                self.trained_batches.append(batch_id)
                
                # 更新进度
                progress.update(1)
                
                # 释放内存
                del X_batch, y_batch, X_train, X_val, y_train, y_val
                gc.collect()
        
        return self.model
    
    def _generate_batch_results(self, model, batch_id, X_val, y_val, element, batch_results_dir):
        """
        为每个批次生成评估结果和可视化
        
        参数:
            model: 当前批次训练的模型
            batch_id: 批次ID
            X_val: 验证特征
            y_val: 验证标签
            element: 元素名称
            batch_results_dir: 批次结果保存目录
        """
        try:
            # 使用当前批次模型进行预测
            y_pred = model.predict(X_val)
            
            # 计算评估指标
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            scatter = np.std(y_pred - y_val)
            
            # 保存批次评估指标
            batch_summary_path = os.path.join(batch_results_dir, f'batch_{batch_id}_summary.txt')
            with open(batch_summary_path, 'w') as f:
                f.write(f"LightGBM 批次 {batch_id} 在 {element} 上的评估结果\n")
                f.write("=" * 50 + "\n")
                f.write(f"MSE: {mse:.6f}\n")
                f.write(f"RMSE: {rmse:.6f}\n")
                f.write(f"MAE: {mae:.6f}\n")
                f.write(f"R²: {r2:.6f}\n")
                f.write(f"散度: {scatter:.6f}\n")
            
            # 保存预测结果
            batch_results_path = os.path.join(batch_results_dir, f'batch_{batch_id}_results.csv')
            pd.DataFrame({
                'true': y_val,
                'pred': y_pred,
                'error': y_pred - y_val
            }).to_csv(batch_results_path, index=False)
            
            # 生成散点图对比真实值和预测值
            plt.figure(figsize=(10, 6))
            plt.scatter(y_val, y_pred, alpha=0.5)
            plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            plt.title(f'LightGBM 批次 {batch_id} 预测 vs 真实值 (RMSE: {rmse:.4f})')
            plt.grid(True)
            plt.tight_layout()
            
            # 保存图表
            batch_plot_path = os.path.join(batch_results_dir, f'batch_{batch_id}_plot.png')
            plt.savefig(batch_plot_path)
            plt.close()
            
            # 保存批次信息到批次追踪文件
            batch_tracking_path = os.path.join(batch_results_dir, 'batch_tracking.csv')
            
            # 如果追踪文件不存在，创建一个新的
            if not os.path.exists(batch_tracking_path):
                batch_df = pd.DataFrame(columns=['batch_id', 'mse', 'rmse', 'mae', 'r2', 'scatter', 'timestamp'])
            else:
                batch_df = pd.read_csv(batch_tracking_path)
            
            # 添加新批次的结果
            new_row = pd.DataFrame({
                'batch_id': [batch_id],
                'mse': [mse],
                'rmse': [rmse],
                'mae': [mae],
                'r2': [r2],
                'scatter': [scatter],
                'timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')]
            })
            
            batch_df = pd.concat([batch_df, new_row], ignore_index=True)
            batch_df.to_csv(batch_tracking_path, index=False)
            
            # 生成批次进度趋势图
            if len(batch_df) > 1:
                plt.figure(figsize=(12, 8))
                
                plt.subplot(2, 2, 1)
                plt.plot(batch_df['batch_id'], batch_df['rmse'], 'o-')
                plt.xlabel('批次ID')
                plt.ylabel('RMSE')
                plt.title('RMSE趋势')
                plt.grid(True)
                
                plt.subplot(2, 2, 2)
                plt.plot(batch_df['batch_id'], batch_df['mae'], 'o-')
                plt.xlabel('批次ID')
                plt.ylabel('MAE')
                plt.title('MAE趋势')
                plt.grid(True)
                
                plt.subplot(2, 2, 3)
                plt.plot(batch_df['batch_id'], batch_df['r2'], 'o-')
                plt.xlabel('批次ID')
                plt.ylabel('R²')
                plt.title('R²趋势')
                plt.grid(True)
                
                plt.subplot(2, 2, 4)
                plt.plot(batch_df['batch_id'], batch_df['scatter'], 'o-')
                plt.xlabel('批次ID')
                plt.ylabel('散度')
                plt.title('散度趋势')
                plt.grid(True)
                
                plt.tight_layout()
                
                # 保存趋势图
                trend_plot_path = os.path.join(batch_results_dir, 'batch_trends.png')
                plt.savefig(trend_plot_path)
                plt.close()
            
            logger.info(f"成功生成批次 {batch_id} 的结果和可视化")
            
        except Exception as e:
            logger.error(f"生成批次 {batch_id} 结果时出错: {e}")
    
    def _generate_final_results(self, element):
        """
        生成并更新最终的评估结果
        
        参数:
            element: 元素名称
        """
        try:
            # 检查是否有批次结果
            batch_results_dir = os.path.join(config.output_config['results_dir'], f'lightgbm_{element}_batch_results')
            batch_tracking_path = os.path.join(batch_results_dir, 'batch_tracking.csv')
            
            if not os.path.exists(batch_tracking_path):
                logger.warning("找不到批次追踪文件，无法生成最终结果")
                return
            
            # 加载批次追踪数据
            batch_df = pd.read_csv(batch_tracking_path)
            
            if len(batch_df) == 0:
                logger.warning("批次追踪文件为空，无法生成最终结果")
                return
            
            # 计算平均指标
            avg_mse = batch_df['mse'].mean()
            avg_rmse = batch_df['rmse'].mean()
            avg_mae = batch_df['mae'].mean()
            avg_r2 = batch_df['r2'].mean()
            avg_scatter = batch_df['scatter'].mean()
            
            # 找出最佳批次
            best_batch_id = batch_df.loc[batch_df['rmse'].idxmin(), 'batch_id']
            best_rmse = batch_df['rmse'].min()
            
            # 保存最终评估摘要
            final_summary_path = os.path.join(config.output_config['results_dir'], f'lightgbm_{element}_final_summary.txt')
            with open(final_summary_path, 'w') as f:
                f.write(f"LightGBM 在 {element} 上的最终评估结果\n")
                f.write("=" * 50 + "\n")
                f.write(f"已处理批次数: {len(batch_df)}\n")
                f.write(f"平均 MSE: {avg_mse:.6f}\n")
                f.write(f"平均 RMSE: {avg_rmse:.6f}\n")
                f.write(f"平均 MAE: {avg_mae:.6f}\n")
                f.write(f"平均 R²: {avg_r2:.6f}\n")
                f.write(f"平均散度: {avg_scatter:.6f}\n")
                f.write("\n")
                f.write(f"最佳批次: {best_batch_id} (RMSE: {best_rmse:.6f})\n")
                f.write("\n")
                f.write("批次处理时间:\n")
                for _, row in batch_df.iterrows():
                    f.write(f"  批次 {int(row['batch_id'])}: {row['timestamp']}\n")
            
            logger.info(f"已成功生成并更新最终结果，平均RMSE: {avg_rmse:.6f}，最佳批次: {best_batch_id}")
            
        except Exception as e:
            logger.error(f"生成最终结果时出错: {e}")
    
    def predict(self, X):
        """
        预测
        """
        if self.model is None:
            raise ValueError("模型未训练")
            
        return self.model.predict(X)
    
    def predict_in_batches(self, X, batch_size=1000):
        """
        分批预测
        
        参数:
            X: 特征数据
            batch_size: 每批大小
            
        返回:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型未训练")
        
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        predictions = np.zeros(n_samples)
        
        # 使用进度管理器
        with ProgressManager(n_batches, desc="LightGBM预测") as progress:
            for i in range(n_batches):
                # 获取批次索引
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                # 提取批次数据
                X_batch = X[start_idx:end_idx]
                
                # 预测
                predictions[start_idx:end_idx] = self.model.predict(X_batch)
                
                # 更新进度
                progress.update(1)
                
                # 释放内存
                del X_batch
                gc.collect()
        
        return predictions
    
    def save(self, element):
        """
        保存模型
        
        参数:
            element: 元素名称
        """
        if self.model is None:
            raise ValueError("模型未训练")
            
        os.makedirs(config.output_config['model_dir'], exist_ok=True)
        model_path = os.path.join(config.output_config['model_dir'], f'lightgbm_{element}.txt')
        self.model.save_model(model_path)
        
        # 保存训练状态
        state_path = os.path.join(config.output_config['model_dir'], f'lightgbm_{element}_state.pkl')
        with open(state_path, 'wb') as f:
            pickle.dump({
                'trained_batches': self.trained_batches
            }, f)
        
        logger.info(f"LightGBM模型已保存: {model_path}")
    
    def load(self, element):
        """
        加载模型
        
        参数:
            element: 元素名称
        """
        model_path = os.path.join(config.output_config['model_dir'], f'lightgbm_{element}.txt')
        if not os.path.exists(model_path):
            logger.warning(f"找不到模型文件: {model_path}")
            return False
        
        self.model = lgb.Booster(model_file=model_path)
        
        # 加载训练状态
        state_path = os.path.join(config.output_config['model_dir'], f'lightgbm_{element}_state.pkl')
        if os.path.exists(state_path):
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
                self.trained_batches = state.get('trained_batches', [])
        
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
                               batch_size=Config.BASELINE_BATCH_SIZE, batches_per_round=Config.BASELINE_BATCHES_PER_ROUND, val_size=0.2,
                               force_retrain=False, evaluate_only=False, device=None,
                               xgb_params=None, lgb_params=None):
    """添加device参数支持不同计算设备"""
    # 创建模型
    model_params_update = None
    if model_type.lower() == 'xgboost':
        model_params_update = xgb_params
        model = XGBoostModel(params_update=model_params_update)
        model_name = 'xgboost'
    elif model_type.lower() == 'lightgbm':
        model_params_update = lgb_params
        model = LightGBMModel(params_update=model_params_update)
        model_name = 'lightgbm'
    else:
        logger.error(f"不支持的模型类型: {model_type}")
        return None
    
    # XGBoost和LightGBM配置中添加设备支持 - 将设备配置移到模型创建之后
    if device and 'cuda' in str(device):
        # GPU支持
        if model_type.lower() == 'xgboost':
            model.params.update({'tree_method': 'hist', 'device': 'cuda:0'}) 
            logger.info(f"为 XGBoost 配置 GPU 支持: {{'tree_method': 'hist', 'device': 'cuda:0'}}")
        elif model_type.lower() == 'lightgbm':
            model.params.update({'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0})
            logger.info(f"为 LightGBM 配置 GPU 支持: {{'device': 'gpu'}}")
    elif device and 'xla' in str(device):
        # TPU目前不直接支持这些库，可以考虑用TensorFlow版本替代
        logger.warning("TPU不直接支持XGBoost/LightGBM，将使用CPU")
    
    # 如果仅评估且可以加载模型，则加载模型
    if evaluate_only and not force_retrain:
        if model.load(element):
            logger.info(f"已加载 {model_name} 模型")
        else:
            logger.warning(f"无法加载 {model_name} 模型，将训练新模型")
            evaluate_only = False
    
    # 训练模型
    if not evaluate_only:
        # 加载训练数据
        X_train, y_train = load_processed_data(element, 'train')
        if X_train is None or y_train is None:
            logger.error(f"加载 {element} 的训练数据失败")
            return None
        
        # 分批训练模型
        logger.info(f"分批训练 {model_name} 模型")
        start_time = time.time()
        model.train_in_batches(
            X_train, y_train, 
            batch_size=batch_size, 
            batches_per_round=batches_per_round,
            val_size=val_size,
            element=element
        )
        elapsed_time = time.time() - start_time
        logger.info(f"训练 {model_name} 耗时: {elapsed_time:.2f}秒")
        
        # 保存模型
        model.save(element)
    
    # 评估模型
    # 加载测试数据
    X_test, y_test = load_processed_data(element, 'test')
    if X_test is None or y_test is None:
        logger.error(f"加载 {element} 的测试数据失败")
        return None
    
    # 分批预测
    logger.info(f"分批预测测试集")
    y_pred = model.predict_in_batches(X_test, batch_size)
    
    # 评估模型
    results = evaluate_baseline_model(y_test, y_pred, model_name, element)
    
    # 显示批次处理结果的位置
    batch_results_dir = os.path.join(config.output_config['results_dir'], f'{model_name}_{element}_batch_results')
    if os.path.exists(batch_results_dir):
        logger.info(f"批次处理结果保存在: {batch_results_dir}")
        final_summary_path = os.path.join(config.output_config['results_dir'], f'{model_name}_{element}_final_summary.txt')
        if os.path.exists(final_summary_path):
            with open(final_summary_path, 'r') as f:
                summary_content = f.read()
            logger.info(f"最终汇总结果:\n{summary_content}")
    
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
    parser.add_argument('--batch_size', type=int, default=Config.BASELINE_BATCH_SIZE,
                       help='每批大小，默认为1000')
    parser.add_argument('--batches_per_round', type=int, default=Config.BASELINE_BATCHES_PER_ROUND,
                       help='每轮处理的批次数，默认为5')
    parser.add_argument('--val_size', type=float, default=0.2,
                       help='验证集比例，默认为0.2')
    parser.add_argument('--force_retrain', action='store_true',
                       help='强制重新训练，默认为False')
    parser.add_argument('--evaluate_only', action='store_true',
                       help='仅评估，不训练，默认为False')
    parser.add_argument('--clear_cache', action='store_true',
                       help='清除所有缓存，默认为False')
    parser.add_argument('--show_batch_results', action='store_true',
                       help='显示批次处理结果，默认为False')
    parser.add_argument('--use_optimal_params', action='store_true',
                      help='使用主模型的最优超参数')
    parser.add_argument('--optimal_params_file', type=str,
                      default='results/hyperopt/{element}/best_params.json',
                      help='最优超参数文件路径，{element}会被替换为元素名称')
    parser.add_argument('--use_main_dataset', action='store_true',
                      help='使用主模型的数据集')
    parser.add_argument('--dataset_path', type=str,
                      default='processed_data/{type}_dataset.npz',
                      help='数据集路径，{type}会被替换为train/val/test')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备，可选值: cpu, cuda, tpu')
    
    # 添加直接设置超参数的命令行参数
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=None,
                       help='权重衰减')
    parser.add_argument('--max_depth', type=int, default=None,
                       help='XGBoost/LightGBM的最大树深度')
    parser.add_argument('--n_estimators', type=int, default=None,
                       help='XGBoost/LightGBM的树数量')
    parser.add_argument('--subsample', type=float, default=None,
                       help='XGBoost/LightGBM的样本抽样比例')
    parser.add_argument('--colsample_bytree', type=float, default=None,
                       help='XGBoost/LightGBM的特征抽样比例')
    parser.add_argument('--early_stopping_rounds', type=int, default=None,
                       help='XGBoost/LightGBM的早停轮数')
    parser.add_argument('--min_child_weight', type=int, default=None,
                       help='XGBoost的最小子节点权重')
    parser.add_argument('--num_leaves', type=int, default=None,
                       help='LightGBM的叶子节点数')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 处理缓存
    if args.clear_cache:
        cache_manager.clear_cache()
        logger.info("已清除所有缓存")
    else:
        # 询问是否清除缓存
        ask_clear_cache(cache_manager)
    
    # 创建必要的目录
    os.makedirs(config.output_config['model_dir'], exist_ok=True)
    os.makedirs(config.output_config['results_dir'], exist_ok=True)
    
    # 如果只是显示批次结果，则显示结果并退出
    if args.show_batch_results:
        show_batch_results(args.element, args.model)
        return
    
    # 处理设备选择
    device = None
    if args.device:
        if args.device.lower() == 'tpu':
            try:
                import torch_xla
                import torch_xla.core.xla_model as xm
                device = xm.xla_device()
            except ImportError:
                logger.warning("无法导入torch_xla，回退到CPU")
                device = 'cpu'
        else:
            device = args.device
    
    # 设置通过命令行指定的超参数
    xgb_params = {}
    lgb_params = {}
    
    if args.lr is not None:
        xgb_params['learning_rate'] = args.lr
        lgb_params['learning_rate'] = args.lr
    
    if args.weight_decay is not None:
        xgb_params['reg_lambda'] = args.weight_decay * 1000
        lgb_params['reg_lambda'] = args.weight_decay * 1000
    
    if args.max_depth is not None:
        xgb_params['max_depth'] = args.max_depth
        # LightGBM中-1表示无限制
        lgb_params['max_depth'] = args.max_depth
    
    if args.n_estimators is not None:
        xgb_params['n_estimators'] = args.n_estimators
        lgb_params['n_estimators'] = args.n_estimators
    
    if args.subsample is not None:
        xgb_params['subsample'] = args.subsample
        lgb_params['subsample'] = args.subsample
    
    if args.colsample_bytree is not None:
        xgb_params['colsample_bytree'] = args.colsample_bytree
        lgb_params['colsample_bytree'] = args.colsample_bytree
    
    if args.early_stopping_rounds is not None:
        xgb_params['early_stopping_rounds'] = args.early_stopping_rounds
        lgb_params['early_stopping_rounds'] = args.early_stopping_rounds
    
    if args.min_child_weight is not None:
        xgb_params['min_child_weight'] = args.min_child_weight
    
    if args.num_leaves is not None:
        lgb_params['num_leaves'] = args.num_leaves
    
    if args.seed is not None:
        xgb_params['seed'] = args.seed
        lgb_params['seed'] = args.seed
    
    # 如果有命令行参数，则更新模型参数
    if xgb_params:
        logger.info(f"使用命令行指定的XGBoost参数: {xgb_params}")
    
    if lgb_params:
        logger.info(f"使用命令行指定的LightGBM参数: {lgb_params}")
            
    # 加载最优超参数（只有在未通过命令行指定参数时才使用最优参数）
    if args.use_optimal_params and not xgb_params and not lgb_params:
        params_file = args.optimal_params_file.replace('{element}', args.element)
        if os.path.exists(params_file):
            with open(params_file, 'r') as f:
                optimal_params = json.load(f)
                logger.info(f"加载最优超参数: {optimal_params}")
                
                # 更新模型参数
                if args.model.lower() in ['xgboost', 'both']:
                    optimal_xgb_params = translate_params_to_xgboost(optimal_params)
                    logger.info(f"转换后的XGBoost参数: {optimal_xgb_params}")
                    xgb_params.update(optimal_xgb_params)
                
                if args.model.lower() in ['lightgbm', 'both']:
                    optimal_lgb_params = translate_params_to_lightgbm(optimal_params)
                    logger.info(f"转换后的LightGBM参数: {optimal_lgb_params}")
                    lgb_params.update(optimal_lgb_params)
        else:
            logger.warning(f"找不到最优超参数文件: {params_file}")
    
    # 训练和评估模型
    if args.model.lower() == 'both':
        # 训练和评估两种模型
        logger.info(f"将训练和评估 XGBoost 和 LightGBM 模型用于 {args.element}")
        
        # XGBoost
        xgb_results = train_and_evaluate_baseline(
            args.element, 'xgboost',
            batch_size=args.batch_size,
            batches_per_round=args.batches_per_round,
            val_size=args.val_size,
            force_retrain=args.force_retrain,
            evaluate_only=args.evaluate_only,
            device=device,
            xgb_params=xgb_params,
            lgb_params=lgb_params
        )
        
        # LightGBM
        lgb_results = train_and_evaluate_baseline(
            args.element, 'lightgbm',
            batch_size=args.batch_size,
            batches_per_round=args.batches_per_round,
            val_size=args.val_size,
            force_retrain=args.force_retrain,
            evaluate_only=args.evaluate_only,
            device=device,
            xgb_params=xgb_params,
            lgb_params=lgb_params
        )
        
        # 比较两种模型
        if xgb_results and lgb_results:
            logger.info(f"模型比较 ({args.element}):")
            logger.info(f"  XGBoost RMSE: {xgb_results['rmse']:.6f}, MAE: {xgb_results['mae']:.6f}")
            logger.info(f"  LightGBM RMSE: {lgb_results['rmse']:.6f}, MAE: {lgb_results['mae']:.6f}")
            
            # 确定胜者
            if xgb_results['rmse'] < lgb_results['rmse']:
                winner = "XGBoost"
                winner_rmse = xgb_results['rmse']
                loser = "LightGBM"
                loser_rmse = lgb_results['rmse']
            else:
                winner = "LightGBM"
                winner_rmse = lgb_results['rmse']
                loser = "XGBoost"
                loser_rmse = xgb_results['rmse']
                
            improvement = (loser_rmse - winner_rmse) / loser_rmse * 100
            logger.info(f"  胜者: {winner}，提升了 {improvement:.2f}% 的RMSE")
            
            # 比较结果可视化
            compare_models(args.element, xgb_results, lgb_results)
    else:
        # 只训练和评估一种模型
        results = train_and_evaluate_baseline(
            args.element, args.model,
            batch_size=args.batch_size,
            batches_per_round=args.batches_per_round,
            val_size=args.val_size,
            force_retrain=args.force_retrain,
            evaluate_only=args.evaluate_only,
            device=device,
            xgb_params=xgb_params,
            lgb_params=lgb_params
        )

def show_batch_results(element, model_type='both'):
    """
    显示批次处理结果
    
    参数:
        element: 元素名称
        model_type: 模型类型
    """
    models = []
    if model_type.lower() == 'both':
        models = ['xgboost', 'lightgbm']
    else:
        models = [model_type.lower()]
    
    for model_name in models:
        batch_results_dir = os.path.join(config.output_config['results_dir'], f'{model_name}_{element}_batch_results')
        if not os.path.exists(batch_results_dir):
            logger.warning(f"找不到 {model_name} 的批次结果目录: {batch_results_dir}")
            continue
        
        batch_tracking_path = os.path.join(batch_results_dir, 'batch_tracking.csv')
        if not os.path.exists(batch_tracking_path):
            logger.warning(f"找不到 {model_name} 的批次追踪文件: {batch_tracking_path}")
            continue
        
        # 显示批次追踪数据
        batch_df = pd.read_csv(batch_tracking_path)
        logger.info(f"{model_name} 在 {element} 上的批次处理结果:")
        logger.info(f"总批次数: {len(batch_df)}")
        
        if len(batch_df) > 0:
            # 找出最佳批次
            best_batch_id = batch_df.loc[batch_df['rmse'].idxmin(), 'batch_id']
            best_rmse = batch_df['rmse'].min()
            logger.info(f"最佳批次: {best_batch_id} (RMSE: {best_rmse:.6f})")
            
            # 显示每个批次的指标
            for i, row in batch_df.iterrows():
                batch_id = int(row['batch_id'])
                logger.info(f"批次 {batch_id}: RMSE={row['rmse']:.6f}, MAE={row['mae']:.6f}, R²={row['r2']:.6f}, 时间={row['timestamp']}")
                
            # 显示批次结果文件的位置
            logger.info(f"批次评估指标摘要文件位于: {batch_results_dir}/batch_*_summary.txt")
            logger.info(f"批次预测结果CSV文件位于: {batch_results_dir}/batch_*_results.csv")
            logger.info(f"批次散点图位于: {batch_results_dir}/batch_*_plot.png")
            logger.info(f"批次趋势图位于: {batch_results_dir}/batch_trends.png")

def compare_models(element, xgb_results, lgb_results):
    """
    比较XGBoost和LightGBM模型的性能，生成对比图表
    
    参数:
        element: 元素名称
        xgb_results: XGBoost评估结果
        lgb_results: LightGBM评估结果
    """
    try:
        # 创建比较目录
        compare_dir = os.path.join(config.output_config['results_dir'], f'model_comparison_{element}')
        os.makedirs(compare_dir, exist_ok=True)
        
        # 准备比较数据
        metrics = ['mse', 'rmse', 'mae', 'r2', 'scatter']
        xgb_values = [xgb_results[m] for m in metrics]
        lgb_values = [lgb_results[m] for m in metrics]
        
        # 创建性能对比图
        plt.figure(figsize=(12, 8))
        
        # RMSE和MAE对比
        plt.subplot(2, 2, 1)
        models = ['XGBoost', 'LightGBM']
        x = np.arange(len(models))
        width = 0.35
        plt.bar(x - width/2, [xgb_results['rmse'], lgb_results['rmse']], width, label='RMSE')
        plt.bar(x + width/2, [xgb_results['mae'], lgb_results['mae']], width, label='MAE')
        plt.xlabel('模型')
        plt.ylabel('误差')
        plt.title('RMSE和MAE对比')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(True, axis='y')
        
        # R²对比
        plt.subplot(2, 2, 2)
        plt.bar(models, [xgb_results['r2'], lgb_results['r2']], color=['blue', 'orange'])
        plt.xlabel('模型')
        plt.ylabel('R²')
        plt.title('R²对比')
        plt.grid(True, axis='y')
        
        # 散度对比
        plt.subplot(2, 2, 3)
        plt.bar(models, [xgb_results['scatter'], lgb_results['scatter']], color=['blue', 'orange'])
        plt.xlabel('模型')
        plt.ylabel('散度')
        plt.title('散度对比')
        plt.grid(True, axis='y')
        
        # 雷达图对比
        plt.subplot(2, 2, 4, polar=True)
        # 标准化指标，使得较小的值更好
        normalized_metrics = ['mse', 'rmse', 'mae', 'scatter', 'r2']
        max_values = {
            'mse': max(xgb_results['mse'], lgb_results['mse']),
            'rmse': max(xgb_results['rmse'], lgb_results['rmse']),
            'mae': max(xgb_results['mae'], lgb_results['mae']),
            'scatter': max(xgb_results['scatter'], lgb_results['scatter']),
            'r2': 1.0  # R²的最大值是1
        }
        
        # 对R²进行特殊处理，因为它越接近1越好
        normalized_xgb = [xgb_results[m]/max_values[m] if m != 'r2' else 1-xgb_results[m] for m in normalized_metrics]
        normalized_lgb = [lgb_results[m]/max_values[m] if m != 'r2' else 1-lgb_results[m] for m in normalized_metrics]
        
        # 完成雷达图的圆形
        normalized_xgb.append(normalized_xgb[0])
        normalized_lgb.append(normalized_lgb[0])
        
        # 设置角度
        angles = np.linspace(0, 2*np.pi, len(normalized_metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合多边形
        
        # 绘制雷达图
        plt.polar(angles, normalized_xgb, 'b-', label='XGBoost')
        plt.polar(angles, normalized_lgb, 'r-', label='LightGBM')
        plt.fill(angles, normalized_xgb, 'b', alpha=0.1)
        plt.fill(angles, normalized_lgb, 'r', alpha=0.1)
        
        # 设置雷达图标签
        labels = normalized_metrics + [normalized_metrics[0]]  # 添加第一个标签再次闭合
        plt.xticks(angles, labels)
        plt.title('性能雷达图对比')
        plt.legend(loc='upper right')
        
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
            f.write(f"  MSE: {xgb_results['mse']:.6f}\n")
            f.write(f"  RMSE: {xgb_results['rmse']:.6f}\n")
            f.write(f"  MAE: {xgb_results['mae']:.6f}\n")
            f.write(f"  R²: {xgb_results['r2']:.6f}\n")
            f.write(f"  散度: {xgb_results['scatter']:.6f}\n")
            
            f.write("\nLightGBM性能指标:\n")
            f.write(f"  MSE: {lgb_results['mse']:.6f}\n")
            f.write(f"  RMSE: {lgb_results['rmse']:.6f}\n")
            f.write(f"  MAE: {lgb_results['mae']:.6f}\n")
            f.write(f"  R²: {lgb_results['r2']:.6f}\n")
            f.write(f"  散度: {lgb_results['scatter']:.6f}\n")
            
            # 计算性能提升
            if xgb_results['rmse'] < lgb_results['rmse']:
                winner = "XGBoost"
                improvement = (lgb_results['rmse'] - xgb_results['rmse']) / lgb_results['rmse'] * 100
            else:
                winner = "LightGBM"
                improvement = (xgb_results['rmse'] - lgb_results['rmse']) / xgb_results['rmse'] * 100
                
            f.write(f"\n总结: {winner}在RMSE指标上表现更好，提升了{improvement:.2f}%。\n")
        
        logger.info(f"已生成模型比较结果，保存在: {compare_dir}")
        
    except Exception as e:
        logger.error(f"生成模型比较结果时出错: {e}")

def translate_params_to_xgboost(optimal_params):
    """将主模型参数转换为XGBoost参数"""
    return {
        'learning_rate': optimal_params.get('lr', 0.01),
        'max_depth': 6,  # 可以映射其他参数
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': optimal_params.get('weight_decay', 1e-4) * 1000,
    }

def translate_params_to_lightgbm(optimal_params):
    """将主模型参数转换为LightGBM参数"""
    return {
        'learning_rate': optimal_params.get('lr', 0.01),
        'num_leaves': 31,
        'max_depth': -1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': optimal_params.get('weight_decay', 1e-4) * 1000,
    }

if __name__ == '__main__':
    main() 