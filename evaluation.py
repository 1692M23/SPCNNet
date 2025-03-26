#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估模块：对恒星光谱元素丰度预测模型进行评估和可视化
包含两个阶段的评估：
1. 模型学习阶段：在测试集上评估模型性能，真实值为APOGEE DR17的元素丰度
2. 模型预测阶段：评估生成的星表与GALAH DR3和LASP星表的一致性
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from astropy.io import fits
from scipy.stats import spearmanr, pearsonr
import time
import nn

# 导入自定义模块
import config
from model import load_trained_model, predict
from preprocessdata import LAMOSTPreprocessor
from utils import CacheManager, ProgressManager, ask_clear_cache

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.output_config['log_dir'], 'evaluation.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('evaluation')

# 创建缓存管理器
cache_manager = CacheManager(cache_dir=os.path.join(config.output_config['cache_dir'], 'evaluation'))

def load_element_results(element):
    """
    加载元素测试结果数据
    
    参数:
        element (str): 元素名称
        
    返回:
        dict: 包含测试结果的字典，如果加载失败则返回None
    """
    # 检查缓存
    cache_key = f"results_{element}"
    cached_results = cache_manager.get_cache(cache_key)
    if cached_results is not None:
        logger.info(f"从缓存加载测试结果: {element}")
        return cached_results['results']
    
    result_path = os.path.join(config.output_config['results_dir'], f"{element}_test_results.npz")
    if not os.path.exists(result_path):
        logger.error(f"找不到 {element} 的测试结果文件: {result_path}")
        return None
    
    try:
        data = np.load(result_path)
        results = {key: data[key] for key in data.files}
        logger.info(f"成功加载 {element} 测试结果")
        
        # 保存到缓存
        cache_manager.set_cache(cache_key, {
            'results': results
        })
        
        return results
    except Exception as e:
        logger.error(f"加载 {element} 测试结果失败: {str(e)}")
        return None

def load_csv_metadata(csv_file):
    """
    加载CSV文件中的元数据（teff和logg）
    
    参数:
        csv_file (str): CSV文件路径
        
    返回:
        pandas.DataFrame: 包含obsid、teff和logg的DataFrame
    """
    # 检查缓存
    cache_key = f"metadata_{os.path.basename(csv_file)}"
    cached_metadata = cache_manager.get_cache(cache_key)
    if cached_metadata is not None:
        logger.info(f"从缓存加载元数据: {csv_file}")
        return cached_metadata['metadata']
    
    try:
        df = pd.read_csv(csv_file)
        if 'obsid' not in df.columns or 'teff' not in df.columns or 'logg' not in df.columns:
            logger.error(f"CSV文件 {csv_file} 缺少必要的列(obsid, teff, logg)")
            return None
            
        # 保存到缓存
        cache_manager.set_cache(cache_key, {
            'metadata': df
        })
        
        return df
    except Exception as e:
        logger.error(f"加载CSV元数据失败: {str(e)}")
        return None

def load_test_predictions(elements=None, csv_files=None):
    """
    加载测试数据预测结果和元数据
    
    参数:
        elements (list): 要加载的元素列表
        csv_files (dict): 元素对应的CSV文件路径字典
        
    返回:
        dict: 包含每个元素预测结果和元数据的字典
    """
    if elements is None:
        elements = config.CONFIG['elements']
    
    if csv_files is None:
        csv_files = {
            'C_FE': os.path.join('processed_data', 'C_FE_metadata.csv'),
            'MG_FE': os.path.join('processed_data', 'MG_FE_metadata.csv'),
            'CA_FE': os.path.join('processed_data', 'CA_FE_metadata.csv')
        }
    
    results = {}
    
    # 使用进度管理器
    with ProgressManager(len(elements), desc="加载测试预测结果") as progress:
        for element in elements:
            element_results = load_element_results(element)
            if element_results is None:
                continue
            
            # 加载对应的CSV元数据（如果存在）
            csv_file = csv_files.get(element, '')
            if csv_file and os.path.exists(csv_file):
                metadata_df = load_csv_metadata(csv_file)
                if metadata_df is not None:
                    element_results['metadata'] = metadata_df
            else:
                # 如果没有CSV文件，创建一个简单的元数据
                num_samples = len(element_results['true_values'])
                element_results['metadata'] = pd.DataFrame({
                    'obsid': [f"OBSID{i}" for i in range(num_samples)],
                    'teff': range(num_samples),
                    'logg': range(num_samples)
                })
            
            results[element] = element_results
            
            # 更新进度
            progress.update(1)
    
    return results

def predict_data(data, element, model=None):
    """
    使用训练好的模型预测数据
    
    参数:
        data (numpy.ndarray): 输入数据
        element (str): 元素名称
        model (torch.nn.Module): 模型，如果为None则加载已保存的模型
        
    返回:
        numpy.ndarray: 预测结果
    """
    # 检查缓存
    cache_key = f"predict_{element}_{data.shape}"
    cached_predictions = cache_manager.get_cache(cache_key)
    if cached_predictions is not None:
        logger.info(f"从缓存加载预测结果: {element}")
        return cached_predictions['predictions']
    
    device = config.training_config['device']
    
    # 如果未提供模型，加载保存的模型
    if model is None:
        model_path = os.path.join(config.model_config['model_dir'], f"{element}_model.pth")
        if not os.path.exists(model_path):
            logger.error(f"找不到 {element} 的模型文件: {model_path}")
            return None
        
        model = load_trained_model(element, model_path, device)
    
    model.eval()
    
    # 转换数据格式
    if len(data.shape) == 2:  # [batch_size, features]
        data_tensor = torch.FloatTensor(data).unsqueeze(1)  # 添加通道维度 [batch_size, 1, features]
    elif len(data.shape) == 1:  # [features]
        data_tensor = torch.FloatTensor(data).unsqueeze(0).unsqueeze(0)  # [1, 1, features]
    
    # 预测
    with torch.no_grad():
        data_tensor = data_tensor.to(device)
        predictions = model(data_tensor).cpu().numpy()
    
    predictions = predictions.flatten() if predictions.shape[0] > 1 else predictions[0]
    
    # 保存到缓存
    cache_manager.set_cache(cache_key, {
        'predictions': predictions
    })
    
    return predictions

def plot_mist_isochrones(ax, ages=None, colors=None):
    """
    在图表上绘制MIST恒星等时线
    
    参数:
        ax (matplotlib.axes.Axes): 图表对象
        ages (list): 年龄列表（单位：Gyr）
        colors (list): 对应的颜色列表
    """
    if not config.mist_isochrones['enabled']:
        return
    
    if ages is None:
        ages = config.mist_isochrones['ages']
    
    if colors is None:
        colors = config.mist_isochrones['colors']
    
    # 简化的MIST等时线数据（可以替换为加载实际MIST数据）
    for i, age in enumerate(ages):
        # 这里使用简化模型，实际应用中应该加载真实的MIST等时线数据
        # 假设的等时线数据，实际应用时替换为实际的MIST数据
        teff_range = np.linspace(3500, 8000, 100)
        if age == 1:
            logg = 4.2 + 0.5 * np.sin((teff_range - 3500) / 4500 * np.pi)
        elif age == 3:
            logg = 4.0 + 0.4 * np.sin((teff_range - 3500) / 4500 * np.pi)
        elif age == 5:
            logg = 3.8 + 0.3 * np.sin((teff_range - 3500) / 4500 * np.pi)
        elif age == 10:
            logg = 3.5 + 0.2 * np.sin((teff_range - 3500) / 4500 * np.pi)
        else:
            logg = 4.0 + (age/10) * np.sin((teff_range - 3500) / 4500 * np.pi)
        
        color = colors[i % len(colors)]
        ax.plot(teff_range, logg, '--', color=color, alpha=0.7, linewidth=1.5, label=f'{age} Gyr')

def plot_predictions_vs_true(elements=None, plot_dir=None, figsize=(12, 10)):
    """
    绘制预测值与真实值对比图
    
    参数:
        elements (list): 要绘制的元素列表
        plot_dir (str): 图表保存目录
        figsize (tuple): 图表大小
    """
    if elements is None:
        elements = config.training_config['elements']
    
    if plot_dir is None:
        plot_dir = config.output_config['plots_dir']
    
    # 加载测试预测结果和元数据
    results = load_test_predictions(elements)
    if not results:
        logger.error("没有找到任何测试结果")
        return
    
    # 使用进度管理器
    with ProgressManager(len(elements), desc="生成预测对比图") as progress:
        # 为每个元素创建图表
        for element, data in results.items():
            if 'true_values' not in data or 'predictions' not in data:
                logger.warning(f"{element} 数据中缺少真实值或预测值")
                continue
            
            # 创建图表
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            plt.subplots_adjust(wspace=0.3)
            
            # 真实值散点图
            ax0 = axes[0]
            scatter0 = ax0.scatter(
                data['metadata']['teff'] if 'metadata' in data else np.arange(len(data['true_values'])),
                data['metadata']['logg'] if 'metadata' in data else np.arange(len(data['true_values'])),
                c=data['true_values'],
                cmap='viridis',
                s=40,
                alpha=0.7
            )
            
            # 添加MIST等时线
            plot_mist_isochrones(ax0)
            
            # 预测值散点图
            ax1 = axes[1]
            scatter1 = ax1.scatter(
                data['metadata']['teff'] if 'metadata' in data else np.arange(len(data['predictions'])),
                data['metadata']['logg'] if 'metadata' in data else np.arange(len(data['predictions'])),
                c=data['predictions'],
                cmap='viridis',
                s=40,
                alpha=0.7
            )
            
            # 添加MIST等时线
            plot_mist_isochrones(ax1)
            
            # 设置颜色条比例一致
            vmin = min(data['true_values'].min(), data['predictions'].min())
            vmax = max(data['true_values'].max(), data['predictions'].max())
            norm = Normalize(vmin=vmin, vmax=vmax)
            scatter0.set_norm(norm)
            scatter1.set_norm(norm)
            
            # 添加颜色条
            cbar = fig.colorbar(ScalarMappable(norm=norm, cmap='viridis'), ax=axes, shrink=0.8)
            cbar.set_label(element)
            
            # 设置标题和标签
            ax0.set_title(f'{element} 真实值')
            ax1.set_title(f'{element} 预测值')
            
            for ax in axes:
                ax.set_xlabel('有效温度 (K)')
                ax.set_ylabel('表面重力 log g')
                ax.invert_xaxis()  # 天文学习惯，温度轴反向
                ax.grid(True, alpha=0.3)
            
            # 添加评估指标文本
            metrics_text = (
                f"MAE: {data['mae']:.4f}\n"
                f"RMSE: {data['rmse']:.4f}\n"
                f"R²: {data['r2']:.4f}\n"
                f"STD: {data['std']:.4f}"
            )
            ax1.text(
                0.05, 0.95, metrics_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            # 保存图表
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f'{element}_predictions_comparison.png'), dpi=300, bbox_inches='tight')
            logger.info(f"已保存 {element} 预测对比图")
            plt.close()
            
            # 更新进度
            progress.update(1)

def plot_metrics_comparison(elements=None, plot_dir=None, figsize=(10, 8)):
    """
    绘制所有元素评估指标对比折线图
    
    参数:
        elements (list): 要对比的元素列表
        plot_dir (str): 图表保存目录
        figsize (tuple): 图表大小
    """
    if elements is None:
        elements = config.training_config['elements']
    
    if plot_dir is None:
        plot_dir = config.output_config['plots_dir']
    
    # 加载所有元素的测试结果
    results = {}
    
    # 使用进度管理器
    with ProgressManager(len(elements), desc="加载评估结果") as progress:
        for element in elements:
            element_result = load_element_results(element)
            if element_result:
                results[element] = element_result
            
            # 更新进度
            progress.update(1)
    
    if not results:
        logger.error("没有找到任何测试结果")
        return
    
    # 准备数据
    metrics = ['mae', 'mse', 'rmse', 'r2', 'std']
    metrics_labels = ['MAE', 'MSE', 'RMSE', 'R²', 'STD']
    metrics_values = {metric: [] for metric in metrics}
    elements_list = []
    
    for element, data in results.items():
        elements_list.append(element)
        for metric in metrics:
            if metric in data:
                metrics_values[metric].append(data[metric])
            else:
                metrics_values[metric].append(np.nan)
    
    # 创建数据框
    df = pd.DataFrame({
        'Element': elements_list * len(metrics),
        'Metric': np.repeat(metrics_labels, len(elements_list)),
        'Value': np.concatenate([metrics_values[metric] for metric in metrics])
    })
    
    # 为R²使用不同的y轴（因为它的范围不同）
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    
    # 绘制非R²指标
    non_r2_df = df[df['Metric'] != 'R²']
    sns.lineplot(x='Element', y='Value', hue='Metric', marker='o', data=non_r2_df, ax=ax1)
    
    # 绘制R²指标
    r2_df = df[df['Metric'] == 'R²']
    sns.lineplot(x='Element', y='Value', marker='s', color='purple', data=r2_df, ax=ax2, label='R²')
    
    # 设置图表标题和标签
    plt.title('元素丰度预测模型性能对比')
    ax1.set_xlabel('元素')
    ax1.set_ylabel('误差值')
    ax2.set_ylabel('R²值')
    
    # 调整图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    ax2.get_legend().remove()  # 移除第二个图例
    
    # 添加网格
    ax1.grid(True, alpha=0.3)
    
    # 保存图表
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("已保存评估指标对比图")

def evaluate_all_elements(elements=None, save_dir=None):
    """
    评估所有元素模型的性能并生成可视化图表
    
    参数:
        elements (list): 要评估的元素列表
        save_dir (str): 结果保存目录
    """
    if elements is None:
        elements = config.training_config['elements']
    
    if save_dir is None:
        save_dir = config.output_config['results_dir']
    
    # 加载测试数据
    test_data = np.load(config.data_paths['test_data'])
    
    # 评估结果
    all_results = {}
    
    # 使用进度管理器
    with ProgressManager(len(elements), desc="评估元素模型") as progress:
        for element in elements:
            logger.info(f"评估 {element} 模型...")
            
            # 加载模型
            model_path = os.path.join(config.model_config['model_dir'], f"{element}_model.pth")
            if not os.path.exists(model_path):
                logger.error(f"找不到 {element} 的模型文件")
                continue
            
            device = config.training_config['device']
            model = load_trained_model(element, model_path, device)
            
            # 预测测试数据
            spectra = test_data['spectra']
            true_values = test_data[element]
            
            try:
                # 转换数据格式并预测
                spectra_tensor = torch.FloatTensor(spectra).unsqueeze(1).to(device)
                model.eval()
                with torch.no_grad():
                    predictions = model(spectra_tensor).cpu().numpy().flatten()
                
                # 计算评估指标
                mae = mean_absolute_error(true_values, predictions)
                mse = mean_squared_error(true_values, predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(true_values, predictions)
                std = np.std(predictions - true_values)
                
                # 保存结果
                results = {
                    'element': element,
                    'true_values': true_values,
                    'predictions': predictions,
                    'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
                    'std': std
                }
                
                # 保存到文件
                results_path = os.path.join(save_dir, f"{element}_test_results.npz")
                np.savez(results_path, **results)
                
                # 添加到总结果
                all_results[element] = results
                
                # 输出评估指标
                logger.info(f"{element} 评估结果:")
                logger.info(f"  MAE:  {mae:.6f}")
                logger.info(f"  MSE:  {mse:.6f}")
                logger.info(f"  RMSE: {rmse:.6f}")
                logger.info(f"  R²:   {r2:.6f}")
                logger.info(f"  STD:  {std:.6f}")
                
            except Exception as e:
                logger.error(f"{element} 评估失败: {str(e)}")
            
            # 更新进度
            progress.update(1)
    
    return all_results

def load_catalog_data(catalog_file, element_name):
    """
    加载交叉匹配的星表数据文件
    
    参数:
        catalog_file (str): 星表文件路径，如 'galah_MG_FE.csv'
        element_name (str): 元素名称
        
    返回:
        pd.DataFrame: 包含OBSID和真实元素丰度的数据框
    """
    # 检查缓存
    cache_key = f"catalog_{os.path.basename(catalog_file)}_{element_name}"
    cached_data = cache_manager.get_cache(cache_key)
    if cached_data is not None:
        logger.info(f"从缓存加载星表数据: {catalog_file}")
        return cached_data['data']
    
    try:
        logger.info(f"正在加载星表数据: {catalog_file}")
        df = pd.read_csv(catalog_file)
        
        # 检查是否包含必需列
        if 'obsid' not in df.columns:
            logger.error(f"星表文件 {catalog_file} 中缺少'obsid'列")
            return None
            
        # 确保元素丰度列在最后一列
        abundance_col = df.columns[-1]
        logger.info(f"元素丰度列名: {abundance_col}")
        
        # 提取只需要的列：OBSID和元素丰度
        result_df = df[['obsid', abundance_col]].copy()
        result_df.rename(columns={abundance_col: 'true_abundance'}, inplace=True)
        
        logger.info(f"已加载 {len(result_df)} 个样本")
        
        # 保存到缓存
        cache_manager.set_cache(cache_key, {
            'data': result_df
        })
        
        return result_df
    except Exception as e:
        logger.error(f"加载星表数据出错: {e}")
        return None

def predict_catalog_spectra(spectra_df, element, fits_dir='preFits', model=None):
    """
    预测星表中光谱的元素丰度
    
    参数:
        spectra_df (pd.DataFrame): 包含OBSID的数据框
        element (str): 元素名称
        fits_dir (str): FITS文件目录
        model: 预训练模型，如果为None则加载
        
    返回:
        pd.DataFrame: 包含预测结果的数据框
    """
    if model is None:
        # 加载预训练模型
        input_size = config.model_config['input_dim']
        model = load_trained_model(input_size, element)
    
    # 创建预处理器
    preprocessor = LAMOSTPreprocessor()
    
    # 初始化结果列表
    results = []
    
    # 使用进度管理器
    with ProgressManager(len(spectra_df), desc=f"预测 {element} 元素丰度") as progress:
        for idx, row in spectra_df.iterrows():
            try:
                obsid = row['obsid']
                
                # 查找并读取FITS文件
                fits_file = preprocessor._find_fits_file(obsid)
                if fits_file is None:
                    logger.warning(f"找不到OBSID为{obsid}的FITS文件")
                    continue
                
                # 预处理光谱数据
                processed_data = preprocessor.process_single_spectrum(obsid, None)
                if not processed_data or 'spectrum' not in processed_data:
                    logger.warning(f"处理OBSID为{obsid}的光谱失败")
                    continue
                    
                # 获取处理后的光谱
                spectrum = processed_data['spectrum']
                
                # 转换为张量并预测
                input_tensor = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                abundance_pred = predict(model, input_tensor, config.training_config['device'])
                
                # 保存结果
                results.append({
                    'obsid': obsid,
                    'true_abundance': row['true_abundance'],
                    'predicted_abundance': float(abundance_pred)
                })
                
                # 更新进度
                progress.update(1)
                    
            except Exception as e:
                logger.error(f"处理OBSID {row['obsid']} 时出错: {e}")
                import traceback
                traceback.print_exc()
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def evaluate_catalog_match(element, catalog_type, save_predictions=True):
    """
    评估模型在交叉匹配的星表上的性能
    
    参数:
        element (str): 元素名称，如'MG_FE'
        catalog_type (str): 星表类型，'galah'或'lasp'
        save_predictions (bool): 是否保存预测结果
        
    返回:
        dict: 评估指标结果
    """
    # 构建星表文件路径
    catalog_file = f"{catalog_type}_{element}.csv"
    catalog_path = os.path.join(config.output_config['results_dir'], catalog_file)
    
    if not os.path.exists(catalog_path):
        logger.error(f"找不到星表文件: {catalog_path}")
        return None
    
    # 加载星表数据
    catalog_df = load_catalog_data(catalog_path, element)
    if catalog_df is None or catalog_df.empty:
        logger.error(f"无法加载星表数据: {catalog_path}")
        return None
    
    # 预测元素丰度
    results_df = predict_catalog_spectra(catalog_df, element)
    if results_df is None or results_df.empty:
        logger.error(f"预测失败: {element}")
        return None
    
    # 计算评估指标
    y_true = results_df['true_abundance'].values
    y_pred = results_df['predicted_abundance'].values
    
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'pearson_r': pearsonr(y_true, y_pred)[0],
        'spearman_r': spearmanr(y_true, y_pred)[0],
        'std_diff': np.std(y_pred - y_true),
        'median_diff': np.median(np.abs(y_pred - y_true)),
        'num_samples': len(y_true)
    }
    
    logger.info(f"{catalog_type.upper()} {element} 评估结果:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # 保存预测结果
    if save_predictions:
        out_dir = os.path.join(config.output_config['predictions_dir'])
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{catalog_type}_{element}_predictions.csv")
        results_df.to_csv(out_file, index=False)
        logger.info(f"已保存预测结果到: {out_file}")
    
    return metrics

def plot_catalog_comparison(elements, catalog_types=None, metrics=None, plot_dir=None, figsize=(12, 8)):
    """
    绘制不同星表评估结果的对比图
    
    参数:
        elements (list): 要绘制的元素列表
        catalog_types (list): 星表类型列表，默认为['galah', 'lasp']
        metrics (list): 要对比的评估指标列表
        plot_dir (str): 图像保存目录
        figsize (tuple): 图像大小
    """
    if elements is None:
        elements = config.training_config['elements']
    if isinstance(elements, str):
        elements = [elements]
        
    if catalog_types is None:
        catalog_types = ['galah', 'lasp']
        
    if metrics is None:
        metrics = ['mae', 'rmse', 'r2', 'pearson_r']
        
    if plot_dir is None:
        plot_dir = config.output_config['plots_dir']
    
    os.makedirs(plot_dir, exist_ok=True)
    
    # 存储评估结果
    all_results = {}
    
    # 获取评估结果
    for element in elements:
        element_results = {}
        for catalog in catalog_types:
            # 评估当前星表的结果
            metrics_results = evaluate_catalog_match(element, catalog)
            if metrics_results:
                element_results[catalog] = metrics_results
        
        if element_results:
            all_results[element] = element_results
    
    if not all_results:
        logger.error("没有可用的评估结果")
        return
    
    # 为每个指标绘制一个条形图
    for metric in metrics:
        plt.figure(figsize=figsize)
        
        # 准备数据
        metric_data = []
        for element in elements:
            if element in all_results:
                for catalog in catalog_types:
                    if catalog in all_results[element]:
                        if metric in all_results[element][catalog]:
                            metric_data.append({
                                'Element': element,
                                'Catalog': catalog.upper(),
                                'Value': all_results[element][catalog][metric]
                            })
        
        if not metric_data:
            logger.warning(f"没有可用的 {metric} 指标数据")
            continue
            
        # 创建DataFrame
        df = pd.DataFrame(metric_data)
        
        # 使用Seaborn绘制条形图
        ax = sns.barplot(x='Element', y='Value', hue='Catalog', data=df)
        
        # 设置标题和标签
        metric_names = {
            'mae': '平均绝对误差 (MAE)',
            'mse': '均方误差 (MSE)',
            'rmse': '均方根误差 (RMSE)',
            'r2': '决定系数 (R²)',
            'pearson_r': '皮尔逊相关系数',
            'spearman_r': '斯皮尔曼相关系数',
            'std_diff': '误差标准差',
            'median_diff': '绝对误差中位数'
        }
        metric_title = metric_names.get(metric, metric)
        
        plt.title(f'不同星表的{metric_title}对比', fontsize=14)
        plt.ylabel(metric_title, fontsize=12)
        plt.xlabel('元素', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='星表', fontsize=10)
        
        # 添加具体数值标签
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'bottom', 
                        fontsize=8, rotation=0)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'catalog_comparison_{metric}.png'), dpi=300)
    plt.close()

    logger.info(f"已保存星表对比图到: {plot_dir}")

def plot_teff_logg_abundance(element, catalog_type, plot_dir=None, figsize=(10, 8)):
    """
    在teff-logg平面上绘制元素丰度分布图
    
    参数:
        element (str): 元素名称
        catalog_type (str): 星表类型，'galah'或'lasp'
        plot_dir (str): 图像保存目录
        figsize (tuple): 图像大小
    """
    if plot_dir is None:
        plot_dir = config.output_config['plots_dir']
    
    os.makedirs(plot_dir, exist_ok=True)
    
    # 构建预测结果文件路径
    pred_file = os.path.join(config.output_config['predictions_dir'], 
                            f"{catalog_type}_{element}_predictions.csv")
    
    if not os.path.exists(pred_file):
        logger.error(f"找不到预测结果文件: {pred_file}")
        return
    
    # 加载预测结果
    pred_df = pd.read_csv(pred_file)
    
    # 构建原始星表文件路径
    catalog_file = f"{catalog_type}_{element}.csv"
    catalog_path = os.path.join(config.output_config['results_dir'], catalog_file)
    
    if not os.path.exists(catalog_path):
        logger.error(f"找不到星表文件: {catalog_path}")
        return
    
    # 加载原始星表数据
    catalog_df = pd.read_csv(catalog_path)
    
    # 检查是否有teff和logg列
    if 'teff' not in catalog_df.columns or 'logg' not in catalog_df.columns:
        logger.error(f"星表文件 {catalog_path} 中缺少teff或logg列")
        return
    
    # 合并数据
    merged_df = pd.merge(pred_df, catalog_df[['obsid', 'teff', 'logg']], 
                         left_on='obsid', right_on='obsid', how='inner')
    
    # 创建两个子图: 真实丰度和预测丰度
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 第一个子图：真实丰度
    scatter1 = ax1.scatter(merged_df['teff'], merged_df['logg'], 
                          c=merged_df['true_abundance'], cmap='viridis', 
                          alpha=0.7, s=20)
    
    ax1.set_xlim(ax1.get_xlim()[::-1])  # 反转x轴（温度通常从高到低显示）
    ax1.set_ylim(ax1.get_ylim()[::-1])  # 反转y轴（恒星表面重力通常从低到高显示）
    
    ax1.set_xlabel('有效温度 (K)', fontsize=12)
    ax1.set_ylabel('表面重力 (log g)', fontsize=12)
    ax1.set_title(f'真实{element}丰度分布', fontsize=14)
    
    # 第二个子图：预测丰度
    scatter2 = ax2.scatter(merged_df['teff'], merged_df['logg'], 
                          c=merged_df['predicted_abundance'], cmap='viridis', 
                          alpha=0.7, s=20)
    
    ax2.set_xlim(ax2.get_xlim()[::-1])  # 反转x轴
    ax2.set_ylim(ax2.get_ylim()[::-1])  # 反转y轴
    
    ax2.set_xlabel('有效温度 (K)', fontsize=12)
    ax2.set_ylabel('表面重力 (log g)', fontsize=12)
    ax2.set_title(f'预测{element}丰度分布', fontsize=14)
    
    # 添加颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(scatter2, cax=cbar_ax)
    cbar.set_label(f'{element}元素丰度', fontsize=12)
    
    # 设置整体标题
    plt.suptitle(f'基于{catalog_type.upper()}星表的{element}丰度在恒星H-R图中的分布', fontsize=16)
    
    # 添加MIST等时线
    try:
        plot_mist_isochrones(ax1)
        plot_mist_isochrones(ax2)
    except Exception as e:
        logger.warning(f"无法添加MIST等时线: {e}")
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # 调整布局，为颜色条留出空间
    
    # 保存图像
    fig_path = os.path.join(plot_dir, f'{catalog_type}_{element}_teff_logg.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()

    logger.info(f"已保存 {element} 在teff-logg空间的分布图: {fig_path}")

def evaluate_catalogs_all_elements(elements=None, catalog_types=None, save_plots=True):
    """
    评估所有元素在所有星表上的性能
    
    参数:
        elements (list): 要评估的元素列表
        catalog_types (list): 星表类型列表
        save_plots (bool): 是否保存图像
        
    返回:
        dict: 包含所有评估结果的字典
    """
    if elements is None:
        elements = [elem.get('element') for elem in config.CONFIG['elements']]
    if isinstance(elements, str):
        elements = [elements]
        
    if catalog_types is None:
        catalog_types = ['galah', 'lasp']
    
    # 创建结果目录
    results_dir = config.output_config['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # 存储所有结果
    all_results = {}
    
    # 评估每个元素在每个星表上的性能
    for element in elements:
        logger.info(f"正在评估元素 {element}")
        element_results = {}
        
        for catalog in catalog_types:
            logger.info(f"使用星表 {catalog} 评估 {element}")
            
            # 评估模型在当前星表上的性能
            metrics = evaluate_catalog_match(element, catalog)
            
            if metrics:
                element_results[catalog] = metrics
                
                # 绘制teff-logg空间分布图
                if save_plots:
                    plot_teff_logg_abundance(element, catalog)
        
        if element_results:
            all_results[element] = element_results
    
    # 绘制比较图
    if save_plots and len(elements) > 0 and len(catalog_types) > 0:
        plot_catalog_comparison(elements, catalog_types)
    
    # 保存评估结果
    results_file = os.path.join(results_dir, 'catalog_evaluation_results.json')
    import json
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    logger.info(f"已保存所有评估结果到: {results_file}")
    
    return all_results

def evaluate_element_by_stellar_type(model, test_data, element, logg_threshold=4.0, config=None):
    """按恒星类型评估元素丰度预测性能
    
    参数:
        model: 训练好的模型
        test_data: 测试数据(X, y, metadata)
        element: 元素名称
        logg_threshold: 矮星/巨星的logg阈值
        config: 配置对象
    
    返回:
        评估结果字典
    """
    X, y, metadata = test_data
    
    # 获取logg值（如果有）
    has_logg = 'logg' in metadata.columns
    
    # 进行预测
    device = config.training_config['device']
    predictions, uncertainties = model.predict(X, device=device)
    
    # 计算整体评估指标
    metrics = {
        'mae': np.mean(np.abs(predictions - y)),
        'rmse': np.sqrt(np.mean((predictions - y) ** 2)),
        'bias': np.mean(predictions - y),
        'scatter': np.std(predictions - y)
    }
    
    # 如果有logg值，按恒星类型分组评估
    if has_logg:
        # 矮星掩码 (logg >= threshold)
        dwarfs_mask = metadata['logg'] >= logg_threshold
        
        # 矮星评估
        if np.sum(dwarfs_mask) > 0:
            dwarfs_pred = predictions[dwarfs_mask]
            dwarfs_true = y[dwarfs_mask]
            metrics['dwarfs'] = {
                'count': np.sum(dwarfs_mask),
                'mae': np.mean(np.abs(dwarfs_pred - dwarfs_true)),
                'rmse': np.sqrt(np.mean((dwarfs_pred - dwarfs_true) ** 2)),
                'bias': np.mean(dwarfs_pred - dwarfs_true),
                'scatter': np.std(dwarfs_pred - dwarfs_true)
            }
        
        # 巨星评估
        giants_mask = metadata['logg'] < logg_threshold
        if np.sum(giants_mask) > 0:
            giants_pred = predictions[giants_mask]
            giants_true = y[giants_mask]
            metrics['giants'] = {
                'count': np.sum(giants_mask),
                'mae': np.mean(np.abs(giants_pred - giants_true)),
                'rmse': np.sqrt(np.mean((giants_pred - giants_true) ** 2)),
                'bias': np.mean(giants_pred - giants_true),
                'scatter': np.std(giants_pred - giants_true)
            }
    
    # 打印评估结果
    logger.info(f"{element}元素丰度预测评估结果:")
    logger.info(f"  整体MAE: {metrics['mae']:.6f}, RMSE: {metrics['rmse']:.6f}")
    logger.info(f"  偏差: {metrics['bias']:.4f}, 散度: {metrics['scatter']:.4f}")
    
    if has_logg and 'dwarfs' in metrics:
        logger.info(f"  矮星(样本数: {metrics['dwarfs']['count']}): "
                   f"MAE: {metrics['dwarfs']['mae']:.4f}, "
                   f"偏差: {metrics['dwarfs']['bias']:.4f}, "
                   f"散度: {metrics['dwarfs']['scatter']:.4f}")
    
    if has_logg and 'giants' in metrics:
        logger.info(f"  巨星(样本数: {metrics['giants']['count']}): "
                   f"MAE: {metrics['giants']['mae']:.4f}, "
                   f"偏差: {metrics['giants']['bias']:.4f}, "
                   f"散度: {metrics['giants']['scatter']:.4f}")
    
    return metrics

def visualize_hyperopt_results(element, plot_dir=None):
    """
    可视化超参数优化过程和结果
    
    参数:
        element (str): 元素名称
        plot_dir (str): 图表保存目录
    """
    import os
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    from hyperopt import space_eval
    
    if plot_dir is None:
        plot_dir = os.path.join(config.output_config['plots_dir'], 'hyperopt')
    
    os.makedirs(plot_dir, exist_ok=True)
    
    # 加载优化结果 - 先尝试加载批量优化结果
    results_dir = os.path.join(config.output_config['results_dir'], 'hyperopt')
    if not os.path.exists(results_dir):
        logger.warning(f"找不到超参数优化结果目录: {results_dir}")
        return
    
    # 尝试加载批量超参数优化的结果
    batch_results_path = os.path.join(results_dir, f'{element}_best_params.pkl')
    
    # 检查是否有批量优化的结果
    if os.path.exists(batch_results_path):
        try:
            with open(batch_results_path, 'rb') as f:
                batch_data = pickle.load(f)
            
            # 获取两组最佳参数
            best_params_set1 = batch_data.get('best_params_set1', {})
            best_params_set2 = batch_data.get('best_params_set2', {})
            best_loss_set1 = batch_data.get('best_loss_set1', 0)
            best_loss_set2 = batch_data.get('best_loss_set2', 0)
            
            # 生成比较图表
            plt.figure(figsize=(12, 8))
            
            # 按参数类型分组
            param_groups = {
                '学习率': ['lr'],
                '正则化': ['weight_decay'],
                '网络结构': ['dropout_rate'],
                '批次大小': ['batch_size']
            }
            
            # 创建子图，用于显示两组参数的对比
            fig, axes = plt.subplots(len(param_groups), 1, figsize=(10, 14))
            plt.subplots_adjust(hspace=0.4)
            
            # 循环绘制每组参数
            for i, (group_name, param_names) in enumerate(param_groups.items()):
                ax = axes[i]
                ax.set_title(f'{group_name}参数对比', fontsize=14)
                
                # 收集参数值
                param_values_set1 = []
                param_values_set2 = []
                param_labels = []
                
                for param_name in param_names:
                    if param_name in best_params_set1 and param_name in best_params_set2:
                        param_values_set1.append(best_params_set1[param_name])
                        param_values_set2.append(best_params_set2[param_name])
                        param_labels.append(param_name)
                
                # 设置x位置
                x = np.arange(len(param_labels))
                width = 0.35
                
                # 绘制条形图
                ax.bar(x - width/2, param_values_set1, width, label='参数组1')
                ax.bar(x + width/2, param_values_set2, width, label='参数组2')
                
                # 设置标签
                ax.set_xticks(x)
                ax.set_xticklabels(param_labels)
                ax.legend()
                
                # 添加数值标签
                for j, v in enumerate(param_values_set1):
                    ax.text(j - width/2, v * 1.05, f'{v:.6f}' if v < 0.01 else f'{v:.4f}', 
                           ha='center', va='bottom', rotation=45 if v < 0.01 else 0)
                
                for j, v in enumerate(param_values_set2):
                    ax.text(j + width/2, v * 1.05, f'{v:.6f}' if v < 0.01 else f'{v:.4f}', 
                           ha='center', va='bottom', rotation=45 if v < 0.01 else 0)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'{element}_params_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 将参数和效果写入文本文件
            with open(os.path.join(plot_dir, f'{element}_hyperopt_summary.txt'), 'w') as f:
                f.write(f"===== {element} 超参数优化结果汇总 =====\n\n")
                
                f.write("第一组最佳参数:\n")
                for param, value in best_params_set1.items():
                    f.write(f"  {param}: {value}\n")
                f.write(f"验证损失: {best_loss_set1}\n\n")
                
                f.write("第二组最佳参数:\n")
                for param, value in best_params_set2.items():
                    f.write(f"  {param}: {value}\n")
                f.write(f"验证损失: {best_loss_set2}\n")
            
            logger.info(f"已生成 {element} 的批量超参数优化可视化结果")
            return True
            
        except Exception as e:
            logger.error(f"处理批量超参数优化结果时出错: {str(e)}")
            # 继续尝试常规的hyperopt结果
    
    # 如果没有批量优化结果，尝试加载传统hyperopt结果
    final_path = os.path.join(results_dir, f'{element}_hyperopt_final.pkl')
    if not os.path.exists(final_path):
        logger.warning(f"找不到 {element} 的最终超参数结果: {final_path}")
        return
    
    try:
        with open(final_path, 'rb') as f:
            data = pickle.load(f)
        
        best_params_stage1 = data.get('best_params_stage1', {})
        best_params_final = data.get('best_params_final', {})
        trials_stage1 = data.get('trials_stage1', None)
        trials_stage2 = data.get('trials_stage2', None)
        
        # 绘制损失趋势图
        plt.figure(figsize=(12, 6))
        
        # 第一阶段损失
        if trials_stage1:
            losses_stage1 = [t['result']['loss'] for t in trials_stage1.trials if t['result']['loss'] < float('inf')]
            xs = np.arange(len(losses_stage1))
            ys = losses_stage1
            best_ys = np.minimum.accumulate(ys)
            
            plt.subplot(1, 2, 1)
            plt.plot(xs, ys, 'o--', markersize=4, alpha=0.8, label='第一阶段损失')
            plt.plot(xs, best_ys, '-', label='第一阶段最佳损失')
            plt.title(f"{element} 第一阶段超参数优化", fontsize=14)
            plt.xlabel('评估次数', fontsize=12)
            plt.ylabel('验证损失', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # 第二阶段损失
        if trials_stage2:
            losses_stage2 = [t['result']['loss'] for t in trials_stage2.trials if t['result']['loss'] < float('inf')]
            xs = np.arange(len(losses_stage2))
            ys = losses_stage2
            best_ys = np.minimum.accumulate(ys)
            
            plt.subplot(1, 2, 2)
            plt.plot(xs, ys, 'o--', markersize=4, alpha=0.8, label='第二阶段损失', color='orange')
            plt.plot(xs, best_ys, '-', label='第二阶段最佳损失', color='red')
            plt.title(f"{element} 第二阶段超参数优化", fontsize=14)
            plt.xlabel('评估次数', fontsize=12)
            plt.ylabel('验证损失', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{element}_hyperopt_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制超参数分布
        plt.figure(figsize=(15, 10))
        param_names = ['lr', 'weight_decay', 'batch_size', 'dropout_rate']
        
        for i, param in enumerate(param_names):
            if param in best_params_stage1 and param in best_params_final:
                plt.subplot(2, 2, i+1)
                
                # 准备第一阶段数据
                if trials_stage1:
                    param_values = []
                    losses = []
                    
                    for t in trials_stage1.trials:
                        if 'loss' in t['result'] and t['result']['loss'] < float('inf'):
                            if param == 'batch_size':
                                # batch_size是选择项，需要特殊处理
                                values = [16, 32, 64, 128]
                                val = values[t['misc']['vals']['batch_size'][0]] if t['misc']['vals']['batch_size'] else None
                            elif param == 'lr' or param == 'weight_decay':
                                # 对数参数需要转换回实数
                                val = np.exp(t['misc']['vals'][param][0]) if t['misc']['vals'][param] else None
                            else:
                                val = t['misc']['vals'][param][0] if t['misc']['vals'][param] else None
                                
                            if val is not None:
                                param_values.append(val)
                                losses.append(t['result']['loss'])
                    
                    if param_values:
                        sc = plt.scatter(param_values, losses, alpha=0.6, label='第一阶段')
                        plt.axvline(x=best_params_stage1[param], color='blue', linestyle='--', 
                                  label=f'第一阶段最佳: {best_params_stage1[param]:.6f}' if param != 'batch_size' else f'第一阶段最佳: {best_params_stage1[param]}')
                
                # 准备第二阶段数据
                if trials_stage2:
                    param_values = []
                    losses = []
                    
                    for t in trials_stage2.trials:
                        if 'loss' in t['result'] and t['result']['loss'] < float('inf'):
                            if param == 'batch_size':
                                # 第二阶段batch_size通常是固定的
                                val = best_params_stage1['batch_size']  
                            elif param == 'lr' or param == 'weight_decay':
                                # 对数参数需要转换回实数
                                val = np.exp(t['misc']['vals'][param][0]) if t['misc']['vals'][param] else None
                            else:
                                val = t['misc']['vals'][param][0] if t['misc']['vals'][param] else None
                                
                            if val is not None:
                                param_values.append(val)
                                losses.append(t['result']['loss'])
                    
                    if param_values:
                        sc = plt.scatter(param_values, losses, alpha=0.6, color='orange', label='第二阶段')
                        plt.axvline(x=best_params_final[param], color='red', linestyle='--', 
                                  label=f'最终最佳: {best_params_final[param]:.6f}' if param != 'batch_size' else f'最终最佳: {best_params_final[param]}')
                
                # 图表标题和标签
                plt.title(f"{element} - {param} 分布", fontsize=14)
                plt.xlabel(param, fontsize=12)
                plt.ylabel('验证损失', fontsize=12)
                plt.grid(True, alpha=0.3)
                
                # 对数尺度(对lr和weight_decay)
                if param == 'lr' or param == 'weight_decay':
                    plt.xscale('log')
                
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{element}_hyperopt_params.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 将参数和效果写入文本文件
        with open(os.path.join(plot_dir, f'{element}_hyperopt_summary.txt'), 'w') as f:
            f.write(f"===== {element} 超参数优化结果汇总 =====\n\n")
            
            f.write("第一阶段最佳参数:\n")
            for param, value in best_params_stage1.items():
                f.write(f"  {param}: {value}\n")
            
            f.write("\n最终最佳参数:\n")
            for param, value in best_params_final.items():
                f.write(f"  {param}: {value}\n")
            
            # 如果第一阶段有最佳试验
            if trials_stage1 and trials_stage1.best_trial:
                best_loss_stage1 = trials_stage1.best_trial['result']['loss']
                f.write(f"\n第一阶段最佳验证损失: {best_loss_stage1:.6f}\n")
            
            # 如果第二阶段有最佳试验
            if trials_stage2 and trials_stage2.best_trial:
                best_loss_stage2 = trials_stage2.best_trial['result']['loss']
                f.write(f"\n最终最佳验证损失: {best_loss_stage2:.6f}\n")
        
        logger.info(f"已生成 {element} 的超参数优化可视化结果")
        return True
        
    except Exception as e:
        logger.error(f"生成 {element} 超参数优化可视化时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def evaluate_model(model, test_loader, device=None):
    """
    在测试集上评估模型，并为每个批次生成单独的评估结果
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
    Returns:
        评估结果字典
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model.eval()
    test_loss = 0
    predictions = []
    targets = []
    
    criterion = nn.MSELoss()
    
    # 创建批次结果目录
    batch_results_dir = os.path.join('results', 'evaluation', f'evaluation_{model.element}_batch_results')
    os.makedirs(batch_results_dir, exist_ok=True)
    
    # 创建批次追踪文件
    batch_tracking_path = os.path.join(batch_results_dir, 'batch_tracking.csv')
    if os.path.exists(batch_tracking_path):
        batch_df = pd.read_csv(batch_tracking_path)
    else:
        batch_df = pd.DataFrame(columns=['batch_id', 'mse', 'rmse', 'mae', 'r2', 'scatter', 'timestamp'])
    
    # 分批次评估
    with torch.no_grad():
        for batch_idx, (spectra, abundances) in enumerate(test_loader):
            try:
                spectra = spectra.to(device)
                abundances = abundances.to(device)
                
                outputs = model(spectra)
                loss = criterion(outputs.squeeze(), abundances)
                
                test_loss += loss.item()
                batch_predictions = outputs.squeeze().cpu().numpy()
                batch_targets = abundances.cpu().numpy()
                
                predictions.extend(batch_predictions)
                targets.extend(batch_targets)
                
                # 为当前批次生成评估结果
                batch_mse = np.mean((batch_predictions - batch_targets) ** 2)
                batch_rmse = np.sqrt(batch_mse)
                batch_mae = np.mean(np.abs(batch_predictions - batch_targets))
                if np.var(batch_targets) == 0:  # 防止除零错误
                    batch_r2 = 0
                else:
                    batch_r2 = 1 - (np.sum((batch_targets - batch_predictions) ** 2) / np.sum((batch_targets - np.mean(batch_targets)) ** 2))
                batch_scatter = np.std(batch_predictions - batch_targets)
                
                # 保存批次评估指标
                metrics_path = os.path.join(batch_results_dir, f'batch_{batch_idx+1}_metrics.txt')
                with open(metrics_path, 'w') as f:
                    f.write(f"批次 {batch_idx+1} 在 {model.element} 上的评估结果\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"样本数: {len(batch_targets)}\n")
                    f.write(f"MSE: {batch_mse:.6f}\n")
                    f.write(f"RMSE: {batch_rmse:.6f}\n")
                    f.write(f"MAE: {batch_mae:.6f}\n")
                    f.write(f"R²: {batch_r2:.6f}\n")
                    f.write(f"散度: {batch_scatter:.6f}\n")
                
                # 生成批次散点图
                plt.figure(figsize=(10, 6))
                plt.scatter(batch_targets, batch_predictions, alpha=0.5)
                plt.plot([min(batch_targets), max(batch_targets)], [min(batch_targets), max(batch_targets)], 'r--')
                plt.xlabel('真实值')
                plt.ylabel('预测值')
                plt.title(f'批次 {batch_idx+1} 预测 vs 真实值 (RMSE: {batch_rmse:.4f})')
                plt.grid(True)
                plt.tight_layout()
                scatter_path = os.path.join(batch_results_dir, f'batch_{batch_idx+1}_scatter.png')
                plt.savefig(scatter_path)
                plt.close()
                
                # 生成误差分布图
                plt.figure(figsize=(10, 6))
                errors = batch_predictions - batch_targets
                plt.hist(errors, bins=30, alpha=0.7)
                plt.axvline(x=0, color='r', linestyle='--')
                plt.xlabel('预测误差')
                plt.ylabel('频率')
                plt.title(f'批次 {batch_idx+1} 预测误差分布 (MAE: {batch_mae:.4f})')
                plt.grid(True)
                plt.tight_layout()
                hist_path = os.path.join(batch_results_dir, f'batch_{batch_idx+1}_error_hist.png')
                plt.savefig(hist_path)
                plt.close()
                
                # 更新批次追踪文件
                new_row = pd.DataFrame({
                    'batch_id': [batch_idx+1],
                    'mse': [batch_mse],
                    'rmse': [batch_rmse],
                    'mae': [batch_mae],
                    'r2': [batch_r2],
                    'scatter': [batch_scatter],
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
                    trend_path = os.path.join(batch_results_dir, 'batch_trends.png')
                    plt.savefig(trend_path)
                    plt.close()
                
                logger.info(f"成功生成批次 {batch_idx+1} 的评估结果和可视化")
                
            except Exception as e:
                logger.error(f"评估批次 {batch_idx+1} 时出错: {str(e)}")
    
    # 完成后生成总结报告
    if len(predictions) > 0 and len(targets) > 0:
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # 计算评估指标
        mae = np.mean(np.abs(predictions - targets))
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2))
        std_diff = np.std(predictions - targets)
        
        # 保存最终评估报告
        summary_path = os.path.join(batch_results_dir, 'evaluation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"{model.element} 元素评估总结\n")
            f.write("=" * 50 + "\n")
            f.write(f"总批次数: {len(batch_df)}\n")
            f.write(f"总样本数: {len(targets)}\n")
            f.write(f"平均 MSE: {mse:.6f}\n")
            f.write(f"平均 RMSE: {rmse:.6f}\n")
            f.write(f"平均 MAE: {mae:.6f}\n")
            f.write(f"总体 R²: {r2:.6f}\n")
            f.write(f"总体散度: {std_diff:.6f}\n")
            
            # 找出最佳批次
            if len(batch_df) > 0:
                best_batch_id = batch_df.loc[batch_df['rmse'].idxmin(), 'batch_id']
                best_rmse = batch_df['rmse'].min()
                f.write(f"\n最佳批次: {best_batch_id} (RMSE: {best_rmse:.6f})\n")
        
        # 生成总体散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(targets, predictions, alpha=0.5)
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(f'总体预测 vs 真实值 (RMSE: {rmse:.4f})')
        plt.grid(True)
        plt.tight_layout()
        total_scatter_path = os.path.join(batch_results_dir, 'total_scatter.png')
        plt.savefig(total_scatter_path)
        plt.close()
        
        # 生成总体误差分布图
        plt.figure(figsize=(10, 6))
        errors = predictions - targets
        plt.hist(errors, bins=30, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('预测误差')
        plt.ylabel('频率')
        plt.title(f'总体预测误差分布 (MAE: {mae:.4f})')
        plt.grid(True)
        plt.tight_layout()
        total_hist_path = os.path.join(batch_results_dir, 'total_error_hist.png')
        plt.savefig(total_hist_path)
        plt.close()
        
        logger.info(f"已生成评估总结报告: {summary_path}")
        
        return {
            'test_loss': test_loss / len(test_loader),
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'dex': std_diff,
            'predictions': predictions,
            'targets': targets,
            'summary_path': summary_path,
            'batch_results_dir': batch_results_dir
        }
    else:
        logger.error("评估过程中没有收集到有效的预测结果")
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='元素丰度预测模型评估工具')
    parser.add_argument('--elements', nargs='+', default=None, help='要评估的元素列表')
    parser.add_argument('--save_dir', type=str, default=None, help='结果保存目录')
    parser.add_argument('--catalog_eval', action='store_true', help='执行星表评估')
    parser.add_argument('--catalogs', nargs='+', default=['galah', 'lasp'], help='要评估的星表类型列表')
    parser.add_argument('--no_plots', action='store_true', help='不生成图像')
    parser.add_argument('--clear_cache', action='store_true', help='清除所有缓存')
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(config.output_config['results_dir'], exist_ok=True)
    os.makedirs(config.output_config['plots_dir'], exist_ok=True)
    
    # 处理缓存
    if args.clear_cache:
        cache_manager.clear_cache()
        logger.info("已清除所有缓存")
    else:
        ask_clear_cache(cache_manager)
    
    logger.info("开始元素丰度评估")
    
    # 检查是否执行星表评估
    if args.catalog_eval:
        logger.info("执行星表评估")
        evaluate_catalogs_all_elements(
            elements=args.elements,
            catalog_types=args.catalogs,
            save_plots=not args.no_plots
        )
        logger.info("星表评估完成")
    else:
        # 执行测试集评估
        logger.info("执行测试集评估")
        results = evaluate_all_elements(
            elements=args.elements,
            save_dir=args.save_dir
        )
        logger.info("测试集评估完成")
        
        # 绘制评估结果图像
        if not args.no_plots:
            logger.info("生成评估结果图像")
            plot_predictions_vs_true(args.elements)
            plot_metrics_comparison(args.elements)
            logger.info("图像生成完成")
    
    logger.info("评估过程全部完成")

if __name__ == "__main__":
    main()