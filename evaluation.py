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
        pandas.DataFrame: 包含spec、teff和logg的DataFrame
    """
    # 检查缓存
    cache_key = f"metadata_{os.path.basename(csv_file)}"
    cached_metadata = cache_manager.get_cache(cache_key)
    if cached_metadata is not None:
        logger.info(f"从缓存加载元数据: {csv_file}")
        return cached_metadata['metadata']
    
    try:
        df = pd.read_csv(csv_file)
        if 'spec' not in df.columns or 'teff' not in df.columns or 'logg' not in df.columns:
            logger.error(f"CSV文件 {csv_file} 缺少必要的列")
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
        elements = config.training_config['elements']
    
    if csv_files is None:
        csv_files = {
            'C_FE': config.data_paths.get('C_FE_csv', ''),
            'MG_FE': config.data_paths.get('MG_FE_csv', ''),
            'CA_FE': config.data_paths.get('CA_FE_csv', '')
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
                    'spec': range(num_samples),
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
        pd.DataFrame: 包含光谱文件名和真实元素丰度的数据框
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
        if 'spec' not in df.columns:
            logger.error(f"星表文件 {catalog_file} 中缺少'spec'列")
            return None
            
        # 确保元素丰度列在最后一列
        abundance_col = df.columns[-1]
        logger.info(f"元素丰度列名: {abundance_col}")
        
        # 提取只需要的列：光谱文件名和元素丰度
        result_df = df[['spec', abundance_col]].copy()
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
        spectra_df (pd.DataFrame): 包含光谱文件名的数据框
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
                spec_name = row['spec']
                fits_file = os.path.join(fits_dir, f"{spec_name}.fits.gz")
                
                if not os.path.exists(fits_file):
                    # 尝试其他可能的后缀
                    alt_fits_file = os.path.join(fits_dir, f"{spec_name}.fits")
                    if os.path.exists(alt_fits_file):
                        fits_file = alt_fits_file
                    else:
                        logger.warning(f"找不到FITS文件: {fits_file}")
                        continue
                
                # 读取并预处理光谱
                wavelength, flux = preprocessor.read_fits_file(fits_file)
                if wavelength is None or flux is None:
                    logger.warning(f"无法读取光谱: {fits_file}")
                    continue
                    
                # 预处理光谱
                # 去噪
                flux = preprocessor.denoise_spectrum(wavelength, flux)
                flux = preprocessor.denoise_spectrum_second(wavelength, flux)
                
                # 重采样
                resampled_flux = preprocessor.resample_spectrum(wavelength, flux)
                if resampled_flux is None:
                    logger.warning(f"无法重采样光谱: {fits_file}")
                    continue
                    
                # 归一化
                normalized_flux = preprocessor.normalize_spectrum(resampled_flux)
                if normalized_flux is None:
                    logger.warning(f"无法归一化光谱: {fits_file}")
                    continue
                
                # 转换为张量并预测
                input_tensor = torch.tensor(normalized_flux, dtype=torch.float32).unsqueeze(0)
                abundance_pred = predict(model, input_tensor, config.training_config['device'])
                
                # 保存结果
                results.append({
                    'spec_name': spec_name,
                    'true_abundance': row['true_abundance'],
                    'predicted_abundance': float(abundance_pred)
                })
                
                # 更新进度
                progress.update(1)
                    
            except Exception as e:
                logger.error(f"处理光谱 {row['spec']} 时出错: {e}")
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
    merged_df = pd.merge(pred_df, catalog_df[['spec', 'teff', 'logg']], 
                         left_on='spec_name', right_on='spec', how='inner')
    
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