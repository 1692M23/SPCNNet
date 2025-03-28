#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
预测模块：使用训练好的模型预测恒星光谱元素丰度
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import time

# 导入自定义模块
import config
from model import load_trained_model, predict
from multi_element_processor import MultiElementProcessor
from utils import CacheManager, ProgressManager, ask_clear_cache
from config import Config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.output_config['log_dir'], 'predict.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('predict')

# 创建缓存管理器
cache_manager = CacheManager(cache_dir=os.path.join(config.output_config['cache_dir'], 'predict'))

def load_data(data_path):
    """
    加载待预测的光谱数据，支持preprocessdata7.py处理的数据格式
    
    参数:
        data_path (str): 数据文件路径，支持.npz和.csv格式
        
    返回:
        numpy.ndarray: 光谱数据
        pandas.DataFrame: 元数据（如果有）
    """
    # 检查缓存
    cache_key = f"data_{os.path.basename(data_path)}"
    cached_data = cache_manager.get_cache(cache_key)
    if cached_data is not None:
        logger.info(f"从缓存加载数据: {data_path}")
        return cached_data['spectra'], cached_data['metadata']
    
    try:
        if not os.path.exists(data_path):
            logger.error(f"数据文件不存在: {data_path}")
            return None, None
            
        if data_path.endswith('.npz'):
            data = np.load(data_path, allow_pickle=True)
            logger.info(f"NPZ文件包含以下键: {data.files}")
            
            # 尝试不同的键名 - 增加对preprocessdata7.py格式的支持
            if 'spectra' in data.files:
                spectra = data['spectra']
            elif 'data' in data.files:
                spectra = data['data']
            elif 'X' in data.files:
                spectra = data['X']
            elif 'flux' in data.files:
                spectra = data['flux']
            elif 'wavelength' in data.files and 'flux' in data.files:
                # 如果分别存储了波长和流量，可以选择合并或只使用流量
                logger.info("找到分离的波长和流量数据")
                spectra = data['flux']
            else:
                # 尝试使用第一个数组
                first_key = list(data.keys())[0]
                logger.warning(f"未找到标准光谱数据键，尝试使用第一个键 {first_key}")
                spectra = data[first_key]
            
            if spectra is None:
                logger.error("无法从NPZ文件中提取光谱数据")
                return None, None
                
            logger.info(f"成功从NPZ文件加载光谱数据，形状: {spectra.shape}")
            
            # 提取可能的元数据
            metadata = None
            if 'metadata' in data.files:
                metadata = data['metadata']
            elif 'obsid' in data.files or 'spec' in data.files:
                # 创建元数据DataFrame
                id_key = 'obsid' if 'obsid' in data.files else 'spec'
                id_values = data[id_key]
                metadata = pd.DataFrame({id_key: id_values})
                logger.info(f"从NPZ文件创建元数据DataFrame，包含{len(metadata)}行")
            
            # 保存到缓存
            cache_manager.set_cache(cache_key, {
                'spectra': spectra,
                'metadata': metadata
            })
            
            return spectra, metadata
            
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            logger.info(f"CSV文件包含以下列: {df.columns.tolist()}")
            
            # 兼容preprocessdata7.py，检查obsid或spec列
            id_col = None
            if 'obsid' in df.columns:
                id_col = 'obsid'
            elif 'spec' in df.columns:
                id_col = 'spec'
                
            if id_col is None:
                logger.error("CSV文件缺少'obsid'或'spec'列，需要识别符来匹配FITS文件")
                return None, None
            
            # 尝试从preprocessdata7.py获取处理好的光谱数据
            try:
                # 优先尝试导入preprocessdata7模块
                import importlib
                try:
                    pp7 = importlib.import_module('preprocessdata7')
                    if hasattr(pp7, 'LAMOSTPreprocessor'):
                        logger.info("使用preprocessdata7.LAMOSTPreprocessor处理FITS数据")
                        preprocessor = pp7.LAMOSTPreprocessor()
                        
                        # 获取所有ID
                        ids = df[id_col].values
                        logger.info(f"从CSV文件加载了{len(ids)}个{id_col}")
                        
                        # 处理每个ID对应的光谱
                        spectra = []
                        valid_indices = []
                        
                        for i, id_value in enumerate(ids):
                            try:
                                # 查找并读取FITS文件
                                fits_file = preprocessor._find_fits_file(id_value)
                                if fits_file is None:
                                    logger.warning(f"找不到ID为{id_value}的FITS文件")
                                    continue
                                    
                                # 处理光谱数据
                                processed_data = preprocessor.process_single_spectrum(fits_file, None)
                                if processed_data and 'spectrum' in processed_data:
                                    spectra.append(processed_data['spectrum'])
                                    valid_indices.append(i)
                                else:
                                    logger.warning(f"处理ID为{id_value}的光谱失败")
                            except Exception as e:
                                logger.warning(f"处理ID为{id_value}的光谱时出错: {str(e)}")
                        
                        if not spectra:
                            logger.error("没有成功处理任何光谱")
                            return None, None
                            
                        spectra = np.array(spectra)
                        logger.info(f"成功处理{len(spectra)}个光谱，形状: {spectra.shape}")
                        
                        # 获取有效的元数据
                        metadata = df.iloc[valid_indices].reset_index(drop=True) if len(df.columns) > 1 else None
                        
                        # 保存到缓存
                        cache_manager.set_cache(cache_key, {
                            'spectra': spectra,
                            'metadata': metadata
                        })
                        
                        return spectra, metadata
                except (ImportError, ModuleNotFoundError):
                    logger.warning("无法导入preprocessdata7模块，尝试使用原始方法")
                    
                # 如果无法使用preprocessdata7，回退到原始方法
                # 从preprocessdata.py导入相关功能
                try:
                    from preprocessdata7 import LAMOSTPreprocessor
                    preprocessor = LAMOSTPreprocessor()
                    
                    # 获取所有OBSID
                    obsids = df[id_col].values
                    logger.info(f"从CSV文件加载了{len(obsids)}个{id_col}")
                    
                    # 处理每个OBSID对应的光谱
                    spectra = []
                    valid_indices = []
                    
                    for i, obsid in enumerate(obsids):
                        try:
                            # 查找并读取FITS文件
                            fits_file = preprocessor._find_fits_file(obsid)
                            if fits_file is None:
                                logger.warning(f"找不到ID为{obsid}的FITS文件")
                                continue
                                
                            # 读取和预处理光谱
                            wavelength, flux, _ = preprocessor.read_fits_file(fits_file)
                            if wavelength is None or flux is None:
                                logger.warning(f"无法读取FITS文件: {fits_file}")
                                continue
                            
                            # 预处理光谱数据
                            processed_data = preprocessor.process_single_spectrum(obsid, None)
                            if processed_data and 'spectrum' in processed_data:
                                spectra.append(processed_data['spectrum'])
                                valid_indices.append(i)
                            else:
                                logger.warning(f"处理ID为{obsid}的光谱失败")
                        except Exception as e:
                            logger.warning(f"处理ID为{obsid}的光谱时出错: {str(e)}")
                    
                    if not spectra:
                        logger.error("没有成功处理任何光谱")
                        return None, None
                        
                    spectra = np.array(spectra)
                    logger.info(f"成功处理{len(spectra)}个光谱，形状: {spectra.shape}")
                    
                    # 获取有效的元数据
                    metadata = df.iloc[valid_indices].reset_index(drop=True) if len(df.columns) > 1 else None
                    
                    # 保存到缓存
                    cache_manager.set_cache(cache_key, {
                        'spectra': spectra,
                        'metadata': metadata
                    })
                    
                    return spectra, metadata
                    
                except (ImportError, ModuleNotFoundError):
                    logger.error("无法导入预处理模块，无法处理FITS文件")
                    return None, None
            except Exception as e:
                logger.error(f"处理CSV文件中的光谱数据失败: {str(e)}")
                import traceback
                traceback.print_exc()
                return None, None
            
        else:
            logger.error(f"不支持的文件格式: {data_path}")
            return None, None
            
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        logger.exception("详细错误信息:")
        return None, None

def predict_element(spectra, element, batch_size=32, device=None):
    """
    使用训练好的模型预测特定元素的丰度
    
    参数:
        spectra (numpy.ndarray): 光谱数据，形状为 [n_samples, n_wavelengths]
        element (str): 元素名称
        batch_size (int): 批次大小
        device (str): 设备名称，默认为配置中指定的设备
        
    返回:
        numpy.ndarray: 预测结果
    """
    # 检查缓存
    cache_key = f"predictions_{element}_{spectra.shape}"
    cached_predictions = cache_manager.get_cache(cache_key)
    if cached_predictions is not None:
        logger.info(f"从缓存加载预测结果: {element}")
        return cached_predictions['predictions']
    
    if device is None:
        device = config.training_config['device']
    
    # 加载模型
    model_path = os.path.join(config.model_config['model_dir'], f"{element}_model.pth")
    if not os.path.exists(model_path):
        logger.error(f"找不到 {element} 的模型文件: {model_path}")
        return None
    
    model = load_trained_model(element, model_path, device)
    model.eval()
    
    # 准备数据
    predictions = []
    
    # 分批次预测
    with torch.no_grad():
        for i in range(0, len(spectra), batch_size):
            batch_data = spectra[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch_data).unsqueeze(1).to(device)  # [batch_size, 1, n_wavelengths]
            batch_predictions = model(batch_tensor).cpu().numpy().flatten()
            predictions.extend(batch_predictions)
    
    predictions = np.array(predictions)
    
    # 保存到缓存
    cache_manager.set_cache(cache_key, {
        'predictions': predictions
    })
    
    return predictions

def predict_all_elements(spectra, elements=None, batch_size=Config.PREDICTION_BATCH_SIZE):
    """
    预测所有元素的丰度，并为每批次生成实时结果
    
    参数:
        spectra (numpy.ndarray): 光谱数据
        elements (list): 要预测的元素列表，默认为配置中指定的所有元素
        batch_size (int): 批次大小
        
    返回:
        dict: 包含各元素预测结果的字典
    """
    if elements is None:
        elements = config.training_config['elements']
    
    results = {}
    uncertainties = {}
    
    # 使用进度管理器
    with ProgressManager(len(elements), desc="预测元素丰度") as progress:
        for element in elements:
            logger.info(f"预测 {element} 元素丰度...")
            
            # 创建批次结果目录
            batch_results_dir = os.path.join(config.output_config['results_dir'], f'prediction_{element}_batch_results')
            os.makedirs(batch_results_dir, exist_ok=True)
            
            # 初始化批次追踪文件
            batch_tracking_path = os.path.join(batch_results_dir, 'batch_tracking.csv')
            if os.path.exists(batch_tracking_path):
                batch_df = pd.read_csv(batch_tracking_path)
            else:
                batch_df = pd.DataFrame(columns=['batch_id', 'mean', 'std', 'min', 'max', 'timestamp'])
            
            # 收集预测结果
            all_predictions = []
            
            # 分批次进行预测并生成结果
            for batch_id, i in enumerate(range(0, len(spectra), batch_size)):
                try:
                    # 获取当前批次数据
                    batch_data = spectra[i:i+batch_size]
                    
                    # 预测当前批次
                    batch_predictions = predict_element(batch_data, element, batch_size)
                    if batch_predictions is None:
                        logger.warning(f"批次 {batch_id+1} 预测失败，跳过")
                        continue
                    
                    # 添加到总预测结果
                    all_predictions.extend(batch_predictions)
                    
                    # 计算当前批次的统计信息
                    batch_mean = np.mean(batch_predictions)
                    batch_std = np.std(batch_predictions)
                    batch_min = np.min(batch_predictions)
                    batch_max = np.max(batch_predictions)
                    
                    # 保存批次评估指标
                    metrics_path = os.path.join(batch_results_dir, f'batch_{batch_id+1}_metrics.txt')
                    with open(metrics_path, 'w') as f:
                        f.write(f"批次 {batch_id+1} 在 {element} 上的预测结果\n")
                        f.write("=" * 50 + "\n")
                        f.write(f"样本数: {len(batch_predictions)}\n")
                        f.write(f"平均值: {batch_mean:.6f}\n")
                        f.write(f"标准差: {batch_std:.6f}\n")
                        f.write(f"最小值: {batch_min:.6f}\n")
                        f.write(f"最大值: {batch_max:.6f}\n")
                    
                    # 生成批次预测分布图
                    plt.figure(figsize=(10, 6))
                    plt.hist(batch_predictions, bins=30, alpha=0.7)
                    plt.axvline(x=batch_mean, color='r', linestyle='--', label=f'平均值 = {batch_mean:.4f}')
                    plt.axvline(x=batch_mean + batch_std, color='g', linestyle='--', label=f'+1σ = {batch_mean + batch_std:.4f}')
                    plt.axvline(x=batch_mean - batch_std, color='g', linestyle='--', label=f'-1σ = {batch_mean - batch_std:.4f}')
                    plt.xlabel(f'{element} 预测值')
                    plt.ylabel('频率')
                    plt.title(f'批次 {batch_id+1} {element} 预测分布')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    dist_path = os.path.join(batch_results_dir, f'batch_{batch_id+1}_distribution.png')
                    plt.savefig(dist_path)
                    plt.close()
                    
                    # 更新批次追踪文件
                    new_row = pd.DataFrame({
                        'batch_id': [batch_id+1],
                        'mean': [batch_mean],
                        'std': [batch_std],
                        'min': [batch_min],
                        'max': [batch_max],
                        'timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')]
                    })
                    
                    batch_df = pd.concat([batch_df, new_row], ignore_index=True)
                    batch_df.to_csv(batch_tracking_path, index=False)
                    
                    # 生成批次进度趋势图
                    if len(batch_df) > 1:
                        plt.figure(figsize=(12, 8))
                        
                        plt.subplot(2, 2, 1)
                        plt.plot(batch_df['batch_id'], batch_df['mean'], 'o-')
                        plt.xlabel('批次ID')
                        plt.ylabel('平均值')
                        plt.title('平均值趋势')
                        plt.grid(True)
                        
                        plt.subplot(2, 2, 2)
                        plt.plot(batch_df['batch_id'], batch_df['std'], 'o-')
                        plt.xlabel('批次ID')
                        plt.ylabel('标准差')
                        plt.title('标准差趋势')
                        plt.grid(True)
                        
                        plt.subplot(2, 2, 3)
                        plt.plot(batch_df['batch_id'], batch_df['min'], 'o-')
                        plt.xlabel('批次ID')
                        plt.ylabel('最小值')
                        plt.title('最小值趋势')
                        plt.grid(True)
                        
                        plt.subplot(2, 2, 4)
                        plt.plot(batch_df['batch_id'], batch_df['max'], 'o-')
                        plt.xlabel('批次ID')
                        plt.ylabel('最大值')
                        plt.title('最大值趋势')
                        plt.grid(True)
                        
                        plt.tight_layout()
                        trend_path = os.path.join(batch_results_dir, 'batch_trends.png')
                        plt.savefig(trend_path)
                        plt.close()
                    
                    logger.info(f"成功生成批次 {batch_id+1} 的预测结果和可视化")
                
                except Exception as e:
                    logger.error(f"预测批次 {batch_id+1} 时出错: {str(e)}")
            
            # 完成该元素的所有批次预测后，生成总结报告
            if all_predictions:
                all_predictions = np.array(all_predictions)
                results[element] = all_predictions
                
                # 计算总体统计信息
                total_mean = np.mean(all_predictions)
                total_std = np.std(all_predictions)
                total_min = np.min(all_predictions)
                total_max = np.max(all_predictions)
                
                # 保存最终预测报告
                summary_path = os.path.join(batch_results_dir, 'prediction_summary.txt')
                with open(summary_path, 'w') as f:
                    f.write(f"{element} 元素预测总结\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"总批次数: {len(batch_df)}\n")
                    f.write(f"总样本数: {len(all_predictions)}\n")
                    f.write(f"总体平均值: {total_mean:.6f}\n")
                    f.write(f"总体标准差: {total_std:.6f}\n")
                    f.write(f"总体最小值: {total_min:.6f}\n")
                    f.write(f"总体最大值: {total_max:.6f}\n")
                
                # 生成总体预测分布图
                plt.figure(figsize=(10, 6))
                plt.hist(all_predictions, bins=30, alpha=0.7)
                plt.axvline(x=total_mean, color='r', linestyle='--', label=f'平均值 = {total_mean:.4f}')
                plt.axvline(x=total_mean + total_std, color='g', linestyle='--', label=f'+1σ = {total_mean + total_std:.4f}')
                plt.axvline(x=total_mean - total_std, color='g', linestyle='--', label=f'-1σ = {total_mean - total_std:.4f}')
                plt.xlabel(f'{element} 预测值')
                plt.ylabel('频率')
                plt.title(f'总体 {element} 预测分布')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                total_dist_path = os.path.join(batch_results_dir, 'total_distribution.png')
                plt.savefig(total_dist_path)
                plt.close()
                
                logger.info(f"已生成预测总结报告: {summary_path}")
                logger.info(f"{element} 预测完成: 均值={total_mean:.4f}, 标准差={total_std:.4f}")
            else:
                logger.error(f"{element} 预测失败，没有生成有效的预测结果")
            
            # 更新进度
            progress.update(1)
    
    return results

def save_predictions(predictions, metadata=None, output_dir=None):
    """
    保存预测结果
    
    参数:
        predictions (dict): 预测结果字典
        metadata (pandas.DataFrame): 元数据
        output_dir (str): 输出目录
    """
    if output_dir is None:
        output_dir = config.output_config['predictions_dir']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(predictions)
    
    # 如果有元数据，合并到结果中
    if metadata is not None:
        results_df = pd.concat([metadata, results_df], axis=1)
    
    # 保存结果
    output_file = os.path.join(output_dir, 'predictions.csv')
    results_df.to_csv(output_file, index=False)
    logger.info(f"已保存预测结果到: {output_file}")
    
    return output_file

def predict_multiple_elements(csv_file, elements=None, models=None, config=None):
    """预测多个元素丰度
    
    参数:
        csv_file: 包含obsid的CSV文件
        elements: 要预测的元素列表
        models: 训练好的模型字典
        config: 配置对象
    
    返回:
        predictions_df: 包含预测结果的DataFrame
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 验证obsid列是否存在
    if 'obsid' not in df.columns:
        raise ValueError(f"CSV文件必须包含obsid列，现有列: {df.columns}")
    
    # 初始化数据处理器
    processor = MultiElementProcessor(fits_dir=config.data_paths['fits_dir'])
    
    # 初始化结果字典
    predictions = {}
    uncertainties = {}
    
    # 处理每个obsid
    for i, obsid in enumerate(df['obsid']):
        # 获取光谱数据
        spectrum = processor.get_spectrum(obsid)
        if spectrum is None:
            logger.warning(f"无法处理OBSID为{obsid}的光谱，跳过")
            continue
        
        # 预测每个元素
        for element in elements:
            if element not in models:
                logger.warning(f"找不到{element}的模型，跳过")
                continue
                
            # 进行预测
            model = models[element]
            pred, unc = model.predict(spectrum[np.newaxis, :], 
                                     device=config.training_config['device'])
            
            # 添加到结果字典
            if element not in predictions:
                predictions[element] = np.zeros(len(df)) * np.nan
                uncertainties[element] = np.zeros(len(df)) * np.nan
                
            predictions[element][i] = pred
            uncertainties[element][i] = unc
    
    # 创建结果DataFrame
    result_df = df.copy()
    
    # 添加预测结果
    for element in elements:
        if element in predictions:
            result_df[f'{element}_pred'] = predictions[element]
            result_df[f'{element}_unc'] = uncertainties[element]
    
    return result_df

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='恒星光谱元素丰度预测')
    parser.add_argument('--data_path', type=str, required=True,
                        help='输入数据文件路径')
    parser.add_argument('--elements', nargs='+', default=None,
                        help='要预测的元素列表，默认为所有配置的元素')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--clear_cache', action='store_true',
                        help='清除所有缓存')
    
    args = parser.parse_args()
    
    # 处理缓存
    if args.clear_cache:
        cache_manager.clear_cache()
        logger.info("已清除所有缓存")
    else:
        ask_clear_cache(cache_manager)
    
    # 加载数据
    logger.info(f"加载数据: {args.data_path}")
    spectra, metadata = load_data(args.data_path)
    if spectra is None:
        logger.error("加载数据失败，退出程序")
        return
    
    # 预测元素丰度
    predictions = predict_all_elements(spectra, args.elements, args.batch_size)
    
    # 保存结果
    if predictions:
        output_file = save_predictions(predictions, metadata, args.output_dir)
        logger.info(f"预测完成，结果已保存到: {output_file}")
    else:
        logger.error("预测失败，没有生成任何结果")

if __name__ == '__main__':
    main() 