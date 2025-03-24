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

# 导入自定义模块
import config
from model import load_trained_model, predict
from utils import CacheManager, ProgressManager, ask_clear_cache

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
    加载待预测的光谱数据
    
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
            data = np.load(data_path)
            logger.info(f"NPZ文件包含以下键: {data.files}")
            
            # 尝试不同的键名
            if 'spectra' in data.files:
                spectra = data['spectra']
            elif 'data' in data.files:
                spectra = data['data']
            elif 'X' in data.files:
                spectra = data['X']
            else:
                logger.error(f"NPZ文件中未找到光谱数据，可用的键: {data.files}")
                return None, None
            
            if spectra is None:
                logger.error("无法从NPZ文件中提取光谱数据")
                return None, None
                
            logger.info(f"成功从NPZ文件加载光谱数据，形状: {spectra.shape}")
            
            # 保存到缓存
            cache_manager.set_cache(cache_key, {
                'spectra': spectra,
                'metadata': None
            })
            
            return spectra, None
            
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            logger.info(f"CSV文件包含以下列: {df.columns.tolist()}")
            
            # 假设CSV文件中包含spec列和其他元数据列
            if 'spec' not in df.columns:
                logger.error("CSV文件缺少'spec'列")
                return None, None
            
            # 从CSV的spec列解析光谱数据
            spectra = []
            for spec_str in df['spec']:
                try:
                    # 假设光谱数据存储为空格分隔的数字字符串
                    spectrum = np.array([float(x) for x in spec_str.split()])
                    spectra.append(spectrum)
                except:
                    logger.warning("解析光谱数据失败，尝试其他方法")
                    # 如果失败，尝试其他解析方式
                    try:
                        spectrum = np.array(eval(spec_str))
                        spectra.append(spectrum)
                    except:
                        logger.error("无法解析光谱数据")
                        return None, None
            
            spectra = np.array(spectra)
            logger.info(f"成功从CSV文件加载光谱数据，形状: {spectra.shape}")
            
            # 获取元数据（除spec列外的所有列）
            metadata = df.drop('spec', axis=1) if len(df.columns) > 1 else None
            
            # 保存到缓存
            cache_manager.set_cache(cache_key, {
                'spectra': spectra,
                'metadata': metadata
            })
            
            return spectra, metadata
            
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

def predict_all_elements(spectra, elements=None, batch_size=32):
    """
    预测所有元素的丰度
    
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
    
    # 使用进度管理器
    with ProgressManager(len(elements), desc="预测元素丰度") as progress:
        for element in elements:
            logger.info(f"预测 {element} 元素丰度...")
            predictions = predict_element(spectra, element, batch_size)
            if predictions is not None:
                results[element] = predictions
                logger.info(f"完成 {element} 预测: 均值={predictions.mean():.4f}, 标准差={predictions.std():.4f}")
            
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