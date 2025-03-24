#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
预测模块：使用训练好的模型预测新的恒星光谱元素丰度
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

# 导入自定义模块
import config
from model import load_trained_model

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

def load_data(data_path):
    """
    加载待预测的光谱数据
    
    参数:
        data_path (str): 数据文件路径，支持.npz和.csv格式
        
    返回:
        numpy.ndarray: 光谱数据
        pandas.DataFrame: 元数据（如果有）
    """
    try:
        if data_path.endswith('.npz'):
            data = np.load(data_path)
            spectra = data['spectra'] if 'spectra' in data.files else None
            
            if spectra is None and 'data' in data.files:
                spectra = data['data']
            
            metadata = None
            
            logger.info(f"成功从NPZ文件加载 {spectra.shape[0]} 条光谱")
            return spectra, metadata
            
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            
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
            
            # 获取元数据（除spec列外的所有列）
            metadata = df.drop('spec', axis=1) if len(df.columns) > 1 else None
            
            logger.info(f"成功从CSV文件加载 {len(spectra)} 条光谱")
            return spectra, metadata
            
        else:
            logger.error(f"不支持的文件格式: {data_path}")
            return None, None
            
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
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
    
    return np.array(predictions)

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
    
    for element in elements:
        logger.info(f"预测 {element} 元素丰度...")
        predictions = predict_element(spectra, element, batch_size)
        if predictions is not None:
            results[element] = predictions
            logger.info(f"完成 {element} 预测: 均值={predictions.mean():.4f}, 标准差={predictions.std():.4f}")
    
    return results

def save_predictions(predictions, metadata=None, output_path=None, format='csv'):
    """
    保存预测结果
    
    参数:
        predictions (dict): 包含各元素预测结果的字典
        metadata (pandas.DataFrame): 元数据
        output_path (str): 输出文件路径
        format (str): 输出格式，支持'csv'和'npz'
        
    返回:
        bool: 是否成功保存
    """
    if not predictions:
        logger.error("没有可保存的预测结果")
        return False
    
    if output_path is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(config.output_config['predictions_dir'], f"predictions_{timestamp}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        if format.lower() == 'csv':
            # 创建DataFrame
            result_df = pd.DataFrame()
            
            # 添加预测结果
            for element, preds in predictions.items():
                result_df[element] = preds
            
            # 添加元数据
            if metadata is not None:
                for col in metadata.columns:
                    result_df[col] = metadata[col].values
            
            # 保存为CSV
            if not output_path.endswith('.csv'):
                output_path += '.csv'
            
            result_df.to_csv(output_path, index=False)
            logger.info(f"预测结果已保存至: {output_path}")
            
        elif format.lower() == 'npz':
            # 准备保存的数据
            save_data = {element: preds for element, preds in predictions.items()}
            
            # 添加元数据
            if metadata is not None:
                for col in metadata.columns:
                    save_data[col] = metadata[col].values
            
            # 保存为NPZ
            if not output_path.endswith('.npz'):
                output_path += '.npz'
            
            np.savez(output_path, **save_data)
            logger.info(f"预测结果已保存至: {output_path}")
            
        else:
            logger.error(f"不支持的输出格式: {format}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"保存预测结果失败: {str(e)}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='恒星光谱元素丰度预测')
    parser.add_argument('--input', type=str, required=True,
                        help='输入数据文件路径，支持.npz和.csv格式')
    parser.add_argument('--output', type=str, default=None,
                        help='输出结果文件路径')
    parser.add_argument('--elements', nargs='+', default=None,
                        help='要预测的元素列表，默认为所有配置的元素')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--format', type=str, choices=['csv', 'npz'], default='csv',
                        help='输出格式')
    
    args = parser.parse_args()
    
    # 加载数据
    logger.info(f"加载数据: {args.input}")
    spectra, metadata = load_data(args.input)
    
    if spectra is None:
        logger.error("加载数据失败，退出程序")
        return
    
    # 预测所有元素
    logger.info("开始预测...")
    predictions = predict_all_elements(spectra, args.elements, args.batch_size)
    
    if not predictions:
        logger.error("预测失败，退出程序")
        return
    
    # 保存结果
    logger.info("保存预测结果...")
    save_predictions(predictions, metadata, args.output, args.format)
    
    logger.info("预测完成")

def batch_predict_directory(input_dir, output_dir=None, elements=None, batch_size=32, format='csv'):
    """
    批量预测目录中的所有数据文件
    
    参数:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径
        elements (list): 要预测的元素列表
        batch_size (int): 批次大小
        format (str): 输出格式
    """
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        logger.error(f"输入目录不存在: {input_dir}")
        return
    
    if output_dir is None:
        output_dir = os.path.join(config.output_config['predictions_dir'], 'batch_predictions')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取目录中的所有.npz和.csv文件
    data_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.npz') or file.endswith('.csv'):
            data_files.append(os.path.join(input_dir, file))
    
    if not data_files:
        logger.warning(f"目录中没有找到.npz或.csv文件: {input_dir}")
        return
    
    # 批量预测
    successful = 0
    failed = 0
    
    for data_file in tqdm(data_files, desc="批量预测进度"):
        try:
            # 加载数据
            spectra, metadata = load_data(data_file)
            
            if spectra is None:
                logger.error(f"加载数据失败: {data_file}")
                failed += 1
                continue
            
            # 预测
            predictions = predict_all_elements(spectra, elements, batch_size)
            
            if not predictions:
                logger.error(f"预测失败: {data_file}")
                failed += 1
                continue
            
            # 生成输出文件名
            base_name = os.path.basename(data_file)
            output_name = os.path.splitext(base_name)[0] + f"_predictions.{format}"
            output_path = os.path.join(output_dir, output_name)
            
            # 保存结果
            success = save_predictions(predictions, metadata, output_path, format)
            
            if success:
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            logger.error(f"处理文件 {data_file} 时出错: {str(e)}")
            failed += 1
    
    logger.info(f"批量预测完成: 成功 {successful} 个文件, 失败 {failed} 个文件")

if __name__ == '__main__':
    # 检查是否使用批量预测模式
    if len(sys.argv) > 1 and sys.argv[1] == '--batch_dir':
        parser = argparse.ArgumentParser(description='批量预测恒星光谱元素丰度')
        parser.add_argument('--batch_dir', type=str, required=True,
                            help='输入数据目录，将处理目录中的所有.npz和.csv文件')
        parser.add_argument('--output_dir', type=str, default=None,
                            help='输出结果目录')
        parser.add_argument('--elements', nargs='+', default=None,
                            help='要预测的元素列表，默认为所有配置的元素')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='批次大小')
        parser.add_argument('--format', type=str, choices=['csv', 'npz'], default='csv',
                            help='输出格式')
        
        args = parser.parse_args()
        
        batch_predict_directory(
            args.batch_dir,
            args.output_dir,
            args.elements,
            args.batch_size,
            args.format
        )
    else:
        main() 