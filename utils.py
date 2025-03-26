#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具模块：提供缓存和进度管理功能
"""

import os
import json
import pickle
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
import psutil

logger = logging.getLogger('utils')

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir='cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_info_file = os.path.join(cache_dir, 'cache_info.json')
        self.cache_info = self._load_cache_info()
        self.checkpoint_interval = 1000  # 每处理1000个样本保存一次检查点
        self.validation_metrics = {
            'min_data_size': 100,
            'max_memory_usage': 0.8,
            'required_fields': ['data', 'metadata', 'validation_metrics'],
            'data_quality_thresholds': {
                'snr': 3.0,  # 最小信噪比
                'wavelength_coverage': 0.9,  # 最小波长覆盖率
                'normalization_quality': 0.8  # 最小归一化质量
            }
        }
    
    def _load_cache_info(self):
        """加载缓存信息"""
        if os.path.exists(self.cache_info_file):
            try:
                with open(self.cache_info_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache_info(self):
        """保存缓存信息"""
        with open(self.cache_info_file, 'w') as f:
            json.dump(self.cache_info, f, indent=4)
    
    def get_cache(self, key):
        """获取缓存数据"""
        if key in self.cache_info:
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    # 验证缓存数据的完整性
                    if self._validate_cache_data(data):
                        return data
                    else:
                        logger.warning(f"缓存数据 {key} 验证失败，将重新计算")
                        return None
                except:
                    logger.warning(f"读取缓存 {key} 失败")
                    return None
        return None
    
    def set_cache(self, key, data, is_checkpoint=False):
        """设置缓存数据"""
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            # 更新缓存信息
            self.cache_info[key] = {
                'timestamp': datetime.now().isoformat(),
                'size': os.path.getsize(cache_file),
                'is_checkpoint': is_checkpoint,
                'validation_status': self._validate_cache_data(data)
            }
            self._save_cache_info()
            return True
        except Exception as e:
            logger.error(f"保存缓存 {key} 失败: {str(e)}")
            return False
    
    def _validate_cache_data(self, data):
        """验证缓存数据的完整性"""
        try:
            if not isinstance(data, dict):
                logger.warning("缓存数据格式错误：不是字典类型")
                return False
                
            # 检查必要字段
            required_fields = self.validation_metrics['required_fields']
            if not all(field in data for field in required_fields):
                missing_fields = [field for field in required_fields if field not in data]
                logger.warning(f"缓存数据缺少必要字段：{missing_fields}")
                return False
            
            # 验证数据维度
            if not isinstance(data['data'], np.ndarray):
                logger.warning("光谱数据格式错误：不是numpy数组")
                return False
                
            if data['data'].size == 0:
                logger.warning("光谱数据为空")
                return False
            
            # 验证元数据
            metadata = data['metadata']
            if not isinstance(metadata, dict):
                logger.warning("元数据格式错误：不是字典类型")
                return False
                
            # 验证评估指标
            metrics = data['validation_metrics']
            if not isinstance(metrics, dict):
                logger.warning("评估指标格式错误：不是字典类型")
                return False
                
            # 检查数据质量指标
            if 'quality_metrics' in metrics:
                quality = metrics['quality_metrics']
                if quality['snr'] < self.validation_metrics['data_quality_thresholds']['snr']:
                    logger.warning(f"信噪比过低：{quality['snr']:.2f}")
                    return False
                if quality['wavelength_coverage'] < self.validation_metrics['data_quality_thresholds']['wavelength_coverage']:
                    logger.warning(f"波长覆盖率不足：{quality['wavelength_coverage']:.2f}")
                    return False
                if quality['normalization_quality'] < self.validation_metrics['data_quality_thresholds']['normalization_quality']:
                    logger.warning(f"归一化质量不足：{quality['normalization_quality']:.2f}")
                    return False
            
            # 检查内存使用
            data_size = data['data'].nbytes / (1024 * 1024)  # 转换为MB
            if data_size > self.validation_metrics['max_memory_usage'] * psutil.virtual_memory().total / (1024 * 1024 * 1024):
                logger.warning(f"数据大小超过内存限制：{data_size:.2f}MB")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"数据验证过程出错：{str(e)}")
            return False
    
    def validate_and_report(self, data):
        """验证数据并生成详细报告"""
        validation_report = {
            'is_valid': False,
            'issues': [],
            'metrics': {},
            'recommendations': []
        }
        
        try:
            # 基本格式验证
            if not isinstance(data, dict):
                validation_report['issues'].append("数据格式错误：不是字典类型")
                return validation_report
            
            # 字段检查
            for field in self.validation_metrics['required_fields']:
                if field not in data:
                    validation_report['issues'].append(f"缺少必要字段：{field}")
            
            # 数据质量检查
            if 'validation_metrics' in data and 'quality_metrics' in data['validation_metrics']:
                quality = data['validation_metrics']['quality_metrics']
                validation_report['metrics'] = quality
                
                # 检查各项指标
                if quality['snr'] < self.validation_metrics['data_quality_thresholds']['snr']:
                    validation_report['issues'].append(f"信噪比过低：{quality['snr']:.2f}")
                    validation_report['recommendations'].append("考虑增加曝光时间或使用更好的观测条件")
                
                if quality['wavelength_coverage'] < self.validation_metrics['data_quality_thresholds']['wavelength_coverage']:
                    validation_report['issues'].append(f"波长覆盖率不足：{quality['wavelength_coverage']:.2f}")
                    validation_report['recommendations'].append("检查光谱仪器的波长范围设置")
                
                if quality['normalization_quality'] < self.validation_metrics['data_quality_thresholds']['normalization_quality']:
                    validation_report['issues'].append(f"归一化质量不足：{quality['normalization_quality']:.2f}")
                    validation_report['recommendations'].append("尝试调整归一化参数或使用不同的归一化方法")
            
            # 内存使用检查
            if isinstance(data.get('data'), np.ndarray):
                data_size = data['data'].nbytes / (1024 * 1024)
                if data_size > self.validation_metrics['max_memory_usage'] * psutil.virtual_memory().total / (1024 * 1024 * 1024):
                    validation_report['issues'].append(f"数据大小超过内存限制：{data_size:.2f}MB")
                    validation_report['recommendations'].append("考虑使用数据分块处理或降低数据精度")
            
            validation_report['is_valid'] = len(validation_report['issues']) == 0
            return validation_report
            
        except Exception as e:
            validation_report['issues'].append(f"验证过程出错：{str(e)}")
            return validation_report
    
    def save_checkpoint(self, key, data, current_progress):
        """保存检查点"""
        if current_progress % self.checkpoint_interval == 0:
            checkpoint_key = f"{key}_checkpoint_{current_progress}"
            return self.set_cache(checkpoint_key, data, is_checkpoint=True)
        return False
    
    def get_latest_checkpoint(self, key):
        """获取最新的检查点"""
        checkpoints = [k for k in self.cache_info.keys() 
                      if k.startswith(f"{key}_checkpoint_") 
                      and self.cache_info[k].get('is_checkpoint', False)]
        if checkpoints:
            latest_checkpoint = max(checkpoints, 
                                  key=lambda x: int(x.split('_')[-1]))
            return self.get_cache(latest_checkpoint)
        return None
    
    def clear_cache(self, key=None):
        """清除缓存"""
        if key is None:
            # 清除所有缓存
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, file))
            self.cache_info = {}
            self._save_cache_info()
        else:
            # 清除特定缓存
            if key in self.cache_info:
                cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                del self.cache_info[key]
                self._save_cache_info()
    
    def get_cache_size(self):
        """获取缓存总大小（字节）"""
        total_size = 0
        for info in self.cache_info.values():
            total_size += info['size']
        return total_size
    
    def list_caches(self):
        """列出所有缓存"""
        return self.cache_info

def ask_clear_cache(cache_manager):
    """询问是否清除缓存"""
    if cache_manager.cache_info:
        print("\n发现现有缓存:")
        for key, info in cache_manager.cache_info.items():
            print(f"- {key}: {datetime.fromisoformat(info['timestamp'])}")
        print(f"\n总缓存大小: {cache_manager.get_cache_size() / 1024 / 1024:.2f} MB")
        
        while True:
            response = input("\n是否清除缓存？(y/n): ").lower()
            if response in ['y', 'n']:
                if response == 'y':
                    cache_manager.clear_cache()
                    print("已清除所有缓存")
                return response == 'y'
            print("请输入 y 或 n")

class ProgressManager:
    """进度管理器"""
    
    def __init__(self, total, desc=None):
        self.total = total
        self.desc = desc or "处理进度"
        self.pbar = None
    
    def __enter__(self):
        self.pbar = tqdm(total=self.total, desc=self.desc)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar:
            self.pbar.close()
    
    def update(self, n=1):
        """更新进度"""
        if self.pbar:
            self.pbar.update(n)
    
    def set_description(self, desc):
        """设置描述"""
        if self.pbar:
            self.pbar.set_description(desc) 