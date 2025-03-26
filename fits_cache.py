#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FITS文件缓存模块：提供FITS文件的读取、缓存和管理功能
"""

import os
import json
import pickle
import logging
import numpy as np
import time
from astropy.io import fits
from tqdm import tqdm
import glob

logger = logging.getLogger('fits_cache')

class FITSCache:
    """FITS文件缓存管理器"""
    
    def __init__(self, cache_dir='processed_data/fits_cache', batch_size=100):
        """
        初始化FITS缓存管理器
        
        参数:
            cache_dir (str): 缓存目录路径
            batch_size (int): 批处理大小，用于分批处理和验证数据
        """
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.cache_index_file = os.path.join(cache_dir, 'cache_index.json')
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 加载缓存索引
        self.cache_index = self._load_cache_index()
        
        # 文件扩展名缓存
        self.extension_cache = {}
        
        # 支持的FITS文件扩展名
        self.supported_extensions = ['.fits', '.fits.gz', '.fit', '.fit.gz']
        
    def _load_cache_index(self):
        """加载缓存索引"""
        if os.path.exists(self.cache_index_file):
            try:
                with open(self.cache_index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载缓存索引失败: {e}")
                return {}
        return {}
    
    def _save_cache_index(self):
        """保存缓存索引"""
        try:
            with open(self.cache_index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=4)
        except Exception as e:
            logger.error(f"保存缓存索引失败: {e}")
    
    def get_fits_data(self, obsid, validate=True):
        """
        获取FITS数据，如果缓存中没有则返回None
        
        参数:
            obsid (str): 观测ID
            validate (bool): 是否验证数据完整性
            
        返回:
            dict: 包含FITS数据的字典，如果不存在则返回None
        """
        cache_file = os.path.join(self.cache_dir, f'{obsid}.npz')
        
        if obsid in self.cache_index and os.path.exists(cache_file):
            try:
                data = np.load(cache_file, allow_pickle=True)
                data_dict = {key: data[key] for key in data.files}
                
                if validate and not self._validate_data(data_dict):
                    logger.warning(f"缓存数据验证失败: {obsid}")
                    return None
                    
                return data_dict
            except Exception as e:
                logger.error(f"读取缓存数据失败 {obsid}: {e}")
                return None
        return None
    
    def cache_fits_data(self, obsid, data_dict, validate=True):
        """
        缓存FITS数据
        
        参数:
            obsid (str): 观测ID
            data_dict (dict): 要缓存的数据字典
            validate (bool): 是否在缓存前验证数据
            
        返回:
            bool: 是否成功缓存
        """
        if validate and not self._validate_data(data_dict):
            logger.error(f"数据验证失败，不进行缓存: {obsid}")
            return False
            
        try:
            cache_file = os.path.join(self.cache_dir, f'{obsid}.npz')
            np.savez_compressed(cache_file, **data_dict)
            
            self.cache_index[obsid] = {
                'file': cache_file,
                'timestamp': time.time(),
                'keys': list(data_dict.keys())
            }
            self._save_cache_index()
            return True
        except Exception as e:
            logger.error(f"缓存数据失败 {obsid}: {e}")
            return False
    
    def _validate_data(self, data_dict):
        """
        验证数据完整性
        
        参数:
            data_dict (dict): 要验证的数据字典
            
        返回:
            bool: 数据是否有效
        """
        required_keys = {'wavelength', 'flux', 'error', 'mask'}
        
        # 检查必需的键
        if not all(key in data_dict for key in required_keys):
            missing_keys = required_keys - set(data_dict.keys())
            logger.warning(f"数据缺少必需的键: {missing_keys}")
            return False
        
        # 检查数组维度和大小
        try:
            wavelength = data_dict['wavelength']
            flux = data_dict['flux']
            error = data_dict['error']
            mask = data_dict['mask']
            
            if not all(isinstance(arr, np.ndarray) for arr in [wavelength, flux, error, mask]):
                logger.warning("数据类型错误：不是NumPy数组")
                return False
            
            if not all(len(arr) > 0 for arr in [wavelength, flux, error, mask]):
                logger.warning("数据长度为0")
                return False
            
            if not all(len(arr) == len(wavelength) for arr in [flux, error, mask]):
                logger.warning("数组长度不一致")
                return False
            
            # 检查数值有效性
            if np.isnan(wavelength).all() or np.isnan(flux).all():
                logger.warning("数据全为NaN")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"数据验证过程出错: {e}")
            return False
    
    def process_fits_file(self, fits_file, obsid=None):
        """
        处理FITS文件并返回数据
        
        参数:
            fits_file (str): FITS文件路径
            obsid (str): 观测ID，如果为None则从文件名推断
            
        返回:
            dict: 处理后的数据字典
        """
        if obsid is None:
            obsid = self._extract_obsid_from_file(fits_file)
        
        # 检查缓存
        cached_data = self.get_fits_data(obsid)
        if cached_data is not None:
            return cached_data
        
        try:
            with fits.open(fits_file, ignore_missing_end=True, memmap=False) as hdul:
                # 读取所有可能需要的信息
                data_dict = {
                    'wavelength': hdul[0].data,  # 波长信息
                    'flux': hdul[1].data,        # 光谱流量
                    'error': hdul[2].data,       # 误差信息
                    'mask': hdul[3].data,        # 掩码信息
                    'header_0': dict(hdul[0].header),  # 头文件信息
                    'header_1': dict(hdul[1].header),
                    'header_2': dict(hdul[2].header),
                    'header_3': dict(hdul[3].header)
                }
                
                # 缓存数据
                self.cache_fits_data(obsid, data_dict)
                
                return data_dict
                
        except Exception as e:
            logger.error(f"处理FITS文件失败 {fits_file}: {e}")
            return None
    
    def _extract_obsid_from_file(self, file_path):
        """从文件名中提取观测ID"""
        base_name = os.path.basename(file_path)
        # 移除扩展名
        for ext in self.supported_extensions:
            if base_name.lower().endswith(ext):
                base_name = base_name[:-len(ext)]
                break
        return base_name
    
    def clean_cache(self, force=False):
        """
        清理缓存文件
        
        参数:
            force (bool): 是否强制清理而不询问
        """
        if os.path.exists(self.cache_dir):
            cache_files = glob.glob(os.path.join(self.cache_dir, "*"))
            if cache_files:
                logger.info(f"发现{len(cache_files)}个缓存文件")
                if force or input("是否清理缓存? (y/n): ").lower() == 'y':
                    for file in tqdm(cache_files, desc="清理缓存"):
                        try:
                            os.remove(file)
                        except Exception as e:
                            logger.error(f"删除文件失败 {file}: {e}")
                    self.cache_index = {}
                    self._save_cache_index()
                    logger.info("缓存清理完成")
    
    def get_cache_info(self):
        """获取缓存信息"""
        total_size = 0
        file_count = 0
        
        for obsid, info in self.cache_index.items():
            file_path = info['file']
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
                file_count += 1
        
        return {
            'total_files': file_count,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': self.cache_dir
        }
    
    def validate_all_cache(self, batch_size=None):
        """
        验证所有缓存数据的完整性
        
        参数:
            batch_size (int): 批处理大小，如果为None则使用默认值
            
        返回:
            tuple: (有效文件数, 总文件数)
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        valid_count = 0
        total_count = len(self.cache_index)
        
        for i, (obsid, info) in enumerate(self.cache_index.items()):
            if i % batch_size == 0:
                logger.info(f"正在验证第 {i+1}/{total_count} 批数据...")
            
            data = self.get_fits_data(obsid, validate=True)
            if data is not None:
                valid_count += 1
        
        logger.info(f"缓存验证完成: {valid_count}/{total_count} 个文件有效")
        return valid_count, total_count 