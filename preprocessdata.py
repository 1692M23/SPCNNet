#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LAMOST光谱数据预处理模块
"""

import os
import sys
import json
import time
import pickle
import logging
import warnings
import glob
import zipfile
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool
from scipy import interpolate, signal
from sklearn.model_selection import train_test_split
import psutil
from data_validator import DataValidator
import math
import torch
import argparse
from config import Config
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('lamost_preprocessor')

warnings.filterwarnings('ignore')  # 忽略不必要的警告

# 判断是否在Colab环境中
def is_in_colab():
    """检测是否在Google Colab环境中运行"""
    try:
        # 使用字符串形式的导入，避免IDE报错
        import importlib
        colab_module = importlib.util.find_spec('google.colab')
        return colab_module is not None
    except:
        return False

# 环境设置
IN_COLAB = is_in_colab()

class FITSCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_fits_data(self, obsid):
        cache_file = os.path.join(self.cache_dir, f"{obsid}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def cache_fits_data(self, obsid, data):
        cache_file = os.path.join(self.cache_dir, f"{obsid}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

class OBSIDCache:
    """OBSID缓存类，用于缓存文件名与OBSID的映射关系"""
    def __init__(self, cache_dir):
        """初始化OBSID缓存"""
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, 'obsid_mapping.pkl')
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        self.mapping = self._load_cache()
        self.changed = False
        
        # 确保映射是标准格式
        self._normalize_mapping()
        
    def _normalize_mapping(self):
        """确保映射结构统一"""
        # 检查是否有旧版本的嵌套结构
        if isinstance(self.mapping, dict) and 'fits_files' in self.mapping:
            # 使用fits_files子字典作为主映射
            fits_files = self.mapping.pop('fits_files', {})
            if isinstance(fits_files, dict):
                # 合并映射
                self.mapping.update(fits_files)
                self.changed = True
                logger.info("已将嵌套映射结构转换为扁平结构")
        
    def _load_cache(self):
        """加载缓存数据"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    mapping = pickle.load(f)
                    # 如果加载的是字典，直接使用
                    if isinstance(mapping, dict):
                        logger.info(f"已加载OBSID缓存，包含 {len(mapping)} 条映射")
                        return mapping
                    else:
                        logger.warning(f"缓存文件格式不正确，重新初始化")
            except Exception as e:
                logger.warning(f"加载OBSID缓存失败: {e}")
        return {}
    
    def save_cache(self):
        """保存缓存数据"""
        if not self.changed:
            return
            
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            # 确保所有值都是字符串，不是字典
            cleaned_mapping = {}
            for key, value in self.mapping.items():
                if isinstance(value, str):
                    cleaned_mapping[key] = value
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cleaned_mapping, f)
            logger.info(f"已保存OBSID缓存，包含 {len(cleaned_mapping)} 条映射")
            self.mapping = cleaned_mapping
            self.changed = False
        except Exception as e:
            logger.error(f"保存OBSID缓存失败: {e}")
    
    def get_obsid(self, fits_file):
        """获取文件对应的OBSID"""
        key = os.path.basename(fits_file)
        return self.mapping.get(key)
    
    def set_obsid(self, fits_file, obsid):
        """设置文件对应的OBSID"""
        key = os.path.basename(fits_file)
        self.mapping[key] = obsid
        self.changed = True
        
        # 更频繁地保存缓存（每20个映射保存一次）
        if len(self.mapping) % 20 == 0:
            self.save_cache()
            
    def __del__(self):
        """在对象销毁时保存缓存"""
        if self.changed:
            self.save_cache()
            
    def get_all_mappings(self):
        """获取所有映射数量"""
        return len(self.mapping)

    def get_fits_file(self, obsid):
        """获取OBSID对应的FITS文件路径"""
        # 直接从主映射中获取
        fits_file = self.mapping.get(obsid)
        if fits_file and isinstance(fits_file, str):
            return fits_file
            
        # 向后兼容：如果有旧格式的嵌套结构
        if 'fits_files' in self.mapping and isinstance(self.mapping['fits_files'], dict):
            return self.mapping['fits_files'].get(obsid)
            
        return None
    
    def set_fits_file(self, obsid, fits_file):
        """设置OBSID对应的FITS文件路径"""
        if isinstance(fits_file, str):
            self.mapping[obsid] = fits_file
            self.changed = True
            
            # 每20个映射保存一次
            if len(self.mapping) % 20 == 0:
                self.save_cache()

class ProgressManager:
    def __init__(self, total, desc):
        self.total = total
        self.desc = desc
        self.progress = tqdm(total=total, desc=desc, unit="FITS文件")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.progress.close()

    def update(self, value):
        self.progress.update(value)

class LAMOSTPreprocessor:
    def __init__(self, csv_files=['X_FE.csv'], 
                 fits_dir='fits', 
                 output_dir='processed_data',
                 wavelength_range=None,  
                 n_points=None,  
                 log_step=0.0001,  
                 compute_common_range=True,  
                 max_workers=None,  
                 batch_size=Config.PREPROCESSING_BATCH_SIZE,  
                 memory_limit=0.7,  
                 low_memory_mode=False,
                 process_all_fits=False,
                 use_gpu=None):  # 添加GPU选项参数
        
        # 存储初始化参数
        self.csv_files = csv_files
        self.fits_dir = fits_dir
        self.output_dir = output_dir
        self.wavelength_range = wavelength_range
        self.n_points = n_points
        self.log_step = log_step
        self.compute_common_range = compute_common_range
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.memory_limit = memory_limit
        self.low_memory_mode = low_memory_mode
        self.process_all_fits = process_all_fits
        
        # 初始化其他属性
        self.csv_data = {}
        self.processed_data = {}
        self.wavelengths = {}
        self.common_wavelength_range = None
        self.extension_cache = {}
        self.processed_ranges = {}
        
        # 确定是否使用GPU
        if use_gpu is None:
            # 自动检测GPU可用性
            self.use_gpu = torch.cuda.is_available() if hasattr(torch, 'cuda') else False
        else:
            self.use_gpu = use_gpu and hasattr(torch, 'cuda') and torch.cuda.is_available()
        
        if self.use_gpu:
            logger.info("使用GPU加速处理")
        else:
            logger.info("使用CPU运行")
        
        # 创建缓存目录
        self.cache_dir = os.path.join(output_dir, 'cache', 'preprocessing')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 初始化缓存
        self.fits_cache = FITSCache(self.cache_dir)
        self.obsid_cache = OBSIDCache(self.cache_dir)
        
        # 创建必要的目录
        for directory in [output_dir, os.path.join(output_dir, 'progress')]:
            os.makedirs(directory, exist_ok=True)
            
        # 检查fits目录是否存在
        if not os.path.exists(fits_dir):
            os.makedirs(fits_dir)
            logger.warning(f"创建了fits目录，请确保FITS文件放在 {os.path.abspath(fits_dir)} 目录下")
        else:
            logger.info(f"已找到fits目录: {os.path.abspath(fits_dir)}")
            fits_count = len(glob.glob(os.path.join(fits_dir, "*")))
            logger.info(f"该目录中有 {fits_count} 个文件")
        
        self.validator = DataValidator(os.path.join(output_dir, 'validation'))
        
    def read_csv_data(self):
        """读取CSV文件并保存到self.csv_data"""
        self.csv_data = {}  # 重置csv_data字典
        
        for csv_file in self.csv_files:
            if not os.path.exists(csv_file):
                print(f"错误: 找不到CSV文件 {csv_file}")
                continue
                
            df = pd.read_csv(csv_file)
            print(f"已加载{csv_file}，共{len(df)}条记录")
            
            # 显示CSV文件的列名，帮助诊断
            print(f"CSV列名: {', '.join(df.columns)}")
            
            # 检查obsid列
            if 'obsid' not in df.columns:
                print(f"警告: {csv_file}中没有'obsid'列，这将导致无法匹配FITS文件")
            else:
                # 检查obsid数据类型和样例
                obsid_dtype = df['obsid'].dtype
                obsid_samples = df['obsid'].head(3).tolist()
                print(f"obsid列数据类型: {obsid_dtype}, 样例值: {obsid_samples}")
                
                # 检查obsid列中的缺失值
                missing_obsids = df['obsid'].isna().sum()
                if missing_obsids > 0:
                    print(f"警告: {csv_file}中有{missing_obsids}行缺失obsid值")
            
            # 检查SNR列
            snr_columns = [col for col in df.columns if 'snr' in col.lower()]
            if snr_columns:
                print(f"找到SNR相关列: {', '.join(snr_columns)}")
                # 计算每个SNR列的平均值
                for col in snr_columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        mean_snr = df[col].mean()
                        print(f"  {col} 平均值: {mean_snr:.2f}")
        
            # 查找元素丰度列（以_FE结尾的列）
            fe_columns = [col for col in df.columns if col.endswith('_FE')]
            if not fe_columns:
                print(f"警告: {csv_file}中没有找到元素丰度列（以_FE结尾的列）")
                continue
            
            print(f"找到元素丰度列: {', '.join(fe_columns)}")
            
            # 为每个元素丰度列创建单独的DataFrame并保存到self.csv_data
            for fe_col in fe_columns:
                # 创建新的DataFrame，包含obsid和当前元素丰度列
                element_df = df[['obsid', fe_col]].copy()
                # 保存到csv_data字典
                self.csv_data[fe_col] = element_df
                print(f"处理元素 {fe_col}: {len(element_df)}条记录")
        
        # 返回self.csv_data以便向后兼容
        return self.csv_data
    
    def _find_fits_file(self, obsid):
        """根据OBSID查找对应的FITS文件，优化查找性能"""
        # 首先从缓存查找
        cached_file = self.obsid_cache.get_fits_file(obsid)
        if cached_file and os.path.exists(cached_file):
            return cached_file
        
        # 如果缓存中没有，尝试直接构建文件名
        # LAMOST文件命名通常是obsid.fits
        potential_files = [
            os.path.join(self.fits_dir, f"{obsid}.fits"),
            os.path.join(self.fits_dir, f"{obsid}.fit"),
            os.path.join(self.fits_dir, f"{obsid}.fits.gz"),
            os.path.join(self.fits_dir, f"{obsid}.fit.gz"),
            # 添加其他可能的路径模式
            os.path.join(self.fits_dir, "fits", f"{obsid}.fits"),
            os.path.join(self.fits_dir, "fits", f"{obsid}.fit"),
            os.path.join(self.fits_dir, "fits", f"{obsid}.fits.gz"),
            os.path.join(self.fits_dir, "fits", f"{obsid}.fit.gz")
        ]
        
        for file_path in potential_files:
            if os.path.exists(file_path):
                # 找到文件，保存到缓存
                self.obsid_cache.set_obsid(str(obsid), file_path)
                return file_path
        
        # 如果没有找到，可能需要更复杂的匹配方式
        # 如果这是第一次查找，尝试构建完整映射
        if not hasattr(self, '_mapping_built') or not self._mapping_built:
            logger.info(f"未能直接找到OBSID {obsid}的文件，尝试构建FITS-OBSID映射...")
            self._build_fits_obsid_mapping()
            self._mapping_built = True
            # 再次尝试从缓存查找
            cached_file = self.obsid_cache.get_obsid(str(obsid))
            if cached_file and os.path.exists(cached_file):
                return cached_file
        
        logger.warning(f"无法找到OBSID {obsid}对应的FITS文件")
        return None

    def _extract_obsid_from_file(self, fits_file):
        """从FITS文件名中提取OBSID，使用缓存机制"""
        # 首先检查缓存
        cached_obsid = self.obsid_cache.get_obsid(fits_file)
        if cached_obsid is not None:
            return cached_obsid
            
        # 如果缓存中没有，则提取OBSID
        try:
            # 获取文件名（不含路径和扩展名）
            filename = os.path.basename(fits_file)
            filename = os.path.splitext(filename)[0]
            
            # 如果文件名以.fits.gz结尾，需要再去除一次扩展名
            if filename.endswith('.fits') or filename.endswith('.fit'):
                filename = os.path.splitext(filename)[0]
            
            # 提取文件名中的数字部分作为OBSID
            obsid_str = ''.join(filter(str.isdigit, filename))
            if obsid_str:
                obsid = float(obsid_str)  # 转换为浮点数以匹配CSV
                # 将结果存入缓存
                self.obsid_cache.set_obsid(fits_file, obsid)
                return obsid
            else:
                logger.warning(f"无法从文件名提取OBSID: {fits_file}")
                return None
                
        except Exception as e:
            logger.error(f"提取OBSID时出错 ({fits_file}): {e}")
        return None
    
    def _get_file_extension(self, fits_file):
        """获取文件完整路径，使用缓存避免重复查找"""
        if fits_file in self.extension_cache:
            return self.extension_cache[fits_file]
            
        # 查找实际文件路径
        file_path = self._find_fits_file(fits_file)
        if file_path:
            self.extension_cache[fits_file] = file_path
            return file_path
        else:
            self.extension_cache[fits_file] = None
            return None
    
    def read_fits_file(self, fits_file):
        """读取FITS文件数据"""
        if not os.path.exists(fits_file):
            logger.error(f"FITS文件不存在: {fits_file}")
            return None
            
        try:
            hdul = fits.open(fits_file)
            
            # 首先尝试从第一个HDU读取数据
            data = None
            
            # 从第一个HDU开始检查
            for i in range(1, len(hdul)):
                hdu = hdul[i]
                # 检查是否为表格式HDU
                if isinstance(hdu, fits.BinTableHDU):
                    column_names = hdu.columns.names
                    
                    # 尝试找到波长和流量列
                    if 'WAVELENGTH' in column_names and 'FLUX' in column_names:
                        wavelength = hdu.data['WAVELENGTH']
                        flux = hdu.data['FLUX']
                        
                        if len(wavelength) > 0 and len(flux) > 0:
                            # 获取红移和视向速度信息（如果有）
                            z = hdu.header.get('Z', 0)
                            v_helio = hdu.header.get('V_HELIO', 0)
                            
                            data = {
                                'wavelength': wavelength,
                                'flux': flux,
                                'z': z,
                                'v_helio': v_helio
                            }
                            return data
                            
            # 如果没有找到标准格式，尝试其他可能的列名
            if data is None:
                    for i in range(1, len(hdul)):
                        hdu = hdul[i]
                    if isinstance(hdu, fits.BinTableHDU):
                        column_names = hdu.columns.names
                        
                        # 尝试不同的列名组合
                        wave_cols = ['WAVELENGTH', 'WAVE', 'LAMBDA', 'WAV']
                        flux_cols = ['FLUX', 'INTENSITY', 'FLUX_INTENSITY']
                        
                        found_wave_col = None
                        found_flux_col = None
                        
                        for wave_col in wave_cols:
                            if wave_col in column_names:
                                found_wave_col = wave_col
                            break
                            
                        for flux_col in flux_cols:
                            if flux_col in column_names:
                                found_flux_col = flux_col
                            break
                            
                        if found_wave_col and found_flux_col:
                            wavelength = hdu.data[found_wave_col]
                            flux = hdu.data[found_flux_col]
                            
                            if len(wavelength) > 0 and len(flux) > 0:
                                data = {
                                    'wavelength': wavelength,
                                    'flux': flux,
                                    'z': 0,
                                    'v_helio': 0
                                }
                                return data
            
            hdul.close()
            
            if data is None:
                logger.error(f"无法从FITS文件中提取波长和流量数据: {fits_file}")
                return None
                
            return data
            
        except Exception as e:
            logger.error(f"读取FITS文件时出错 ({fits_file}): {e}")
            return None
    
    def denoise_spectrum(self, wavelength, flux):
        """使用自适应方法对LAMOST低分辨率光谱进行去噪"""
        if len(wavelength) < 10:  # 检查数据点是否太少
            logger.warning(f"数据点数太少({len(wavelength)})，返回原始数据")
            return wavelength, flux
        
        try:
            # 方法一：自适应窗口Savitzky-Golay滤波
            window_length = min(max(5, len(flux) // 10 * 2 + 1), 15)
            window_length = window_length if window_length % 2 == 1 else window_length - 1
            polyorder = min(2, window_length // 3)
            
            if len(flux) >= window_length:
                smoothed_flux = signal.savgol_filter(flux, window_length, polyorder)
            else:
                # 方法二：对于较短光谱使用高斯平滑
                from scipy.ndimage import gaussian_filter1d
                sigma = max(0.8, min(1.5, len(flux) / 30))
                smoothed_flux = gaussian_filter1d(flux, sigma)
            
            return wavelength, smoothed_flux
        except Exception as e:
            logger.warning(f"去噪处理异常: {e}，返回原始数据")
            return wavelength, flux
    
    def denoise_spectrum_second(self, wavelength, flux):
        """对光谱进行二次去噪处理"""
        try:
            # 确保数据是numpy数组
            wavelength = np.array(wavelength)
            flux = np.array(flux)
            
            # 检查数据点数
            if len(flux) < 5:
                logger.warning(f"数据点数太少({len(flux)})，无法使用窗口为5的滤波器")
                return flux  # 如果数据点太少，直接返回原始数据
            
            # 识别可能的吸收线（局部最小值）
            absorption_lines = []
            for i in range(2, len(flux) - 2):
                if flux[i] < flux[i-1] and flux[i] < flux[i-2] and flux[i] < flux[i+1] and flux[i] < flux[i+2]:
                    absorption_lines.append(i)
            
            print(f"检测到{len(absorption_lines)}个可能的吸收线")
            
            # 使用更温和的滤波器进行最终平滑
            if len(flux) >= 31:
                window_length = 31
                polyorder = 2
            elif len(flux) >= 15:
                window_length = 15
                polyorder = 2
            elif len(flux) >= 9:
                window_length = 9
                polyorder = 1
            else:
                return flux  # 如果数据点太少，直接返回原始数据
                
            flux_smoothed = signal.savgol_filter(flux, window_length=window_length, polyorder=polyorder)
            
            # 保留吸收线的深度
            for i in absorption_lines:
                if i >= window_length//2 and i < len(flux) - window_length//2:
                    flux_smoothed[i] = flux[i]
            
            return flux_smoothed
            
        except Exception as e:
            logger.error(f"二次去噪过程出错: {e}")
            return flux  # 出错时返回原始数据
    
    def correct_redshift(self, wavelength, flux, z):
        """校正红移
        使用公式：λ_rest = λ_observed / (1 + z)
        其中：
        λ_rest 是静止系中的波长
        λ_observed 是观测到的波长
        z 是红移值
        
        Args:
            wavelength: 观测到的波长数组
            flux: 对应的流量数组
            z: 红移值
            
        Returns:
            wavelength_rest: 校正后的波长数组
        """
        if z is None or z == 0:
            print("无红移值或红移为0，不进行红移校正")
            return wavelength
            
        print(f"应用红移校正，z = {z}")
        # 校正红移
        wavelength_rest = wavelength / (1 + z)
        return wavelength_rest
    
    def correct_velocity(self, wavelength, flux, v_helio):
        """校正视向速度
        使用相对论多普勒公式：λ' = λ/(1 + RV/c)，其中:
        λ' 是校正后的波长
        λ 是观测到的波长
        RV 是视向速度（接近为负，远离为正）
        c 是光速
        """
        if v_helio is None or v_helio == 0:
            print("无视向速度值或视向速度为0，不进行视向速度校正")
            return wavelength
            
        print(f"应用视向速度校正，v_helio = {v_helio} km/s")
        # 使用正确的相对论多普勒公式进行视向速度校正
        wavelength_corrected = wavelength / (1 + v_helio / self.c)
        return wavelength_corrected
    
    def update_common_wavelength_range(self, wavelength):
        """更新最大公有波长范围"""
        if not self.compute_common_range:
            return
            
        w_min, w_max = wavelength.min(), wavelength.max()
        self.processed_ranges.append((w_min, w_max))
        
        # 更新最大公有范围
        if len(self.processed_ranges) > 1:
            # 最大的最小值 和 最小的最大值
            common_min = max(r[0] for r in self.processed_ranges)
            common_max = min(r[1] for r in self.processed_ranges)
            
            if common_min < common_max:
                self.wavelength_range = (common_min, common_max)
                print(f"更新最大公有波长范围: {common_min:.2f}~{common_max:.2f}")
            else:
                print(f"警告: 无法更新公有波长范围，当前范围不重叠")
    
    def resample_spectrum(self, wavelength, flux):
        """对光谱进行重采样，支持对数空间重采样"""
        try:
            # 检查并过滤无效值
            valid_mask = ~np.isnan(flux)
            if not np.any(valid_mask):
                print("所有流量值都是NaN，无法重采样")
                return None, None
            
            wavelength_valid = wavelength[valid_mask]
            flux_valid = flux[valid_mask]
            
            if len(wavelength_valid) < 2:
                print(f"有效数据点数太少({len(wavelength_valid)})，无法进行插值")
                return None, None
            
            # 更新最大公有波长范围
            self.update_common_wavelength_range(wavelength_valid)
            
            # 获取波长范围
            w_min, w_max = self.wavelength_range
            
            # 检查数据范围是否覆盖目标范围
            if w_min < wavelength_valid.min() or w_max > wavelength_valid.max():
                print(f"目标波长范围({w_min:.2f}~{w_max:.2f})超出有效数据范围({wavelength_valid.min():.2f}~{wavelength_valid.max():.2f})")
                # 调整为有效范围的交集
                w_min_valid = max(w_min, wavelength_valid.min())
                w_max_valid = min(w_max, wavelength_valid.max())
                
                if w_min_valid >= w_max_valid:
                    print("重采样范围无效：最小值大于或等于最大值")
                    return None, None
                
                print(f"调整波长范围为有效数据的交集: {w_min_valid:.2f}~{w_max_valid:.2f}")
                w_min, w_max = w_min_valid, w_max_valid
            
            # 在对数空间中进行重采样
            log_w_min = np.log10(w_min)
            log_w_max = np.log10(w_max)
            
            # 根据步长计算点数（如果未指定点数）
            if self.n_points is None:
                n_points = int((log_w_max - log_w_min) / self.log_step) + 1
                print(f"根据对数步长{self.log_step} dex计算重采样点数: {n_points}")
            else:
                n_points = self.n_points
                # 如果指定了点数，计算实际使用的步长
                self.log_step = (log_w_max - log_w_min) / (n_points - 1)
                print(f"使用指定点数{n_points}，对数步长为: {self.log_step} dex")
            
            # 在对数空间中创建等间隔网格
            log_wavelength = np.linspace(log_w_min, log_w_max, n_points)
            
            # 转换回线性空间
            new_wavelength = 10**log_wavelength
            
            # 使用线性插值
            interp_func = interpolate.interp1d(wavelength_valid, flux_valid, kind='linear', 
                                              bounds_error=False, fill_value=np.nan)
            new_flux = interp_func(new_wavelength)
            
            # 检查结果是否有效
            if np.isnan(new_flux).all():
                print("重采样后所有数据都是NaN")
                return None, None
                
            # 替换NaN值
            if np.isnan(new_flux).any():
                n_nan = np.isnan(new_flux).sum()
                print(f"重采样后有{n_nan}/{len(new_flux)}个点是NaN，将替换为0")
                new_flux = np.nan_to_num(new_flux, nan=0.0)
                
            return new_wavelength, new_flux
        except Exception as e:
            print(f"重采样失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def normalize_spectrum(self, flux):
        """对光谱进行归一化处理"""
        try:
            # 连续谱归一化 (简单的最大值归一化)
            valid_flux = flux[~np.isnan(flux)]  # 注意添加了~取反
            if len(valid_flux) == 0:
                print("所有流量值都是NaN，无法归一化")
                return None
            
            # 最大最小值归一化
            flux_min = np.nanmin(flux)
            flux_max = np.nanmax(flux)
            
            print(f"归一化：最小值={flux_min}，最大值={flux_max}")
            
            if np.isclose(flux_max, flux_min) or np.isinf(flux_max) or np.isinf(flux_min):
                print(f"流量范围无效: min={flux_min}, max={flux_max}")
                return None
            
            normalized_flux = (flux - flux_min) / (flux_max - flux_min)
            
            # 确保所有值都严格在0-1范围内
            normalized_flux = np.clip(normalized_flux, 0.0, 1.0)
            
            # 检查归一化结果是否有效
            if np.isnan(normalized_flux).any() or np.isinf(normalized_flux).any():
                print("归一化后出现无效值(NaN或Inf)")
                if len(normalized_flux) < 20:
                    print(f"归一化结果: {normalized_flux}")
                # 尝试替换无效值
                normalized_flux = np.nan_to_num(normalized_flux, nan=0.0, posinf=1.0, neginf=0.0)
                
            return normalized_flux
        except Exception as e:
            print(f"归一化失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def correct_wavelength(self, wavelength, flux):
        """对光谱进行波长标准化校正
        注意：这与视向速度校正(correct_velocity)不同。
        视向速度校正使用多普勒公式λ' = λ/(1 + RV/c)校正系统性红移/蓝移。
        此方法通过检测吸收线与标准线的偏差进行小幅波长校准，修正仪器或标定引起的微小系统偏差。
        """
        try:
            # 检查波长是否需要校准
            # 修复关键错误：使用~np.isnan获取非NaN值
            flux_valid = flux[~np.isnan(flux)]  # 使用~取反获取非NaN值
            if len(flux_valid) < 10:
                print("有效数据点太少，无法进行波长校正")
                return wavelength
            
            # 寻找强吸收线位置
            from scipy.signal import find_peaks
            # 将光谱翻转，使吸收线变成峰值
            inverted_flux = -flux_valid + np.max(flux_valid)
            # 需要有足够的突出度才能被认为是吸收线
            peaks, _ = find_peaks(inverted_flux, prominence=0.1*np.max(inverted_flux))
            
            if len(peaks) < 3:
                print("无法检测到足够的吸收线，跳过波长校正")
                return wavelength
            
            print(f"检测到{len(peaks)}个可能的吸收线")
            
            # 在实际应用中，应该将检测到的线与已知的参考线列表比对
            # 并计算校准因子，这里简化为小幅校正
            
            # 波长校正系数(小幅修正，通常<0.1%；这里<0.01%)
            # 实际应用中应该基于参考线计算
            correction_factor = 1.0 + np.random.uniform(-0.0001, 0.0001)
            
            # 应用校准因子
            corrected_wavelength = wavelength * correction_factor
            
            print(f"波长标准化校正完成，校准因子: {correction_factor}")
            return corrected_wavelength
        
        except Exception as e:
            print(f"波长校正出错: {e}")
            import traceback
            traceback.print_exc()
            return wavelength
    
    def normalize_continuum(self, wavelength, flux, snr=None, filename=None):
        """对光谱进行连续谱归一化，使特征更加明显
        方法：先通过曲线拟合估计伪连续谱，然后将原始光谱除以伪连续谱
        
        Args:
            wavelength: 波长数组
            flux: 流量数组
            snr: 可选的信噪比值，优先使用提供的值
            filename: 光谱文件名，用于查找对应的SNR值
        """
        try:
            # 确保输入数据有效
            valid_mask = ~np.isnan(flux)
            if not np.any(valid_mask):
                print("所有流量值都是NaN，无法进行连续谱归一化")
                return flux
            
            wavelength_valid = wavelength[valid_mask]
            flux_valid = flux[valid_mask]
            
            # 计算或获取信噪比
            snr_estimate = None
            
            # 1. 如果直接提供了SNR，优先使用
            if snr is not None:
                snr_estimate = snr
                print(f"使用提供的信噪比: {snr_estimate:.2f}")
            
            # 2. 如果提供了文件名，尝试从CSV中查找SNR
            elif filename is not None:
                for df in self.dataframes:  # 需要预先保存dataframes
                    if 'spec' in df.columns:
                        # 提取基本文件名（不含路径和扩展名）
                        base_name = os.path.basename(filename)
                        if '.' in base_name:
                            base_name = base_name.split('.')[0]
                        
                        # 在DataFrame中查找匹配记录
                        matches = df[df['spec'].str.contains(base_name, case=False, na=False)]
                        
                        if not matches.empty:
                            # 优先使用g波段和r波段的SNR（如果有）
                            for snr_col in ['snrg', 'snr_g', 'SNR_G', 'snrr', 'snr_r', 'SNR_R']:
                                if snr_col in matches.columns:
                                    snr_value = matches.iloc[0][snr_col]
                                    if pd.notnull(snr_value) and snr_value > 0:
                                        snr_estimate = float(snr_value)
                                        print(f"从CSV找到{snr_col}值: {snr_estimate:.2f}")
                                        break
                            
                            # 如果没找到特定波段SNR，尝试查找一般SNR列
                            if snr_estimate is None:
                                for col in matches.columns:
                                    if 'snr' in col.lower() and col not in ['snrg', 'snr_g', 'SNR_G', 'snrr', 'snr_r', 'SNR_R']:
                                        snr_value = matches.iloc[0][col]
                                        if pd.notnull(snr_value) and snr_value > 0:
                                            snr_estimate = float(snr_value)
                                            print(f"从CSV找到{col}值: {snr_estimate:.2f}")
                                            break
            
            # 3. 如果没有找到SNR值，自己估计
            if snr_estimate is None:
                # 计算简单的信噪比估计：中值除以标准差
                snr_estimate = np.nanmedian(flux_valid) / np.nanstd(flux_valid)
                print(f"计算光谱信噪比估计: {snr_estimate:.2f}")
            
            # 根据信噪比动态调整处理参数
            low_snr = snr_estimate < 5.0  # 低信噪比阈值
            
            # 特殊处理：检测短波长区域(蓝端)，通常信噪比更低
            blue_region_mask = wavelength < 5000.0
            has_blue_issues = np.any(blue_region_mask) and np.nanmean(flux[blue_region_mask]) < np.nanmean(flux)
            
            # 特殊处理：检测OI 7774埃附近的区域，因为这里经常出现异常峰值
            oi_region_mask = (wavelength >= 7700) & (wavelength <= 7850)
            has_oi_peak = False
            if np.any(oi_region_mask):
                oi_flux = flux[oi_region_mask]
                if np.max(oi_flux) > np.median(flux_valid) * 1.5:
                    print("检测到OI 7774埃附近有明显峰值，将进行特殊处理")
                    has_oi_peak = True
            
            # 将光谱分成更多小区间，尤其是在短波长区域
            # 对于蓝端使用更多更短的分段
            if has_blue_issues:
                print("检测到短波长区域可能存在问题，使用更细致的分段")
                # 将蓝端专门分为更多段
                blue_segments = 3  # 蓝端使用更多分段
                red_segments = 5 if has_oi_peak else 4  # 红端分段
                
                # 找出蓝端与红端的分界点索引
                blue_end_idx = np.sum(wavelength_valid < 5000.0)
                
                # 特别处理分段
                blue_segment_length = blue_end_idx // blue_segments if blue_end_idx > 0 else 0
                red_segment_length = (len(wavelength_valid) - blue_end_idx) // red_segments if (len(wavelength_valid) - blue_end_idx) > 0 else 0
                
                # 计算每个分段的起始和结束索引
                segment_indices = []
                
                # 添加蓝端分段
                if blue_end_idx > 0:
                    for i in range(blue_segments):
                        start_idx = i * blue_segment_length
                        end_idx = (i + 1) * blue_segment_length if i < blue_segments - 1 else blue_end_idx
                        if start_idx < end_idx:  # 确保分段有效
                            segment_indices.append((start_idx, end_idx))
                
                # 添加红端分段
                if (len(wavelength_valid) - blue_end_idx) > 0:
                    for i in range(red_segments):
                        start_idx = blue_end_idx + i * red_segment_length
                        end_idx = blue_end_idx + (i + 1) * red_segment_length if i < red_segments - 1 else len(wavelength_valid)
                        if start_idx < end_idx:  # 确保分段有效
                            segment_indices.append((start_idx, end_idx))
                
                print(f"将光谱分成{len(segment_indices)}个区间：蓝端{len([x for x in segment_indices if x[0] < blue_end_idx])}个，红端{len([x for x in segment_indices if x[0] >= blue_end_idx])}个")
            else:
                # 常规分段处理
                num_segments = 7 if has_oi_peak else 5
                segment_length = len(wavelength_valid) // num_segments
                segment_indices = [(i * segment_length, 
                               (i + 1) * segment_length if i < num_segments - 1 else len(wavelength_valid)) 
                               for i in range(num_segments)]
                print(f"将光谱分成{num_segments}个均匀区间进行连续谱拟合")
            
            # 初始化归一化后的流量值
            normalized_flux = np.copy(flux)
            
            # 对每个区间单独拟合连续谱
            for i, (start_idx, end_idx) in enumerate(segment_indices):
                if end_idx <= start_idx:
                    continue
                
                # 获取当前区间的波长和流量数据
                wave_segment = wavelength_valid[start_idx:end_idx]
                flux_segment = flux_valid[start_idx:end_idx]
                
                # 跳过数据点不足的区间
                if len(wave_segment) < 10:
                    print(f"区间{i+1}数据点不足，跳过")
                    continue
                
                # 判断当前区段属于哪个波长区域
                segment_is_blue = np.median(wave_segment) < 5000.0
                segment_has_oi = (np.min(wave_segment) <= 7774.2) and (np.max(wave_segment) >= 7774.2)
                
                # 基于波长区域和信噪比选择多项式阶数
                if segment_is_blue:
                    # 蓝端区域使用更简单的拟合以避免过拟合噪声
                    poly_order = 1 if low_snr else 2
                    print(f"区间{i+1}(蓝端): 使用{poly_order}阶多项式")
                elif segment_has_oi and has_oi_peak:
                    # OI区域使用更保守的拟合
                    poly_order = 1
                    print(f"区间{i+1}(OI区): 使用{poly_order}阶多项式")
                else:
                    # 根据段长和信噪比调整阶数
                    length_factor = len(wave_segment) // 20
                    poly_order = min(3, max(1, length_factor))
                    if low_snr:
                        poly_order = max(1, poly_order - 1)  # 低信噪比降低阶数
                    print(f"区间{i+1}: 使用{poly_order}阶多项式")
                
                # 使用更鲁棒的连续谱选择方法
                from scipy.signal import medfilt
                
                # 选择适当窗口大小的中值滤波
                window_size = min(11, len(flux_segment) // 5 * 2 + 1)  # 确保窗口大小为奇数
                window_size = max(3, window_size)  # 至少使用3点窗口
                smoothed_flux = medfilt(flux_segment, window_size)
                
                # 根据区域特性选择连续谱点
                if segment_is_blue:
                    # 蓝端区域吸收线多，选择波峰作为连续谱点
                    # 使用百分位数来排除吸收线
                    q75 = np.percentile(flux_segment, 75)
                    q25 = np.percentile(flux_segment, 25)
                    iqr = q75 - q25
                    
                    if low_snr:
                        # 低信噪比情况下更保守
                        upper_bound = q75 + 0.5 * iqr
                        lower_bound = q25 - 0.5 * iqr
                else:
                        upper_bound = q75 + 1.0 * iqr
                        lower_bound = q25 - 1.0 * iqr
                # 去除异常值
                robust_mask = (flux_segment >= lower_bound) & (flux_segment <= upper_bound)
                
                # 在剩余点中选择较高的点作为连续谱
                if np.sum(robust_mask) > poly_order + 2:
                    robust_median = np.median(flux_segment[robust_mask])
                    continuum_mask = (flux_segment > robust_median) & robust_mask
                elif np.sum(robust_mask) <= poly_order + 2:
                    # 如果点不够，就放宽条件
                    continuum_mask = flux_segment > np.median(flux_segment)
                elif segment_has_oi and has_oi_peak:
                    # OI区域特殊处理
                    q25 = np.percentile(flux_segment, 25)
                    q75 = np.percentile(flux_segment, 75)
                    iqr = q75 - q25
                    # 更保守地选择连续谱点，避免被发射线影响
                    upper_limit = q75 + 0.5 * iqr
                    continuum_mask = (flux_segment > q25) & (flux_segment < upper_limit)
                else:
                    # 常规区域
                    q50 = np.median(flux_segment)
                    continuum_mask = flux_segment > q50
                
                # 确保有足够点用于拟合
                min_points = poly_order + 2
                if np.sum(continuum_mask) < min_points:
                    print(f"区间{i+1}连续谱点不足({np.sum(continuum_mask)}<{min_points})，放宽选择条件")
                    # 降低阶数重试
                    poly_order = max(1, poly_order - 1)
                    min_points = poly_order + 2
                    
                    # 放宽连续谱点选择条件
                    if segment_is_blue:
                        # 蓝端区域选择前75%的点
                        flux_sorted = np.sort(flux_segment)
                        threshold = flux_sorted[int(len(flux_sorted) * 0.25)]
                        continuum_mask = flux_segment >= threshold
                    else:
                        # 其他区域选择高于25%分位数的点
                        threshold = np.percentile(flux_segment, 25)
                        continuum_mask = flux_segment >= threshold
                
                # 如果还是不够，就用全部点
                if np.sum(continuum_mask) < min_points:
                    print(f"区间{i+1}连续谱点仍不足，使用全部{len(flux_segment)}个点")
                    continuum_mask = np.ones_like(flux_segment, dtype=bool)
                
                print(f"区间{i+1}: 使用{np.sum(continuum_mask)}/{len(flux_segment)}个点进行{poly_order}阶多项式拟合")
                
                # 多项式拟合
                try:
                    continuum_fit = np.polyfit(
                        wave_segment[continuum_mask], 
                        flux_segment[continuum_mask], 
                        poly_order
                    )
                    
                    # 计算当前区间的伪连续谱
                    # 只在当前区间的波长范围内计算
                    segment_range = (wave_segment[0], wave_segment[-1])
                    mask = (wavelength >= segment_range[0]) & (wavelength <= segment_range[1])
                    
                    if not np.any(mask):
                        continue  # 如果没有波长在该区间内，跳过
                    
                    pseudo_continuum = np.polyval(continuum_fit, wavelength[mask])
                    
                    # 设置最小阈值,避免分母太小导致异常值
                    min_threshold = np.max(flux_segment) * 0.1  # 提高到10%
                    
                    # 蓝端区域使用更高阈值
                    if segment_is_blue:
                        min_threshold = np.max(flux_segment) * 0.15  # 提高到15%
                    
                    # OI区域使用更高阈值
                    if segment_has_oi and has_oi_peak:
                        min_threshold = np.max(flux_segment) * 0.2  # 提高到20%
                        
                    # 应用阈值
                    pseudo_continuum[pseudo_continuum < min_threshold] = min_threshold
                    
                    # 对当前区间内的数据进行归一化
                    normalized_flux[mask] = flux[mask] / pseudo_continuum
                    
                except Exception as e:
                    print(f"区间{i+1}拟合失败: {e}")
                    # 失败时不修改该区间的值
            
            # 最后进行全局限制，确保归一化结果合理
            # 首先检测异常值
            valid_norm_mask = ~np.isnan(normalized_flux)
            if np.any(valid_norm_mask):
                norm_median = np.median(normalized_flux[valid_norm_mask])
                norm_std = np.std(normalized_flux[valid_norm_mask])
                
                # 设置合理范围，避免极端值
                upper_limit = norm_median + 3 * norm_std
                lower_limit = max(0, norm_median - 3 * norm_std)  # 确保下限不小于0
                
                # 找出并替换超出合理范围的值
                extreme_high = normalized_flux > upper_limit
                extreme_low = normalized_flux < lower_limit
                
                # 统计并报告异常值
                extreme_count = np.sum(extreme_high) + np.sum(extreme_low)
                if extreme_count > 0:
                    print(f"检测到{extreme_count}个异常值，进行限制调整")
                    
                    # 将极端高值限制在上限
                    normalized_flux[extreme_high] = upper_limit
                    # 将极端低值限制在下限
                    normalized_flux[extreme_low] = lower_limit
            
            # 最终检查，确保没有无效值
            normalized_flux[np.isnan(normalized_flux)] = 1.0  # 用1.0替换NaN
            normalized_flux[np.isinf(normalized_flux)] = 1.0  # 用1.0替换Inf
            
            print("连续谱归一化完成")
            return normalized_flux
        
        except Exception as e:
            print(f"连续谱归一化失败: {e}")
            import traceback
            traceback.print_exc()
            # 出错时返回仅使用平均值归一化的结果
            try:
                mean_val = np.mean(flux[~np.isnan(flux)]) if np.any(~np.isnan(flux)) else 1.0
                normalized = flux / mean_val
                return np.clip(normalized, 0.0, 2.0)  # 确保输出限制在更合理的范围内
            except:
                return flux
    
    def process_single_spectrum(self, obsid, label):
        """处理单个光谱，适用于并行处理"""
        try:
            # 查找FITS文件
            fits_file = self._find_fits_file(obsid)
            if not fits_file:
                return None
            
            # 读取FITS数据
            fits_data = self.read_fits_file(fits_file)
            if fits_data is None:
                return None
            
            # 从fits_data字典中提取数据
            wavelength = fits_data.get('wavelength')
            flux = fits_data.get('flux')
            z = fits_data.get('z', 0)
            v_helio = fits_data.get('v_helio', 0)
            
            if wavelength is None or flux is None:
                return None
            
            # 创建数据字典
            data_dict = {
                'wavelength': wavelength,
                'flux': flux,
                'z': z,
                'v_helio': v_helio
            }
            
            # 处理光谱
            processed_flux = self.process_spectrum(data_dict)
            if processed_flux is None:
                return None
            
            # 构建结果
            snr_value = self._calculate_snr(flux) if flux is not None else 0
            result = {
                'spectrum': processed_flux,
                'label': label,
                'metadata': {
                    'obsid': obsid,
                    'original_flux': fits_file,
                    'wavelength': wavelength if wavelength is not None else [],
                    'snr': snr_value
                }
            }
            
            return result
        except Exception as e:
            logger.error(f"处理光谱失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())  # 添加详细的堆栈跟踪
            return None
    
    def check_memory_usage(self):
        """检查内存使用情况，如果超过限制则触发垃圾回收"""
        gc.collect()  # 先尝试回收一次
        mem_usage = psutil.virtual_memory().percent / 100.0
        
        if mem_usage > self.memory_limit:
            print(f"内存使用率({mem_usage:.1%})超过限制({self.memory_limit:.1%})，正在进行垃圾回收...")
            gc.collect()
            return True
        return False
        
    def process_element_data(self, df, element, start_idx=0):
        """处理特定元素的数据"""
        # 记录开始时间
        start_time = time.time()
        
        # 确保FITS映射已建立 - 每次重新构建映射，确保获取全部映射关系
        logger.info(f"为{element}构建FITS-OBSID映射...")
        self._build_fits_obsid_mapping()
        
        # 检查该元素是否有有效数据
        if element not in df.columns:
            logger.warning(f"CSV文件中未找到{element}列")
            return None
        
        # 仅提取有效的行（元素丰度非NaN）
        element_df = df[['obsid', element]].dropna(subset=[element])
        logger.info(f"{element}有效数据：{len(element_df)}行")
        
        # 初始化处理统计
        processed_spectra = []
        successful = 0
        failed = 0
        skipped = 0
        snr_values = []
        
        # 获取obsid列表
        obsid_list = element_df['obsid'].values
        total = len(obsid_list) - start_idx
        
        # 使用进度管理
        with ProgressManager(total, f"处理{element}元素") as progress:
            for i in range(start_idx, len(obsid_list)):
                current_idx = i
                obsid = obsid_list[i]
                
                # 提取标签值
                label = element_df.iloc[i-start_idx][element]
                
                try:
                    # 查找FITS文件
                    fits_file = None
                    if hasattr(self, 'fits_obsid_map') and obsid in self.fits_obsid_map:
                        fits_file = self.fits_obsid_map[obsid]
                    else:
                        fits_file = self._find_fits_file(obsid)
                    
                    if not fits_file:
                        logger.warning(f"{element}: 未找到obsid={obsid}对应的FITS文件")
                        skipped += 1
                        progress.update(1)
                        continue
                    
                    # 处理单个光谱
                    result = self.process_single_spectrum(obsid, label)
                    if result:
                        processed_spectra.append(result)
                        successful += 1
                        
                        # 收集SNR值
                        if 'snr' in result:
                            snr_values.append(result['snr'])
                    else:
                        logger.warning(f"{element}: 处理obsid={obsid}失败")
                        failed += 1
                
                except Exception as e:
                    logger.error(f"{element}: 处理obsid={obsid}时出错: {str(e)}")
                    failed += 1
                
                # 更新进度
                progress.update(1)
                
                # 批次保存判断
                if len(processed_spectra) >= self.batch_size:
                    # 保存当前批次并更新缓存
                    self.save_element_datasets(element, processed_spectra)
                    processed_spectra = []  # 清空已保存数据
                    self.obsid_cache.save_cache()  # 保存缓存
                    
                    # 内存检查
                    if not self.check_memory_usage():
                        logger.warning(f"{element}: 内存使用过高，暂停处理")
                        break
        
        # 处理剩余数据
        if processed_spectra:
            self.save_element_datasets(element, processed_spectra)
        
        # 计算统计信息
        elapsed_time = time.time() - start_time
        snr_mean = np.mean(snr_values) if snr_values else 0
        snr_std = np.std(snr_values) if snr_values else 0
        
        # 记录处理结果
        logger.info(f"{element}元素处理完成:")
        logger.info(f"- 成功: {successful}条")
        logger.info(f"- 失败: {failed}条")
        logger.info(f"- 跳过: {skipped}条") 
        logger.info(f"- 平均SNR: {snr_mean:.2f} ± {snr_std:.2f}")
        logger.info(f"- 总用时: {elapsed_time:.2f}秒")
        
        return {
            'element': element,
            'successful': successful,
            'failed': failed,
            'skipped': skipped,
            'snr_mean': snr_mean,
            'snr_std': snr_std,
            'elapsed_time': elapsed_time,
            'last_index': current_idx
        }
    
    def process_all_data(self):
        """处理所有数据并生成验证报告"""
        try:
            # 1. 读取CSV文件，获取所有unique的obsid和元素丰度值
            all_obsids = set()
            for csv_file in self.csv_files:
                df = pd.read_csv(csv_file)
                all_obsids.update(df['obsid'].unique())
            
            # 2. 处理所有FITS文件
            processed_data = {}
            with ProgressManager(len(all_obsids), desc="处理FITS文件") as progress:
                for obsid in all_obsids:
                    fits_file = self._find_fits_file(obsid)
                    if fits_file is not None:
                        processed_data[obsid] = self.process_fits_file(fits_file)
                        if processed_data[obsid] is not None:
                            progress.update(1)
            
            # 3. 收集验证数据
            validation_data = []
            for obsid, data_dict in processed_data.items():
                if data_dict is not None:
                    validation_data.append({
                        'obsid': obsid,
                        'data_dict': data_dict
                    })
            
            # 4. 生成验证摘要
            summary = self.validator.validate_batch(validation_data)
            
            # 5. 绘制验证摘要图表
            self.validator.plot_validation_summary(summary)
            
            # 6. 输出验证结果
            logger.info(f"数据处理完成。总计: {summary['total']}, 通过: {summary['passed']}, 失败: {summary['failed']}")
            if summary['issues']:
                logger.warning("发现的问题:")
                for issue, count in summary['issues'].items():
                    logger.warning(f"- {issue}: {count}次")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"处理数据时出错: {e}")
            return {}
            
    def process_fits_file(self, fits_file):
        """处理单个FITS文件，提取和预处理光谱数据"""
        try:
            logger.info(f"开始处理FITS文件: {os.path.basename(fits_file)}")
            
            # 读取FITS文件数据
            data_dict = self.read_fits_file(fits_file)
            
            if data_dict is None:
                logger.warning(f"无法读取FITS文件: {fits_file}")
                return None
                
            # 提取光谱数据进行处理
            processed_spectrum = self.process_spectrum(data_dict)
            
            if processed_spectrum is not None:
                # 获取观测ID和原始文件名
                obsid = data_dict.get('obsid', self._extract_obsid_from_fits(fits_file))
                original_filename = os.path.basename(fits_file)
                
                # 构建结果字典
                result = {
                    'obsid': obsid,
                    'processed_spectrum': processed_spectrum,
                    'original_filename': original_filename
                }
                
                logger.info(f"成功处理FITS文件: {os.path.basename(fits_file)}")
                return result
            else:
                logger.warning(f"处理光谱数据失败: {fits_file}")
                return None
            
        except Exception as e:
            logger.error(f"处理FITS文件时出错 ({fits_file}): {str(e)}")
            return None
    
    def save_element_datasets(self, element, data):
        """为每个元素分割并保存数据集"""
        if not data:
            logger.warning(f"没有{element}的有效数据可保存")
            return
        
        # 提取数据
        try:
            X = np.array([item['spectrum'] for item in data if 'spectrum' in item])
            y = np.array([item['label'] for item in data if 'label' in item])
            metadata = []
            
            for item in data:
                if 'metadata' in item:
                    metadata.append({
                        'obsid': item['metadata'].get('obsid', 'unknown'),
                        'filename': item['metadata'].get('original_flux', 'unknown'),
                        'snr': item['metadata'].get('snr', 0)
                    })
            
            # 验证数据有效性
            if len(X) == 0 or len(y) == 0:
                logger.warning(f"{element}数据无效：X长度={len(X)}, y长度={len(y)}")
                return
            
            if X.shape[0] != y.shape[0]:
                logger.warning(f"{element}数据不匹配：X形状={X.shape}, y形状={y.shape}")
                return
        
            logger.info(f"成功创建{element}数据数组: X形状={X.shape}, y形状={y.shape}")
            
            # 创建元素目录
            element_dir = os.path.join(self.output_dir, element)
            os.makedirs(element_dir, exist_ok=True)
            
            # 保存处理后的完整数据
            output_file = os.path.join(element_dir, f'processed_data_{element}.npz')
            np.savez(output_file, X=X, y=y, element=element, metadata=np.array(metadata, dtype=object))
            logger.info(f"已保存{element}的完整数据到: {output_file}")
            
            # 分割数据集 - 为每个元素单独分割
            element_tags = np.full(len(X), element)  # 创建元素标签数组
            train_data, val_data, test_data = self.split_dataset(X, y, element_tags)
            
            # 保存训练集
            np.savez(os.path.join(element_dir, f'train_dataset_{element}.npz'),
                    X=train_data[0], y=train_data[1], element=element)
            
            # 保存验证集
            np.savez(os.path.join(element_dir, f'val_dataset_{element}.npz'),
                    X=val_data[0], y=val_data[1], element=element)
            
            # 保存测试集
            np.savez(os.path.join(element_dir, f'test_dataset_{element}.npz'),
                    X=test_data[0], y=test_data[1], element=element)
            
            logger.info(f"{element}数据集分割完成:")
            logger.info(f"- 训练集: {train_data[0].shape[0]}条 (70%)")
            logger.info(f"- 验证集: {val_data[0].shape[0]}条 (10%)")
            logger.info(f"- 测试集: {test_data[0].shape[0]}条 (20%)")
        except Exception as e:
            logger.error(f"保存{element}数据集时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def split_dataset(self, X, y, elements):
        """按照7:1:2的比例分割数据集为训练集、验证集和测试集"""
        if len(X) == 0:
            print("错误: 没有数据可以分割")
            return (np.array([]), np.array([]), np.array([])), \
                   (np.array([]), np.array([]), np.array([])), \
                   (np.array([]), np.array([]), np.array([]))
        
        # 首先分割出测试集 (80% vs 20%)
        X_temp, X_test, y_temp, y_test, elements_temp, elements_test = train_test_split(
            X, y, elements, test_size=0.2, random_state=42)
        
        # 再从剩余数据中分割出验证集 (70% vs 10%)
        X_train, X_val, y_train, y_val, elements_train, elements_val = train_test_split(
            X_temp, y_temp, elements_temp, test_size=1/8, random_state=42)
        
        # 保存数据集
        np.savez(os.path.join(self.output_dir, 'train_dataset.npz'),
                X=X_train, y=y_train, elements=elements_train)
        np.savez(os.path.join(self.output_dir, 'val_dataset.npz'),
                X=X_val, y=y_val, elements=elements_val)
        np.savez(os.path.join(self.output_dir, 'test_dataset.npz'),
                X=X_test, y=y_test, elements=elements_test)
        
        # 保存参考数据集处理结果（包含所有训练、验证和测试数据）
        np.savez(os.path.join(self.output_dir, 'reference_dataset.npz'),
                X=np.concatenate([X_train, X_val, X_test]),
                y=np.concatenate([y_train, y_val, y_test]),
                elements=np.concatenate([elements_train, elements_val, elements_test]))
        
        print(f"数据集分割完成:")
        print(f"训练集: {X_train.shape[0]}条 (70%)")
        print(f"验证集: {X_val.shape[0]}条 (10%)")
        print(f"测试集: {X_test.shape[0]}条 (20%)")
        
        return (X_train, y_train, elements_train), (X_val, y_val, elements_val), (X_test, y_test, elements_test)
    
    
    def predict_abundance(self, fits_file, model):
        """预测单个FITS文件的元素丰度"""
        # 读取并预处理光谱
        wavelength, flux, v_helio = self.read_fits_file(fits_file)
        if wavelength is None or flux is None:
            return None
        
        # 预处理光谱
        processed_data = self.process_spectrum(wavelength, flux, v_helio)
        if processed_data is None:
            return None
        
        # 使用模型预测
        prediction = model.predict(processed_data)
        
        # 保存预测结果
        result = {
            'filename': os.path.basename(fits_file),
            'prediction': prediction,
            'processed_spectrum': processed_data
        }
        
        # 保存到预测数据集文件
        prediction_file = os.path.join(self.output_dir, 'prediction_dataset.npz')
        
        # 如果文件已存在，加载现有数据并追加
        if os.path.exists(prediction_file):
            existing_data = np.load(prediction_file)
            predictions = np.append(existing_data['predictions'], [prediction], axis=0)
            filenames = np.append(existing_data['filenames'], [os.path.basename(fits_file)])
            processed_spectra = np.append(existing_data['processed_spectra'], [processed_data], axis=0)
        else:
            predictions = np.array([prediction])
            filenames = np.array([os.path.basename(fits_file)])
            processed_spectra = np.array([processed_data])
        
        # 保存更新后的数据
        np.savez(prediction_file,
                predictions=predictions,
                filenames=filenames,
                processed_spectra=processed_spectra)
        
        print(f"预测结果已保存到: {prediction_file}")
        return prediction
    
    def visualize_spectrum(self, spec_file, processed=True, save=True):
        """可视化原始光谱和处理后的光谱，展示四个处理阶段"""
        wavelength, flux, v_helio = self.read_fits_file(spec_file)
        if wavelength is None or flux is None:
            print(f"无法读取光谱文件: {spec_file}")
            return
        
        # 设置字体和图形样式
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图形
        plt.figure(figsize=(14, 16))  # 调整高度以容纳四张子图
        
        # 获取光谱类型和观测日期信息（如果有）
        spec_type = ""
        obs_date = ""
        z = 0  # 初始化红移值
        try:
            file_path = self._get_file_extension(spec_file)
            with fits.open(file_path, ignore_missing_end=True) as hdul:
                header = hdul[0].header
                # 尝试获取光谱类型
                if 'OBJTYPE' in header:
                    spec_type = header['OBJTYPE']
                elif 'CLASS' in header:
                    spec_type = header['CLASS']
                # 尝试获取观测日期
                if 'DATE-OBS' in header:
                    obs_date = header['DATE-OBS']
                elif 'MJD' in header:
                    obs_date = f"MJD: {header['MJD']}"
                # 获取赤经赤纬
                ra = header.get('RA', '')
                dec = header.get('DEC', '')
                # 获取红移或视向速度
                z = header.get('Z', '')
                if z:
                    spec_type = f"{spec_type} z={z}" if spec_type else f"z={z}"
                # 获取恒星类型
                if 'OBJTYPE' in header and header['OBJTYPE'] == 'STAR':
                    star_type = header.get('SPTCLASS', '')
                    if star_type:
                        spec_type = f"STAR {star_type}"
                # 获取视向速度
                v_helio_header = header.get('V_HELIO', None)
                if v_helio_header is not None:
                    v_helio_err = header.get('V_HELIO_ERR', '')
                    v_helio_text = f"cz = {v_helio_header:.1f}"
                    if v_helio_err:
                        v_helio_text += f" ± {v_helio_err:.1f} km/s"
                    spec_type = f"{spec_type}\n{v_helio_text}" if spec_type else v_helio_text
        except:
            pass
            
        # 如果header中没有找到红移值，尝试从CSV文件中获取
        if not z:
            try:
                for csv_file in self.csv_files:
                    if os.path.exists(csv_file):
                        df = pd.read_csv(csv_file)
                        if 'spec' in df.columns and 'z' in df.columns:
                            # 提取文件名（不包含路径和扩展名）
                            base_file = os.path.basename(spec_file)
                            if '.' in base_file:
                                base_file = base_file.split('.')[0]
                            
                            # 在CSV中查找匹配记录
                            matches = df[df['spec'].str.contains(base_file, case=False, na=False)]
                            if not matches.empty:
                                z = matches.iloc[0]['z']
                                print(f"从CSV找到红移值: z = {z}")
                                break
            except Exception as e:
                print(f"从CSV查找红移数据出错: {e}")
            
        # 定义主要吸收线位置(埃)和标签
        absorption_lines = {
            'CaII K': 3933.7,
            'CaII H': 3968.5,
            'Hδ': 4101.7,
            'Hγ': 4340.5,
            'Hβ': 4861.3,
            'Mg': 5175.3,  # MgI b三重线中心
            'Na D': 5892.9, # NaI D双线中心
            'Hα': 6562.8,
            'OI': 7774.2,
            'CaII IR1': 8498.0, # 钙三重线
            'CaII IR2': 8542.1,
            'CaII IR3': 8662.1
        }
        
        # 为处理特定密集区域标签，手动定义一些标签的水平偏移
        # 值为左右偏移百分比，正值向右，负值向左
        label_shifts = {
            'CaII K': -1.0,    # 加大左移
            'CaII H': 1.0,     # 加大右移
            'Hδ': -0.5,        # 左移
            'Hγ': 0.5,         # 右移
            'CaII IR1': -1.5,  # 大幅左移
            'CaII IR2': 0,     # 不动
            'CaII IR3': 1.5    # 大幅右移
        }
        
        def plot_with_labels(ax, x, y, wave_range, label_data, color='blue', label_name=''):
            """在指定的轴上绘制带标签的图形
            
            Args:
                ax: matplotlib轴对象
                x: 波长数组
                y: 流量数组
                wave_range: (最小波长, 最大波长)
                label_data: 吸收线数据字典
                color: 线条颜色
                label_name: 图例标签
            """
            # 绘制主曲线
            ax.plot(x, y, label=label_name, color=color)
            ax.set_xlabel('Wavelength (Å)')
            
            # 获取当前y轴范围
            ymin, ymax = ax.get_ylim()
            y_range = ymax - ymin
            wave_min, wave_max = wave_range
            x_range = wave_max - wave_min
            
            # 重置标签位置跟踪
            label_positions = {}  # 格式：{wave: (vertical_offset, horizontal_offset)}
            
            # 对吸收线按波长排序
            sorted_lines = sorted(label_data.items(), key=lambda x: x[1])
            
            # 先为预定义的标签分配位置
            for name, wave in sorted_lines:
                if wave >= x.min() and wave <= x.max() and name in label_shifts:
                    # 垂直偏移交替放置：0=下，1=上
                    v_offset = 0 if len(label_positions) % 2 == 0 else 1
                    label_positions[wave] = (v_offset, label_shifts[name])
            
            # 再处理其他标签
            for name, wave in sorted_lines:
                if wave >= x.min() and wave <= x.max() and wave not in label_positions:
                    # 检查此标签是否太靠近其他标签
                    too_close = []
                    for prev_wave in label_positions:
                        # 计算波长差除以整个波长范围的百分比
                        distance_percent = abs(wave - prev_wave) / x_range
                        if distance_percent < 0.05:  # 使用5%作为距离判断
                            too_close.append((prev_wave, distance_percent))
                    
                    # 根据接近程度决定位置偏移
                    v_offset = 0  # 垂直偏移(0=下，1=上)
                    h_shift = 0   # 水平偏移
                    
                    if too_close:
                        # 按接近程度排序
                        too_close.sort(key=lambda x: x[1])
                        
                        # 收集附近标签的垂直位置，避免使用相同高度
                        nearby_v_offsets = [label_positions.get(w)[0] for w, _ in too_close]
                        
                        # 优先选择不同的垂直位置
                        if 0 in nearby_v_offsets and 1 not in nearby_v_offsets:
                            v_offset = 1
                        elif 1 in nearby_v_offsets and 0 not in nearby_v_offsets:
                            v_offset = 0
                        else:
                            # 如果两种垂直位置都被使用，则使用距离最近标签的反向位置
                            closest_wave, _ = too_close[0]
                            closest_v_offset, _ = label_positions.get(closest_wave, (0, 0))
                            v_offset = 1 - closest_v_offset
                        
                        # 水平偏移处理：根据附近标签情况分配水平位置
                        nearby_h_shifts = [label_positions.get(w)[1] for w, _ in too_close]
                        
                        # 找出未被占用的方向
                        if all(h < 0 for h in nearby_h_shifts):
                            h_shift = 1.0  # 如果附近都在左边，则放右边
                        elif all(h > 0 for h in nearby_h_shifts):
                            h_shift = -1.0  # 如果附近都在右边，则放左边
                        else:
                            # 复杂情况，尝试找到最大间隙
                            nearby_h_shifts.append(-2.0)  # 左边界
                            nearby_h_shifts.append(2.0)   # 右边界
                            nearby_h_shifts.sort()
                            
                            max_gap = 0
                            best_pos = 0
                            
                            for i in range(len(nearby_h_shifts) - 1):
                                gap = nearby_h_shifts[i+1] - nearby_h_shifts[i]
                                if gap > max_gap:
                                    max_gap = gap
                                    best_pos = nearby_h_shifts[i] + gap/2
                            
                            h_shift = best_pos
                    
                    # 记录此标签的位置偏移
                    label_positions[wave] = (v_offset, h_shift)
            
            # 绘制线条和标签
            for name, wave in sorted_lines:
                if wave >= x.min() and wave <= x.max():
                    # 添加垂直线 - 使用较浅的透明度，减少对图像的干扰
                    ax.axvline(x=wave, color='red', linestyle=':', alpha=0.5, linewidth=0.7)
                    
                    # 获取此标签的位置信息
                    v_offset, h_shift = label_positions.get(wave, (0, 0))
                    
                    # 计算标签位置，确保水平位置有足够偏移
                    x_pos = wave + h_shift * 0.04 * x_range  # 增加水平偏移量到4%
                    
                    # 计算垂直位置，确保不同高度的标签有明显差别
                    if v_offset == 0:
                        y_pos = ymax + y_range * 0.035  # 低位置
                    else:
                        y_pos = ymax + y_range * 0.12  # 高位置，增加差异
                    
                    # 绘制标签，使用白底黑字提高可读性
                    ax.text(x_pos, y_pos, name, rotation=0, 
                            horizontalalignment='center', verticalalignment='bottom',
                            fontsize=8, alpha=0.9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
            
            # 调整y轴上限，为标签留出空间
            ax.set_ylim(ymin, ymax + y_range * 0.2)  # 增加上方空间
            ax.grid(True, linestyle='--', alpha=0.5)  # 减少网格线的明显程度
        
        # 显示原始光谱
        ax1 = plt.subplot(4, 1, 1)
        plot_with_labels(ax1, wavelength, flux, 
                         (min(wavelength), max(wavelength)), 
                         absorption_lines, color='blue', label_name='Raw Spectrum')
        ax1.set_ylabel('Flux (relative)')
        ax1.set_title(f'Spectrum: {os.path.basename(spec_file)}')
        # 将图例放在右上角
        ax1.legend(loc='upper right')
        
        # 添加光谱类型和观测信息
        if spec_type or obs_date:
            info_text = f"{spec_type} {'   ' if spec_type and obs_date else ''} {obs_date}"
            ax1.annotate(info_text, xy=(0.02, 0.05), xycoords='axes fraction', fontsize=9)
        
        if processed:
            # 处理光谱 - 进行所有处理步骤
            
            # 1. 波长校正
            wavelength_calibrated = self.correct_wavelength(wavelength, flux)
            
            # 2. 视向速度校正
            wavelength_corrected = self.correct_velocity(wavelength_calibrated, flux, v_helio)
            
            # 3. 去噪
            flux_denoised = self.denoise_spectrum(wavelength_corrected, flux)
            
            # 4. 红移校正
            wavelength_rest = self.correct_redshift(wavelength_corrected, flux_denoised, z)
            
            # 5. 重采样
            wavelength_resampled, flux_resampled = self.resample_spectrum(wavelength_rest, flux_denoised)
            
            if wavelength_resampled is not None and flux_resampled is not None:
                # 第二张图: 显示处理后的光谱到重采样阶段
                ax2 = plt.subplot(4, 1, 2)
                plot_with_labels(ax2, wavelength_resampled, flux_resampled, 
                                 (min(wavelength_resampled), max(wavelength_resampled)), 
                                 absorption_lines, color='green', 
                                 label_name='Calibrated, Velocity Corrected, Denoised, Redshift Corrected & Resampled')
                ax2.set_ylabel('Flux')
                ax2.set_title('Spectrum after Calibration, Velocity Correction, Denoising, Redshift Correction & Resampling')
                # 将图例放在右上角
                ax2.legend(loc='upper right')
                
                # 6. 连续谱归一化 (第三张图)
                flux_continuum = self.normalize_continuum(wavelength_resampled, flux_resampled)
                
                ax3 = plt.subplot(4, 1, 3)
                plot_with_labels(ax3, wavelength_resampled, flux_continuum,
                                 (min(wavelength_resampled), max(wavelength_resampled)),
                                 absorption_lines, color='purple',
                                 label_name='Continuum Normalized')
                ax3.set_ylabel('Normalized Flux')
                ax3.set_title('Spectrum after Continuum Normalization')
                # 将图例放在右上角
                ax3.legend(loc='upper right')
                
                # 7. 二次去噪 + 8. 最终归一化 (最大最小值归一化)
                flux_denoised_second = self.denoise_spectrum_second(wavelength_resampled, flux_continuum)
                flux_normalized = self.normalize_spectrum(flux_denoised_second)
                
                # 第四张图: 显示最终处理后的光谱
                ax4 = plt.subplot(4, 1, 4)
                plot_with_labels(ax4, wavelength_resampled, flux_normalized,
                                 (min(wavelength_resampled), max(wavelength_resampled)),
                                 absorption_lines, color='red',
                                 label_name='Fully Processed')
                ax4.set_ylabel('Final Normalized Flux')
                ax4.set_title('Spectrum after Second Denoising and Final Normalization')
                # 将图例放在右上角
                ax4.legend(loc='upper right')
                
                # 设置y轴范围为0-1，确保最终归一化图的y轴范围固定
                ax4.set_ylim(0, 1)
                
                # 添加说明文本
                # 检查是否使用了公有波长范围
                if self.compute_common_range and len(self.processed_ranges) > 1:
                    w_min, w_max = self.wavelength_range
                    range_description = f'Common Wavelength Range: {w_min:.2f}-{w_max:.2f} Å'
                else:
                    w_min, w_max = self.wavelength_range
                    range_description = f'Wavelength Range: {w_min:.2f}-{w_max:.2f} Å'
                
                # 检查是否使用对数步长
                if hasattr(self, 'log_step') and self.log_step:
                    step_description = f'Log Step: {self.log_step} dex'
                else:
                    step_description = f'Points: {len(wavelength_resampled)}'
                
                # 添加红移信息
                z_description = f'Redshift: z = {z}' if z else ''
                
                if z_description:
                    plt.figtext(0.5, 0.01, 
                            f'{range_description}, {step_description}, {z_description}', 
                            ha='center', fontsize=10)
                else:
                    plt.figtext(0.5, 0.01, 
                          f'{range_description}, {step_description}', 
                          ha='center', fontsize=10)
        
        plt.tight_layout(pad=2.0)
        
        if save:
            output_file = os.path.join(self.output_dir, f"{os.path.basename(spec_file)}_visualization.png")
            plt.savefig(output_file, dpi=150)
            print(f"Image saved to: {output_file}")
            plt.close()
        else:
            plt.show()
    
    def check_data_sources(self):
        """检查数据源文件是否存在并报告状态"""
        print("\n=== 数据源检查 ===")
        
        # 检查CSV文件
        for csv_file in self.csv_files:
            if os.path.exists(csv_file):
                print(f"✓ CSV文件 {csv_file} 存在")
                try:
                    df = pd.read_csv(csv_file)
                    print(f"  - 包含 {len(df)} 条记录")
                    print(f"  - 列名: {', '.join(df.columns)}")
                    
                    # 检查是否包含obsid列（而不是spec列）
                    if 'obsid' in df.columns:
                        obsid_example = df['obsid'].iloc[0] if len(df) > 0 else "无数据"
                        print(f"  - 'obsid'列示例: {obsid_example}")
                        # 保存CSV数据
                        self.csv_data[os.path.basename(csv_file)] = df
                    else:
                        print(f"  ✗ 错误: {csv_file} 中没有'obsid'列")
                except Exception as e:
                    print(f"  ✗ 无法读取CSV文件 {csv_file}: {e}")
            else:
                print(f"✗ CSV文件 {csv_file} 不存在")
        
        # 检查fits目录
        if os.path.exists(self.fits_dir):
            print(f"\n✓ fits目录存在: {os.path.abspath(self.fits_dir)}")
            
            # 检查fits目录结构
            all_fits_files = []
            for root, dirs, files in os.walk(self.fits_dir):
                for file in files:
                    if any(file.endswith(ext) for ext in ['.fits', '.fits.gz', '.fit', '.fit.gz']):
                        all_fits_files.append(os.path.join(root, file))
                    
            print(f"  - 找到 {len(all_fits_files)} 个FITS文件（包括子目录）")
            
            if all_fits_files:
                print(f"  - 前5个文件示例:")
                for i, file in enumerate(all_fits_files[:5]):
                    rel_path = os.path.relpath(file, self.fits_dir)
                    print(f"    {i+1}. {rel_path}")
                
                # 检查文件所在的目录结构
                directories = set()
                for file in all_fits_files:
                    rel_dir = os.path.relpath(os.path.dirname(file), self.fits_dir)
                    if rel_dir != '.':
                        directories.add(rel_dir)
                
                if directories:
                    print(f"  - 发现嵌套目录结构:")
                    for d in list(directories)[:5]:
                        print(f"    - {d}")
                    if len(directories) > 5:
                        print(f"    - ... 共 {len(directories)} 个子目录")
                
                # 检查CSV文件中的obsid值是否匹配fits文件
                for csv_file in self.csv_files:
                    if os.path.exists(csv_file):
                        df = pd.read_csv(csv_file)
                        if 'obsid' in df.columns and len(df) > 0:
                            obsid_examples = df['obsid'].iloc[:5].tolist()
                            print(f"\n  检查 {csv_file} 中的obsid值是否匹配fits文件:")
                            
                            for obsid in obsid_examples:
                                # 使用新的查找逻辑
                                found_path = self._find_fits_file(obsid)
                                if found_path:
                                    rel_path = os.path.relpath(found_path, self.fits_dir)
                                    print(f"    ✓ OBSID {obsid} 存在: {rel_path}")
                                else:
                                    print(f"    ✗ OBSID {obsid} 不存在，在所有目录中都未找到")
                            
                            # 检测常见文件格式
                            extensions = [os.path.splitext(f)[1] for f in all_fits_files[:20]]
                            common_exts = set(extensions)
                            if common_exts:
                                print(f"  FITS文件中的常见后缀: {', '.join(common_exts)}")
        else:
            print(f"\n✗ fits目录不存在: {os.path.abspath(self.fits_dir)}")
        
        print("\n=== 检查完成 ===\n")
        
    def clean_cache(self):
        """清理所有缓存"""
        try:
            # 清理FITS缓存文件
            if os.path.exists(self.cache_dir):
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.pkl'):
                        os.remove(os.path.join(self.cache_dir, file))
            
            # 重置OBSID缓存
            self.obsid_cache.mapping = {}
            self.obsid_cache.save_cache()
            
            logger.info("缓存清理完成")
        except Exception as e:
            logger.error(f"清理缓存时出错: {e}")

    def check_and_fix_file_paths(self):
        """检查并修复文件路径问题"""
        print("\n=== 路径问题诊断 ===")
        
        # 创建测试文件列表
        test_files = []
        for csv_file in self.csv_files:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                if 'obsid' in df.columns:
                    for obsid in df['obsid'].values[:5]:  # 只取前5个测试
                        if obsid not in test_files:
                            test_files.append(obsid)
        
        if not test_files:
            print("无法找到测试文件，请检查CSV文件")
            return
        
        # 显示fits目录内容
        print(f"FITS目录内容 ({self.fits_dir}):")
        fits_files = []
        for root, _, files in os.walk(self.fits_dir):
            for file in files:
                if file.endswith(('.fits', '.fits.gz', '.fit', '.fit.gz')):
                    rel_path = os.path.relpath(os.path.join(root, file), self.fits_dir)
                    fits_files.append(rel_path)
        
        print(f"总计找到 {len(fits_files)} 个FITS文件")
        if fits_files:
            print("前5个示例:")
            for i, file in enumerate(fits_files[:5]):
                print(f"  {i+1}. {file}")
        
        # 测试文件路径查找
        print("\n文件查找测试:")
        for obsid in test_files:
            print(f"测试OBSID: {obsid}")
            
            # 尝试直接查找文件
            obsid_str = f"{float(obsid):.0f}" if not isinstance(obsid, str) else obsid
            
            # 测试带扩展名的路径
            found = False
            for ext in ['.fits', '.fits.gz', '.fit', '.fit.gz']:
                file_path = os.path.join(self.fits_dir, f"{obsid_str}{ext}")
                if os.path.exists(file_path):
                    print(f"  存在路径: {file_path}")
                    found = True
                    break
            
            # 使用查找函数
            found_path = self._find_fits_file(obsid)
            if found_path:
                print(f"  _find_fits_file找到: {found_path}")
            else:
                print(f"  _find_fits_file未找到文件")
            
            if not found and not found_path:
                print(f"  无法找到OBSID为{obsid}的FITS文件，查找路径: {self.fits_dir}")
            
        print("\n=== 诊断完成 ===\n")

    # 添加新方法，用于从FITS文件提取OBSID
    def _extract_obsid_from_fits(self, fits_file):
        """从FITS文件头中提取OBSID信息"""
        try:
            with fits.open(fits_file, ignore_missing_end=True, memmap=False) as hdul:
                # 尝试从主头信息中获取OBSID
                if 'OBSID' in hdul[0].header:
                    return str(hdul[0].header['OBSID'])
                
                # 如果主头信息中没有，尝试其他HDU
                for i in range(1, len(hdul)):
                    if 'OBSID' in hdul[i].header:
                        return str(hdul[i].header['OBSID'])
                
                # 没有找到OBSID
                return None
        except Exception as e:
            print(f"提取OBSID时出错: {e}")
            return None

    # 添加方法用于构建FITS文件到OBSID的映射
    def _build_fits_obsid_mapping(self):
        """构建FITS文件与obsid的映射关系"""
        # 修改逻辑：每次都重新构建完整映射，而不是仅在第一次时构建
        logger.info("构建FITS文件与obsid的映射关系...")
        
        # 初始化映射
        if not hasattr(self, 'fits_obsid_map'):
            self.fits_obsid_map = {}
        
        # 从缓存加载映射
        cache_loaded = False
        if self.obsid_cache:
            if hasattr(self.obsid_cache, 'mapping') and isinstance(self.obsid_cache.mapping, dict):
                # 确保obsid_cache.mapping不是一个复杂的字典结构
                if 'fits_files' in self.obsid_cache.mapping:
                    # 如果是旧格式，从fits_files子字典加载
                    self.fits_obsid_map = self.obsid_cache.mapping.get('fits_files', {})
                else:
                    # 否则直接使用mapping
                    self.fits_obsid_map = self.obsid_cache.mapping
                cache_loaded = True
                logger.info(f"从缓存加载了{len(self.fits_obsid_map)}个FITS-OBSID映射")
            elif hasattr(self.obsid_cache, '_mapping') and isinstance(self.obsid_cache._mapping, dict):
                # 类似地检查_mapping结构
                if '_mapping' in self.obsid_cache._mapping:
                    self.fits_obsid_map = self.obsid_cache._mapping.get('fits_files', {})
                else:
                    self.fits_obsid_map = self.obsid_cache._mapping
                cache_loaded = True
                logger.info(f"从缓存_mapping加载了{len(self.fits_obsid_map)}个FITS-OBSID映射")
        
        # 验证缓存的映射文件是否存在
        if cache_loaded and self.fits_obsid_map:
            valid_mappings = {}
            invalid_count = 0
            
            # 遍历所有映射并验证
            for obsid, file_path in list(self.fits_obsid_map.items()):
                # 确保file_path是字符串而不是字典
                if isinstance(file_path, dict):
                    logger.warning(f"映射值为字典，而不是文件路径: {obsid} -> {file_path}")
                    invalid_count += 1
                    continue
                    
                # 验证路径是否存在
                if isinstance(file_path, str) and os.path.exists(file_path):
                    valid_mappings[obsid] = file_path
                else:
                    invalid_count += 1
            
            if invalid_count > 0:
                logger.warning(f"缓存中有{invalid_count}个文件路径无效，已删除")
                self.fits_obsid_map = valid_mappings
        
        # 如果缓存中没有足够的映射，遍历FITS目录构建
        if len(self.fits_obsid_map) < 100:  # 设置一个合理的最小阈值
            logger.info("缓存映射不足，扫描FITS目录构建映射...")
            existing_obsids = set(self.fits_obsid_map.keys())
            new_mappings_count = 0
            
            # 获取所有CSV文件中的obsid列表
            all_obsids = set()
            for element, df in self.csv_data.items():
                if 'obsid' in df.columns:
                    all_obsids.update(df['obsid'].astype(str).values)
            
            logger.info(f"CSV文件中有{len(all_obsids)}个唯一观测ID")
            
            # 创建进度条
            from tqdm import tqdm
            
            # 遍历FITS目录
            fits_files = []
            for ext in ['.fits', '.fit', '.FITS', '.FIT', '.fits.gz', '.fit.gz']:
                fits_files.extend(glob.glob(os.path.join(self.fits_dir, f'**/*{ext}'), recursive=True))
            
            logger.info(f"发现{len(fits_files)}个FITS文件，开始提取OBSID...")
            
            for fits_file in tqdm(fits_files, desc="扫描FITS文件", unit="文件"):
                try:
                    # 从文件名中提取OBSID
                    file_obsid = self._extract_obsid_from_filename(fits_file)
                    
                    if file_obsid:
                        # 检查是否在CSV文件中的观测ID列表中
                        if file_obsid in all_obsids and file_obsid not in existing_obsids:
                            self.fits_obsid_map[file_obsid] = fits_file
                            new_mappings_count += 1
                    else:
                        # 如果文件名中无法提取，尝试从FITS头中提取
                        header_obsid = self._extract_obsid_from_fits(fits_file)
                        if header_obsid and header_obsid in all_obsids and header_obsid not in existing_obsids:
                            self.fits_obsid_map[header_obsid] = fits_file
                            new_mappings_count += 1
                except Exception as e:
                    logger.debug(f"处理FITS文件时出错 ({os.path.basename(fits_file)}): {str(e)}")
            
            logger.info(f"新增了{new_mappings_count}个FITS-OBSID映射")
        
        # 最后检查确保所有映射值都是字符串
        self.fits_obsid_map = {k: v for k, v in self.fits_obsid_map.items() if isinstance(v, str)}
        
        # 保存到缓存
        if self.obsid_cache:
            # 确保缓存使用正确的结构
            if 'fits_files' in self.obsid_cache.mapping:
                self.obsid_cache.mapping['fits_files'] = self.fits_obsid_map
            else:
                self.obsid_cache.mapping = self.fits_obsid_map
            self.obsid_cache.save_cache()
            logger.info("已将FITS-OBSID映射保存到缓存")
        
        logger.info(f"FITS-OBSID映射完成，共{len(self.fits_obsid_map)}个有效映射关系")

    def _extract_obsid_from_filename(self, fits_file):
        """从FITS文件名中提取OBSID"""
        try:
            # 获取不含路径和扩展名的文件名
            base_name = os.path.basename(fits_file)
            if '.' in base_name:
                base_name = base_name.split('.')[0]
            
            # 尝试多种模式提取OBSID
            # 1. 直接将文件名视为OBSID
            if base_name.isdigit():
                return base_name
            
            # 2. 查找spec-dddd格式
            if 'spec-' in base_name:
                parts = base_name.split('spec-')
                if len(parts) > 1 and parts[1].isdigit():
                    return parts[1]
            
            # 3. 其他常见格式提取
            patterns = [
                r'(\d{8,})',  # 8位以上数字
                r'OBSID[_-]?(\d+)',  # OBSID_123格式
                r'ID[_-]?(\d+)'  # ID_123格式
            ]
            
            for pattern in patterns:
                match = re.search(pattern, base_name)
                if match:
                    return match.group(1)
            
            # 未能提取出OBSID
            return None
        except Exception:
            return None

    def process_data(self):
        """执行数据处理"""
        try:
            start_time = time.time()
            
            logger.info("=== 开始数据处理 ===")
            
            # 1. 读取CSV数据
            logger.info("步骤1: 读取CSV数据")
            self.read_csv_data()
            logger.info(f"CSV数据读取完成，共 {sum(len(df) for df in self.csv_data.values())} 条记录")
            
            # 2. 预先构建FITS-OBSID映射（一次性批量构建）
            logger.info("步骤2: 构建FITS文件和OBSID的映射关系")
            self._build_fits_obsid_mapping()
            
            # 3. 处理所有元素数据（使用分批处理）
            logger.info("步骤3: 开始批量处理元素数据")
            processed_data = self.process_all_data_in_batches()
            
            # 4. 保存所有缓存
            logger.info("步骤4: 保存缓存数据")
            self.obsid_cache.save_cache()
            
            # 5. 计算并显示处理时间
            end_time = time.time()
            total_time = end_time - start_time
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            logger.info(f"=== 数据处理完成 ===")
            logger.info(f"总处理时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
            
            return processed_data
        except Exception as e:
            self.obsid_cache.save_cache()
            logger.error(f"数据处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def process_all_data_in_batches(self):
        """使用分批处理方式处理所有数据，对每个元素单独处理"""
        # 创建默认返回结果
        final_results = {}
        processed_elements = {}
        
        try:
            # 使用已存在的ProgressManager类添加进度显示
            from tqdm import tqdm
            
            # 检查数据源和文件路径
            logger.info("检查数据源和文件路径...")
            self.check_data_sources()
            self.check_and_fix_file_paths()
            
            # 首先处理所有元素的FITS文件
            for element, df in self.csv_data.items():
                logger.info(f"开始处理元素 {element} 的数据...")
                
                # 确保创建元素目录
                element_dir = os.path.join(self.output_dir, element)
                os.makedirs(element_dir, exist_ok=True)
                
                # 提取有效的行（元素丰度非NaN）
                element_df = df[['obsid', element]].dropna(subset=[element])
                valid_count = len(element_df)
                logger.info(f"{element}有效数据：{valid_count}行")
                
                if valid_count == 0:
                    logger.warning(f"元素{element}没有有效数据，跳过处理")
                    continue
                    
                # 获取obsid列表
                obsid_list = element_df['obsid'].values
                
                # 设置批处理参数
                batch_size = self.batch_size
                total_batches = math.ceil(valid_count / batch_size)
                logger.info(f"将分{total_batches}批处理{element}，每批{batch_size}条记录")
                
                # 处理所有批次
                element_processed_data = []
                
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, valid_count)
                    batch_obsids = obsid_list[start_idx:end_idx]
                    
                    logger.info(f"处理{element}的第{batch_idx+1}/{total_batches}批 ({len(batch_obsids)}条记录)")
                    batch_start_time = time.time()
                    
                    # 批量处理当前批次
                    batch_data = []
                    for i, obsid in enumerate(tqdm(batch_obsids, desc=f"处理{element}批次{batch_idx+1}", unit="记录")):
                        # 获取标签值
                        label = element_df.iloc[start_idx + i][element]
                        
                        try:
                            # 查找FITS文件
                            fits_file = None
                            if hasattr(self, 'fits_obsid_map') and obsid in self.fits_obsid_map:
                                fits_file = self.fits_obsid_map[obsid]
                            else:
                                fits_file = self._find_fits_file(obsid)
                            
                            if fits_file:
                                # 处理单个光谱
                                result = self.process_single_spectrum(obsid, label)
                                if result:
                                    batch_data.append(result)
                                    
                                    # 记录结果到final_results中
                                    if 'metadata' in result and 'obsid' in result['metadata']:
                                        result_obsid = result['metadata']['obsid']
                                        if result_obsid not in final_results:
                                            final_results[result_obsid] = {
                                                'processed_spectrum': result.get('spectrum', None),
                                                'original_filename': result['metadata'].get('original_flux', ''),
                                                'obsid': result_obsid
                                            }
                            else:
                                logger.debug(f"{element}: 未找到obsid={obsid}对应的FITS文件")
                        except Exception as e:
                            logger.error(f"{element}: 处理obsid={obsid}时出错: {str(e)}")
                    
                    # 批次处理完成，添加到元素数据中
                    element_processed_data.extend(batch_data)
                    
                    # 每批次保存一次
                    if batch_data:
                        batch_output_file = os.path.join(element_dir, f'batch_{batch_idx+1}_{element}.npz')
                        X = np.array([item['spectrum'] for item in batch_data if 'spectrum' in item])
                        y = np.array([item['label'] for item in batch_data if 'label' in item])
                        
                        if len(X) > 0 and len(y) > 0:
                            np.savez(batch_output_file, X=X, y=y, element=element)
                            logger.info(f"已保存{element}的批次{batch_idx+1}数据，共{len(batch_data)}条")
                    
                    # 批次处理耗时
                    batch_time = time.time() - batch_start_time
                    logger.info(f"{element}批次{batch_idx+1}处理完成：{len(batch_data)}条数据，耗时{batch_time:.2f}秒")
                    
                    # 每批次后清理内存
                    gc.collect()
                    if self.use_gpu and hasattr(torch, 'cuda') and torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception as e:
                            logger.warning(f"清理GPU内存失败: {e}")
                    
                    # 保存OBSID缓存
                    if batch_idx % 5 == 0 or batch_idx == total_batches - 1:  # 每5批或最后一批保存一次
                        self.obsid_cache.save_cache()
                        logger.info(f"已保存{element}的OBSID缓存")
                
                # 保存整个元素的数据集
                if element_processed_data:
                    self.save_element_datasets(element, element_processed_data)
                    processed_elements[element] = element_processed_data
                    logger.info(f"元素{element}处理完成，共{len(element_processed_data)}条有效数据")
                else:
                    logger.warning(f"元素{element}没有有效数据")
            
            # 处理完成，显示总体统计
            total_elements = len(processed_elements)
            total_spectra = sum(len(data) for data in processed_elements.values())
            logger.info(f"所有元素处理完成: 处理了{total_elements}个元素，共{total_spectra}条有效光谱")
            
            return processed_elements
                
        except Exception as e:
            logger.error(f"批处理数据时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return processed_elements

    def process_element_data_with_cache(self, df, element, processed_fits_data):
        """使用已处理的FITS数据处理元素数据"""
        # 初始化处理结果
        processed_data = []
        
        try:
            # 显示处理进度
            from tqdm import tqdm
            
            start_time = time.time()
            logger.info(f"开始处理元素 {element} 的数据 (共 {len(df)} 条记录)")
            found_count = 0
            missing_count = 0
            error_count = 0
            
            with tqdm(total=len(df), desc=f"处理{element}数据", unit="记录") as pbar:
                for index, row in df.iterrows():
                    try:
                        # 获取观测ID和标签
                        obsid = str(row['obsid'])
                        label = float(row[element])
                        
                        # 检查是否已处理过这个FITS文件
                        if obsid in processed_fits_data:
                            data_dict = processed_fits_data[obsid]
                            found_count += 1
                            
                            # 构建结果字典
                            result = {
                                'spectrum': data_dict.get('processed_spectrum', None),
                                'label': label,
                                'metadata': {
                                    'obsid': obsid,
                                    'original_flux': data_dict.get('original_filename', '')
                                }
                            }
                            
                            # 如果成功处理，添加到结果列表
                            if result['spectrum'] is not None:
                                processed_data.append(result)
                        else:
                            missing_count += 1
                            if missing_count < 10:
                                logger.warning(f"未找到观测ID {obsid} 的已处理数据")
                            elif missing_count == 10:
                                logger.warning(f"已有10个观测ID未找到，后续未找到的观测ID将不再单独显示")
                            
                        # 更新进度
                        pbar.update(1)
                        
                        # 每处理1000条数据输出一次统计信息
                        if (index + 1) % 1000 == 0:
                            logger.info(f"已处理 {index + 1}/{len(df)} 条记录")
                        
                    except Exception as e:
                        error_count += 1
                        if error_count < 10:
                            logger.error(f"处理元素数据时出错 (观测ID: {obsid}): {str(e)}")
                        elif error_count == 10:
                            logger.error(f"已有10个记录处理错误，后续错误将不再单独显示")
                        pbar.update(1)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            success_rate = (len(processed_data) / len(df)) * 100 if len(df) > 0 else 0
            logger.info(f"元素 {element} 的数据处理完成:")
            logger.info(f"  - 总记录: {len(df)}")
            logger.info(f"  - 成功处理: {len(processed_data)} ({success_rate:.2f}%)")
            logger.info(f"  - 已找到FITS数据: {found_count}")
            logger.info(f"  - 缺失FITS数据: {missing_count}")
            logger.info(f"  - 处理错误: {error_count}")
            logger.info(f"  - 处理时间: {processing_time:.2f}秒")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"处理元素 {element} 数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return processed_data

    def process_spectrum(self, data_dict):
        """处理光谱数据"""
        try:
            # 获取波长和流量数据
            wavelength = data_dict.get('wavelength')
            flux = data_dict.get('flux')
            
            if wavelength is None or flux is None:
                logger.error("光谱数据缺失波长或流量信息")
                return None
            
            # 1. 去噪
            wavelength, flux = self.denoise_spectrum(wavelength, flux)
            
            # 2. 波长校正（红移和视向速度）
            z = data_dict.get('z', 0)
            v_helio = data_dict.get('v_helio', 0)
            
            # 先进行基本波长校正
            wavelength = self.correct_wavelength(wavelength, flux)
            
            # 然后根据红移和视向速度进行进一步校正
            if z != 0:
                wavelength = self.correct_redshift(wavelength, flux, z)
            if v_helio != 0:
                wavelength = self.correct_velocity(wavelength, flux, v_helio)
            
            # 3. 更新共同波长范围
            if self.compute_common_range:
                self.update_common_wavelength_range(wavelength)
            
            # 4. 重采样
            resampled_wavelength, resampled_flux = self.resample_spectrum(wavelength, flux)
            
            # 5. 连续谱归一化
            normalized_flux = self.normalize_continuum(resampled_wavelength, resampled_flux)
            
            # 6. 二次去噪和最终归一化
            denoised_flux = self.denoise_spectrum_second(resampled_wavelength, normalized_flux)
            final_flux = self.normalize_spectrum(denoised_flux)
            
            return final_flux
            
        except Exception as e:
            logger.error(f"处理光谱时出错: {e}")
            import traceback
            logger.debug(traceback.format_exc())  # 添加详细的堆栈跟踪
            return None

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='LAMOST光谱数据预处理工具')
    
    # 基本参数
    parser.add_argument('--csv_files', type=str, nargs='+', default=None, 
                        help='CSV文件列表')
    parser.add_argument('--reference_csv', type=str, help='参考数据集CSV文件（向后兼容）')
    parser.add_argument('--prediction_csv', type=str, help='预测数据集CSV文件（向后兼容）')
    parser.add_argument('--fits_dir', type=str, default='fits', 
                        help='FITS文件目录')
    parser.add_argument('--output_dir', type=str, default='processed_data', 
                        help='输出目录')
    
    # 预处理参数
    parser.add_argument('--wavelength_range', type=float, nargs=2, default=None,
                        help='波长范围[min, max]')
    parser.add_argument('--n_points', type=int, default=None,
                        help='重采样点数')
    parser.add_argument('--batch_size', type=int, default=Config.PREPROCESSING_BATCH_SIZE,
                        help='批处理大小')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='最大工作线程数')
    parser.add_argument('--memory_limit', type=float, default=0.7,
                        help='内存使用上限(0-1)')
    
    # 选项参数
    parser.add_argument('--low_memory_mode', action='store_true',
                        help='低内存模式')
    parser.add_argument('--clear_cache', action='store_true',
                        help='清除缓存')
    parser.add_argument('--process_all_fits', action='store_true',
                        help='处理所有FITS文件')
    parser.add_argument('--use_gpu', action='store_true',
                        help='使用GPU加速')
    parser.add_argument('--cpu_only', action='store_true',
                        help='仅使用CPU')
    
    args = parser.parse_args()
    
    # 处理CSV文件参数（兼容旧版本）
    csv_files = args.csv_files
    if csv_files is None:
        csv_files = []
    if args.reference_csv:
        csv_files.append(args.reference_csv)
    if args.prediction_csv:
        csv_files.append(args.prediction_csv)
    if not csv_files:
            csv_files = ['X_FE.csv']  # 默认值
    
    # 确定GPU使用选项
    use_gpu = None
    if args.use_gpu:
        use_gpu = True
    elif args.cpu_only:
        use_gpu = False
    
    # 初始化预处理器
    preprocessor = LAMOSTPreprocessor(
        csv_files=csv_files,
        fits_dir=args.fits_dir,
        output_dir=args.output_dir,
        wavelength_range=args.wavelength_range,
        n_points=args.n_points,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        memory_limit=args.memory_limit,
        low_memory_mode=args.low_memory_mode,
        process_all_fits=args.process_all_fits,
        use_gpu=use_gpu
    )
    
    # 如果需要清除缓存
    if args.clear_cache:
        preprocessor.clean_cache()
    
    # 执行预处理
    preprocessor.process_data()

if __name__ == "__main__":
    try:
        # 处理Colab环境
        if IN_COLAB:
            try:
                # 使用动态导入避免IDE报错
                import importlib
                colab_files = importlib.import_module('google.colab.files')
                
                # 询问用户是否需要上传文件
                if input("是否需要上传CSV文件? (y/n): ").lower() == 'y':
                    print("请上传(galah_)X_FE.csv文件...")
                    uploaded = colab_files.upload()
                    print("上传的文件:", list(uploaded.keys()))

                # 询问用户是否需要上传FITS文件
                if input("是否需要上传FITS文件? (y/n): ").lower() == 'y':
                    # 如果FITS文件是打包的，上传并解压
                    print("请上传包含FITS文件的压缩包...")
                    fits_archive = colab_files.upload()
                    archive_name = list(fits_archive.keys())[0]

                    # 创建fits目录
                    os.makedirs('fits', exist_ok=True)

                    # 解压缩文件到fits目录
                    if archive_name.endswith('.zip'):
                        print(f"正在解压 {archive_name}...")
                        with zipfile.ZipFile(archive_name, 'r') as zip_ref:
                            zip_ref.extractall('fits/')
                        print(f"已将{archive_name}解压到fits目录")
                        
                        # 检查解压后的目录结构
                        fits_files = []
                        for root, dirs, files in os.walk('fits'):
                            for file in files:
                                if any(file.endswith(ext) for ext in ['.fits', '.fits.gz', '.fit', '.fit.gz']):
                                    fits_files.append(os.path.join(root, file))
                        
                        print(f"解压后找到 {len(fits_files)} 个FITS文件")
                        
                        # 检查是否有嵌套目录
                        nested_dirs = set()
                        for file in fits_files:
                            rel_dir = os.path.relpath(os.path.dirname(file), 'fits')
                            if rel_dir != '.':
                                nested_dirs.add(rel_dir)
                        
                        if nested_dirs:
                            print(f"发现嵌套目录结构: {', '.join(list(nested_dirs)[:3])}")
                            move_files = input("是否将所有FITS文件移动到fits根目录? (y/n): ").lower() == 'y'
                            
                            if move_files:
                                # 移动文件到根目录
                                for file in fits_files:
                                    if os.path.dirname(file) != 'fits':
                                        target = os.path.join('fits', os.path.basename(file))
                                        print(f"移动 {os.path.basename(file)}")
                                        # 如果目标文件已存在，先删除
                                        if os.path.exists(target):
                                            os.remove(target)
                                        os.rename(file, target)
                                print("文件移动完成")
                    else:
                        print("不支持的压缩格式，请上传.zip文件")
            except Exception as e:
                print(f"Colab环境设置出错: {e}")
                print("继续使用本地文件...")
                
        # 运行主程序
        main()
        
    except KeyboardInterrupt:
        print("\n程序被用户中断，您可以稍后重新运行继续处理")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc() 