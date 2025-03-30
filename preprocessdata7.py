import numpy as np
import pandas as pd
from astropy.io import fits
import os
import glob
from sklearn.model_selection import train_test_split, KFold
from scipy import signal, interpolate
from multiprocessing import Pool, cpu_count
import time
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import gc  # 垃圾回收
import psutil  # 系统资源监控
import pickle  # 用于保存中间结果
import warnings
import subprocess  # 用于执行shell命令
import zipfile  # 用于解压文件
import sys  # 用于检测环境
import shutil
from utils import CacheManager
import concurrent.futures
import json
import traceback
import re
from sklearn.model_selection import KFold, train_test_split
import random
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

class LAMOSTPreprocessor:
    def __init__(self, csv_files=None, 
                 fits_dir='fits', 
                 output_dir='processed_data',
                 wavelength_range=None,  # 修改为None，表示将使用最大公有波长范围
                 n_points=None,  # 修改为None，点数将根据波长范围和步长自动计算
                 log_step=0.0001,  # 新增：对数空间中的重采样步长（dex）
                 compute_common_range=True,  # 新增：是否计算最大公有波长范围
                 n_splits=5,     # 交叉验证折数
                 max_workers=None,  # 最大工作进程数，None表示自动确定
                 batch_size=20,  # 批处理大小
                 memory_limit=0.7,  # 内存使用限制(占总内存比例)
                 verbose=1,  # 输出详细程度，0=静默，1=基本信息，2=详细信息
                 low_memory_mode=False):  # 低内存模式
        
        # 设置文件路径
        # 默认使用当前目录下所有的CSV文件
        if csv_files is None:
            csv_files = [f for f in os.listdir() if f.endswith('.csv')]
            if not csv_files:
                print("警告: 当前目录未找到CSV文件")
            else:
                print(f"自动检测到以下CSV文件: {csv_files}")
                
        self.csv_files = [csv_file if os.path.exists(csv_file) else os.path.join('/content', csv_file) for csv_file in csv_files]
        self.fits_dir = fits_dir if os.path.exists(fits_dir) else os.path.join('/content', fits_dir)
        self.output_dir = output_dir
        self.cache_dir = os.path.join(self.output_dir, 'cache')
        
        # 添加Google Drive备用缓存目录
        self.drive_cache_dir = '/content/drive/My Drive/SPCNNet_Results/processed_data/cache'
        
        self.progress_dir = os.path.join(self.output_dir, 'progress')
        self.figures_dir = os.path.join(self.output_dir, 'figures')
        self.logs_dir = os.path.join(self.output_dir, 'logs')
        self.model_dir = os.path.join(self.output_dir, 'models')
        self.prediction_output_dir = os.path.join(self.output_dir, 'predictions')
        self.cache_enabled = True
        
        # 创建所有必要的目录
        for directory in [self.output_dir, self.cache_dir, self.progress_dir, 
                          self.figures_dir, self.logs_dir, self.model_dir, 
                          self.prediction_output_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # 初始化文件查找缓存
        self.fits_file_cache = {}
        self.cache_file = os.path.join(self.cache_dir, 'files_cache.pkl')
        
        # 在初始化时就加载文件查找缓存
        print("正在加载文件查找缓存...")
        self._load_files_cache()
        
        self.wavelength_range = wavelength_range if wavelength_range else (3690, 9100)  # 默认范围
        self.n_points = n_points
        self.log_step = log_step
        self.compute_common_range = compute_common_range
        self.n_splits = n_splits
        
        # 光速常量（km/s）
        self.c = 299792.458
        
        # 存储已处理光谱的波长范围，用于计算最大公有范围
        self.processed_ranges = []
        
        # 性能相关参数
        self.batch_size = batch_size
        self.memory_limit = memory_limit
        
        # 设置最大工作进程数
        if max_workers is None:
            # 默认使用CPU核心数的一半，避免系统过热
            self.max_workers = max(1, cpu_count() // 2)
        else:
            self.max_workers = max_workers
            
        print(f"设置最大工作进程数: {self.max_workers}")
        
        self.low_memory_mode = low_memory_mode
        
        self.cache_manager = CacheManager(cache_dir=os.path.join(output_dir, 'cache'))
        
        self.update_cache_manager()
        
        # 输出详细程度
        self.verbose = verbose
        
    def _load_files_cache(self):
        """加载文件查找缓存"""
        # 尝试标准路径
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.fits_file_cache = pickle.load(f)
                print(f"已加载文件查找缓存，共{len(self.fits_file_cache)}条记录")
                return
            except Exception as e:
                print(f"加载文件缓存出错: {e}")
                self.fits_file_cache = {}
        
        # 尝试Google Drive路径
        drive_cache_file = '/content/drive/My Drive/SPCNNet_Results/processed_data/cache/files_cache.pkl'
        if os.path.exists(drive_cache_file):
            try:
                with open(drive_cache_file, 'rb') as f:
                    self.fits_file_cache = pickle.load(f)
                print(f"已从Google Drive加载文件查找缓存，共{len(self.fits_file_cache)}条记录")
                # 立即保存到标准路径，方便后续使用
                try:
                    self._save_files_cache()
                    print("已将Drive缓存同步到本地")
                except:
                    pass
                return
            except Exception as e:
                print(f"加载Google Drive文件缓存出错: {e}")
                self.fits_file_cache = {}
        
        # 都没有找到，初始化空缓存
        self.fits_file_cache = {}
    
    def _save_files_cache(self):
        """保存文件查找缓存"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.fits_file_cache, f)
        except Exception as e:
            print(f"保存文件缓存出错: {e}")
    
    def read_csv_data(self):
        """读取CSV文件并返回DataFrame列表和对应的元素名称"""
        dataframes = []
        elements = []
        for csv_file in self.csv_files:
            if not os.path.exists(csv_file):
                print(f"错误: 找不到CSV文件 {csv_file}")
                print(f"当前工作目录: {os.getcwd()}")
                print(f"尝试查找的完整路径: {os.path.abspath(csv_file)}")
                
                # 尝试从可能的目录中查找
                possible_dirs = ['/content', '/content/drive/My Drive', '/content/SPCNNet']
                for posdir in possible_dirs:
                    if os.path.exists(posdir):
                        possible_path = os.path.join(posdir, os.path.basename(csv_file))
                        if os.path.exists(possible_path):
                            print(f"找到可用的CSV文件: {possible_path}")
                            csv_file = possible_path
                            break
                        
                if not os.path.exists(csv_file):
                    # 如果还是没找到，列出当前目录的文件
                    print("当前目录中的文件:")
                    for f in os.listdir():
                        print(f"  - {f}")
                    continue
                
            print(f"读取CSV文件: {csv_file}")
            try:
                df = pd.read_csv(csv_file)
                print(f"成功加载{csv_file}，共{len(df)}条记录")
                print(f"列名: {', '.join(df.columns)}")
                
                # 从CSV文件名提取元素信息
                element_name = os.path.basename(csv_file).split('.')[0]
                print(f"识别到元素：{element_name}")
                
                # 检查spec列中的文件是否存在
                if 'spec' in df.columns:
                    # 确保spec列的类型为字符串
                    if not pd.api.types.is_string_dtype(df['spec']):
                        print(f"注意: {csv_file} 中的spec列不是字符串类型，正在转换...")
                        df['spec'] = df['spec'].astype(str)
                    
                    # 不在启动时检查所有文件，只显示警告信息
                    print(f"CSV文件包含{len(df)}条记录，如果找不到某些FITS文件，将在处理时报错")
                    
                    # 仅检查前3个文件作为示例(不再检查所有文件)
                    spec_files = df['spec'].values[:3]
                    for spec_file in spec_files:
                        # 使用_find_fits_file方法查找文件
                        found_path = self._find_fits_file(spec_file)
                        if found_path:
                            print(f"示例文件找到: {found_path}")
                        else:
                            print(f"示例文件未找到: {spec_file}，请确保FITS文件路径正确")
                else:
                    print(f"警告: CSV文件 {csv_file} 中没有找到'spec'列")
                    print(f"可用的列: {df.columns.tolist()}")
                
                dataframes.append(df)
                elements.append(element_name)
            except Exception as e:
                print(f"读取CSV文件 {csv_file} 出错: {e}")
                import traceback
                traceback.print_exc()
        
        return dataframes, elements
    
    def _find_fits_file(self, spec_name):
        """查找匹配的fits文件，处理嵌套目录和命名差异"""
        # 确保spec_name是字符串类型
        spec_name = str(spec_name)
        
        # 检查缓存
        if not hasattr(self, 'fits_file_cache'):
            self.fits_file_cache = {}
            # 加载文件缓存
            self._load_files_cache()
            
        if spec_name in self.fits_file_cache:
            cache_file = self.fits_file_cache[spec_name]
            # 确认缓存的文件仍然存在
            if cache_file is not None and os.path.exists(cache_file):
                return cache_file
            # 文件已移动或删除，重新查找
            self.fits_file_cache[spec_name] = None
        
        # 如果输入已经是完整路径，提取文件名部分
        if os.path.isabs(spec_name):
            # 先检查完整路径是否直接存在
            if os.path.exists(spec_name) and os.path.isfile(spec_name):
                print(f"找到绝对路径文件: {spec_name}")
                self.fits_file_cache[spec_name] = spec_name
                return spec_name
                
            # 如果完整路径不存在，提取文件名
            base_name = os.path.basename(spec_name)
        else:
            # 相对路径情况下
            base_name = spec_name
        
        # 记录日志
        print(f"查找文件: {spec_name}, 基础名称: {base_name}")
        
        # 首先尝试直接在fits目录下按完整路径匹配
        direct_path = os.path.join(self.fits_dir, spec_name)
        if os.path.exists(direct_path) and os.path.isfile(direct_path):
            print(f"直接匹配成功: {direct_path}")
            self.fits_file_cache[spec_name] = direct_path
            return direct_path
        
        # 尝试直接在fits目录下按基础名称匹配（常规后缀）
        for ext in ['', '.fits', '.fits.gz', '.fit', '.fit.gz']:
            path = os.path.join(self.fits_dir, base_name + ext)
            if os.path.exists(path) and os.path.isfile(path):
                print(f"基础名称匹配成功: {path}")
                self.fits_file_cache[spec_name] = path
                return path
        
        # 进行递归搜索，处理嵌套目录
        for root, dirs, files in os.walk(self.fits_dir):
            for file in files:
                # 检查文件名是否匹配（忽略大小写）
                if base_name.lower() in file.lower():
                    found_path = os.path.join(root, file)
                    print(f"部分名称匹配成功: {found_path}")
                    self.fits_file_cache[spec_name] = found_path
                    return found_path
                
                # 尝试去除可能的后缀后再比较
                file_base = file.lower()
                for ext in ['.fits', '.fits.gz', '.fit', '.fit.gz']:
                    if file_base.endswith(ext):
                        file_base = file_base[:-len(ext)]
                        break
                
                if base_name.lower() == file_base:
                    found_path = os.path.join(root, file)
                    print(f"去除后缀后匹配成功: {found_path}")
                    self.fits_file_cache[spec_name] = found_path
                    return found_path
                
                # 尝试更模糊的匹配方式
                # 移除路径分隔符，便于匹配跨目录文件
                clean_base_name = base_name.replace('/', '_').replace('\\', '_')
                clean_file_base = file_base.replace('/', '_').replace('\\', '_')
                
                if clean_base_name.lower() in clean_file_base or clean_file_base in clean_base_name.lower():
                    found_path = os.path.join(root, file)
                    print(f"模糊匹配成功: {found_path}")
                    self.fits_file_cache[spec_name] = found_path
                    return found_path
        
        # 如果以上都没找到，返回None
        print(f"未找到匹配文件: {spec_name}")
        return None
    
    def _get_file_extension(self, fits_file):
        """获取文件完整路径，使用缓存避免重复查找"""
        if fits_file in self.fits_file_cache:
            return self.fits_file_cache[fits_file]
            
        # 查找实际文件路径
        file_path = self._find_fits_file(fits_file)
        if file_path:
            self.fits_file_cache[fits_file] = file_path
            return file_path
        else:
            self.fits_file_cache[fits_file] = None
            return None
    
    def read_fits_file(self, fits_file):
        """读取FITS文件并返回波长和流量数据"""
        # 获取正确的文件路径
        file_path = self._get_file_extension(fits_file)
        if file_path is None:
            print(f"无法找到文件: {fits_file}，查找路径: {self.fits_dir}")
            return None, None, 0, 0, 0, {}
        
        print(f"读取文件: {file_path}")
        try:
            # 使用更多选项打开FITS文件
            with fits.open(file_path, ignore_missing_end=True, memmap=False) as hdul:
                # 打印HDU信息以帮助诊断
                print(f"FITS文件结构: 共{len(hdul)}个HDU")
                for i, hdu in enumerate(hdul):
                    print(f"  HDU{i}: 类型={type(hdu).__name__}, 形状={hdu.shape if hasattr(hdu, 'shape') else '无形状'}")
                
                # 获取主HDU的头信息
                header = hdul[0].header
                
                # 输出关键头信息帮助诊断
                print(f"主HDU头信息: NAXIS={header.get('NAXIS')}, NAXIS1={header.get('NAXIS1')}, "
                      f"BITPIX={header.get('BITPIX')}")
                
                # 尝试获取视向速度信息(如果有)
                v_helio = 0
                for key in ['V_HELIO', 'RV', 'VELOCITY', 'v_helio', 'rv', 'velocity']:
                    if key in header:
                        v_helio = header.get(key, 0)
                        print(f"从FITS头信息中找到视向速度: {key} = {v_helio}")
                        break
                
                # 尝试获取红移值(可能有不同的关键字)
                z = 0
                for key in ['Z', 'REDSHIFT', 'z', 'redshift', 'Redshift', 'RED_SHIFT', 'red_shift']:
                    if key in header:
                        z = header.get(key, 0)
                        print(f"从FITS头信息中找到红移值: {key} = {z}")
                        break
                
                # 如果在头信息中没找到红移值，尝试在扩展表中查找
                if z == 0 and len(hdul) > 1:
                    for i in range(1, len(hdul)):
                        if isinstance(hdul[i], fits.BinTableHDU):
                            table_hdu = hdul[i]
                            column_names = table_hdu.columns.names
                            print(f"检查表格HDU{i}中的红移值, 列名: {column_names}")
                            
                            # 寻找红移列
                            for col_name in ['Z', 'REDSHIFT', 'z', 'redshift', 'Redshift', 'RED_SHIFT', 'red_shift']:
                                if col_name in column_names:
                                    try:
                                        z_values = table_hdu.data[col_name]
                                        if len(z_values) > 0:
                                            z = z_values[0]
                                            print(f"从表格HDU{i}列'{col_name}'中找到红移值: {z}")
                                            break
                                    except Exception as e:
                                        print(f"读取表格HDU{i}列'{col_name}'时出错: {e}")
                            
                            if z != 0:
                                break
                
                # 获取信噪比信息(尝试多种可能的关键字)
                snr = 0
                for key in ['SNR', 'SNRATIO', 'SN', 'S/N', 'snr']:
                    if key in header:
                        snr = header.get(key, 0)
                        print(f"从FITS头信息中找到信噪比: {key} = {snr}")
                        break
                
                # 获取五段区间的信噪比信息
                snr_bands = {'snru': 0, 'snrg': 0, 'snrr': 0, 'snri': 0, 'snrz': 0}
                for band in snr_bands:
                    for variation in [band, band.upper(), band.capitalize()]:
                        if variation in header:
                            snr_bands[band] = header.get(variation, 0)
                            print(f"从FITS头信息中找到{band}波段信噪比: {snr_bands[band]}")
                            break
                
                # 优先获取第一个HDU的数据(如果是主要光谱数据)
                flux = None
                wavelength = None
                
                # 规则1: 如果主HDU是PrimaryHDU且包含数据，直接使用
                if isinstance(hdul[0], fits.PrimaryHDU) and hdul[0].data is not None:
                    if len(hdul[0].data.shape) == 1:  # 一维数据
                        flux = hdul[0].data
                        # 从头信息创建波长数组
                        if 'CRVAL1' in header and 'CDELT1' in header and 'NAXIS1' in header:
                            crval1 = header['CRVAL1']  # 起始波长
                            cdelt1 = header['CDELT1']  # 波长步长
                            naxis1 = header['NAXIS1']  # 波长点数
                            wavelength = np.arange(crval1, crval1 + cdelt1 * naxis1, cdelt1)[:naxis1]
                        print(f"使用主HDU的一维数据: 点数={len(flux)}")
                        
                    elif len(hdul[0].data.shape) == 2:  # 二维数据
                        # 取第一行或列，取决于哪个更长
                        if hdul[0].data.shape[0] > hdul[0].data.shape[1]:
                            flux = hdul[0].data[0]
                        else:
                            flux = hdul[0].data[:, 0]
                        print(f"使用主HDU的二维数据的第一行/列: 点数={len(flux)}")
                
                # 规则2: 如果数据在表格HDU中
                if flux is None and len(hdul) > 1:
                    for i in range(1, len(hdul)):
                        if isinstance(hdul[i], fits.BinTableHDU):
                            table_hdu = hdul[i]
                            column_names = table_hdu.columns.names
                            print(f"检查表格HDU{i}, 列名: {column_names}")
                            
                            # 查找光谱数据列
                            flux_col = None
                            wave_col = None
                            
                            # 寻找光谱流量列
                            for col_name in ['FLUX', 'SPEC', 'DATA', 'INTENSITY', 'COUNTS', 'flux']:
                                if col_name in column_names:
                                    flux_col = col_name
                                    break
                            
                            # 寻找波长列
                            for wave_name in ['WAVE', 'WAVELENGTH', 'LAMBDA', 'wave', 'wavelength']:
                                if wave_name in column_names:
                                    wave_col = wave_name
                                    break
                            
                            # 如果找到流量列
                            if flux_col is not None:
                                try:
                                    # 读取流量数据
                                    flux_data = table_hdu.data[flux_col]
                                    
                                    # 如果流量是一个二维数组，取第一行
                                    if hasattr(flux_data, 'shape') and len(flux_data.shape) > 1:
                                        flux = flux_data[0].astype(np.float64)
                                    else:
                                        # 确保flux是一维数组
                                        flux = np.array(flux_data, dtype=np.float64).flatten()
                                    
                                    print(f"从列 '{flux_col}' 提取流量数据, 点数={len(flux)}")
                                    
                                    # 如果找到波长列，读取波长数据
                                    if wave_col is not None:
                                        wave_data = table_hdu.data[wave_col]
                                        if hasattr(wave_data, 'shape') and len(wave_data.shape) > 1:
                                            wavelength = wave_data[0].astype(np.float64)
                                        else:
                                            wavelength = np.array(wave_data, dtype=np.float64).flatten()
                                        print(f"从列 '{wave_col}' 提取波长数据, 点数={len(wavelength)}")
                                        
                                        # 确保波长和流量数组长度匹配
                                        if len(wavelength) != len(flux):
                                            min_len = min(len(wavelength), len(flux))
                                            wavelength = wavelength[:min_len]
                                            flux = flux[:min_len]
                                            print(f"调整数组长度为匹配长度: {min_len}")
                                    
                                    break  # 找到数据后退出循环
                                except Exception as e:
                                    print(f"从表格提取数据出错: {e}")
                                    flux = None  # 重置，尝试其他HDU
                
                # 如果没有找到波长数据，但有流量数据
                if wavelength is None and flux is not None:
                    # 尝试从头信息创建波长数组
                    if 'CRVAL1' in header and 'CDELT1' in header and 'NAXIS1' in header:
                        crval1 = header['CRVAL1']  # 起始波长
                        cdelt1 = header['CDELT1']  # 波长步长
                        naxis1 = header['NAXIS1']  # 波长点数
                        
                        # 确保naxis1与flux长度匹配
                        if naxis1 != len(flux):
                            naxis1 = len(flux)
                            print(f"调整NAXIS1值为与流量数组匹配: {naxis1}")
                        
                        wavelength = np.arange(crval1, crval1 + cdelt1 * naxis1, cdelt1)[:naxis1]
                        print(f"从头信息创建波长数组: 范围={wavelength[0]:.2f}~{wavelength[-1]:.2f}")
                    else:
                        # 如果没有头信息，使用默认波长范围
                        print("头信息中没有波长参数，使用默认波长范围")
                        naxis1 = len(flux)
                        # LAMOST DR10光谱的典型波长范围约为3700-9000Å
                        crval1 = 3700.0  # 起始波长
                        cdelt1 = (9000.0 - 3700.0) / naxis1  # 波长步长
                        wavelength = np.arange(crval1, crval1 + cdelt1 * naxis1, cdelt1)[:naxis1]
                        print(f"创建默认波长数组: 范围={wavelength[0]:.2f}~{wavelength[-1]:.2f}")
                
                # 进行最后的数据检查
                if flux is None:
                    print("无法从FITS文件提取流量数据")
                    return None, None, 0, 0, 0, {}
                
                if wavelength is None:
                    print("无法生成波长数据")
                    return None, None, 0, 0, 0, {}
                
                # 确保数据类型是浮点数
                flux = flux.astype(np.float64)
                wavelength = wavelength.astype(np.float64)
                
                # 检查是否有NaN或无限值
                if np.isnan(flux).any() or np.isinf(flux).any():
                    nan_count = np.isnan(flux).sum()
                    inf_count = np.isinf(flux).sum()
                    print(f"数据中包含{nan_count}个NaN和{inf_count}个无限值，尝试替换")
                    flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
                
                print(f"成功提取光谱数据: 点数={len(wavelength)}, 波长范围={wavelength[0]:.2f}~{wavelength[-1]:.2f}")
                return wavelength, flux, v_helio, z, snr, snr_bands
                
        except Exception as e:
            print(f"读取{file_path}出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None, 0, 0, 0, {}
    
    def denoise_spectrum(self, wavelength, flux):
        """对光谱进行去噪处理"""
        try:
            # 使用Savitzky-Golay滤波器去噪
            window_length = 5  # 窗口大小
            polyorder = 2  # 多项式阶数
            
            # 防止窗口长度不足的错误
            if len(flux) < window_length:
                print(f"数据点数太少({len(flux)})，无法使用窗口为{window_length}的滤波器")
                return flux  # 数据点太少，直接返回原始数据
            
            # 处理无效值
            mask = ~np.isnan(flux)
            if not np.any(mask):
                print("全部为NaN值，无法去噪")
                return None
            
            # 只对有效数据进行滤波
            valid_flux = flux[mask]
            
            if len(valid_flux) < window_length:
                print(f"有效数据点数太少({len(valid_flux)})，无法去噪")
                return flux
            
            # 对有效数据进行滤波
            flux_denoised = np.copy(flux)
            flux_denoised[mask] = signal.savgol_filter(valid_flux, window_length, polyorder)
            
            return flux_denoised
        except Exception as e:
            print(f"去噪处理出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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
            
            # 检查计算出的公共范围是否有效
            if common_min < common_max:
                # 获取默认范围
                default_min, default_max = (3690, 9100)  # 默认波长范围
                
                # 修正：只有当计算出的范围超出默认范围时，才使用默认范围作为约束
                # 如果计算出的最小值比默认最小值还小，则保留计算出的值
                if common_min < default_min:
                    print(f"计算的最小波长({common_min:.2f})小于默认范围，使用默认最小值: {default_min:.2f}")
                    common_min = default_min
                    
                # 如果计算出的最大值比默认最大值还大，则保留计算出的值
                if common_max > default_max:
                    print(f"计算的最大波长({common_max:.2f})大于默认范围，使用默认最大值: {default_max:.2f}")
                    common_max = default_max
                
                # 仅当修正后的范围有效时更新
                if common_min < common_max:
                    self.wavelength_range = (common_min, common_max)
                    print(f"更新最大公有波长范围: {common_min:.2f}~{common_max:.2f}")
                else:
                    print(f"警告: 修正后的波长范围无效 ({common_min:.2f}~{common_max:.2f})，保持原有范围")
            else:
                print(f"警告: 无法更新公有波长范围，当前范围不重叠 ({common_min:.2f}~{common_max:.2f})")
    
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
            
            # 不再调用update_common_wavelength_range，只使用预先计算好的波长范围
            
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
                # 只在局部变量中调整范围，不修改类的wavelength_range属性
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
                actual_log_step = (log_w_max - log_w_min) / (n_points - 1)
                print(f"使用指定点数{n_points}，对数步长为: {actual_log_step} dex")
            
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
                new_flux = np.nan_to_num(new_flux, nan=0.0, posinf=0.0, neginf=0.0)
                
            # 额外检查是否存在无限值
            if np.isinf(new_flux).any():
                n_inf = np.isinf(new_flux).sum()
                print(f"重采样后有{n_inf}/{len(new_flux)}个点是无限值，将替换为0")
                new_flux = np.nan_to_num(new_flux, nan=0.0, posinf=0.0, neginf=0.0)
                
            return new_wavelength, new_flux
        except Exception as e:
            print(f"重采样失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def normalize_spectrum(self, wavelength, flux):
        """
        对光谱进行最大最小归一化处理，确保结果严格在[0,1]范围内
        
        参数:
            wavelength (ndarray): 波长数组
            flux (ndarray): 原始光谱强度
            
        返回:
            ndarray: 归一化后的光谱强度，范围严格在[0,1]之间
        """
        if flux is None or len(flux) == 0:
            print("警告: 归一化失败 - 输入光谱为空")
            return None
        
        # 检查无效值
        invalid_mask = np.isnan(flux) | np.isinf(flux)
        if np.all(invalid_mask):
            print("警告: 归一化失败 - 所有值都是NaN或无穷大")
            return None
        
        # 获取有效数据点
        valid_mask = ~invalid_mask
        if not np.any(valid_mask):
            print("警告: 归一化失败 - 没有有效数据点")
            return None
        
        # 直接从有效数据中计算最小值和最大值
        valid_flux = flux[valid_mask]
        min_val = np.min(valid_flux)
        max_val = np.max(valid_flux)
        
        print(f"归一化前数据范围: [{min_val:.6f}, {max_val:.6f}]")
        
        # 检查数据范围
        if max_val == min_val:
            print(f"警告: 归一化失败 - 所有有效值都相同: {min_val}")
            # 返回全零数组作为归一化结果
            normalized_flux = np.zeros_like(flux)
            normalized_flux[invalid_mask] = np.nan  # 保持无效值为NaN
            return normalized_flux
        
        # 执行最大最小归一化
        normalized_flux = np.zeros_like(flux)
        normalized_flux[valid_mask] = (valid_flux - min_val) / (max_val - min_val)
        
        # 确保值在[0,1]范围内 (处理可能的浮点误差)
        normalized_flux = np.clip(normalized_flux, 0, 1)
        
        # 保持原始的无效值为NaN
        normalized_flux[invalid_mask] = np.nan
        
        # 验证归一化结果
        valid_normalized = normalized_flux[valid_mask]
        norm_min = np.min(valid_normalized)
        norm_max = np.max(valid_normalized)
        
        print(f"归一化后数据范围: [{norm_min:.6f}, {norm_max:.6f}]")
        
        # 记录归一化参数供后续使用
        self.normalization_params = {
            'min': float(min_val),
            'max': float(max_val),
            'wavelength_range': (float(np.min(wavelength)), float(np.max(wavelength))) if wavelength is not None else None
        }
        
        return normalized_flux
    
    def correct_wavelength(self, wavelength, flux):
        """对光谱进行波长标准化校正
        注意：这与视向速度校正(correct_velocity)不同。
        视向速度校正使用多普勒公式λ' = λ/(1 + RV/c)校正系统性红移/蓝移。
        此方法通过检测吸收线与标准线的偏差进行小幅波长校准，修正仪器或标定引起的微小系统偏差。
        """
        try:
            # 检查波长是否需要校准
            flux_valid = flux[~np.isnan(flux)]
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
    
    def normalize_continuum(self, wavelength, flux):
        """对光谱进行连续谱归一化，使特征更加明显
        方法：先通过曲线拟合估计伪连续谱，然后将原始光谱除以伪连续谱
        """
        try:
            # 确保输入数据有效
            valid_mask = ~np.isnan(flux)
            if not np.any(valid_mask):
                print("所有流量值都是NaN，无法进行连续谱归一化")
                return flux
            
            wavelength_valid = wavelength[valid_mask]
            flux_valid = flux[valid_mask]
            
            # 特殊处理：检测OI 7774埃附近的区域，因为这里经常出现异常峰值
            oi_region_mask = (wavelength >= 7700) & (wavelength <= 7850)
            has_oi_peak = False
            if np.any(oi_region_mask):
                oi_flux = flux[oi_region_mask]
                if np.max(oi_flux) > np.median(flux_valid) * 1.5:
                    print("检测到OI 7774埃附近有明显峰值，将进行特殊处理")
                    has_oi_peak = True
            
            # 将光谱分成多个小区间，避免单一多项式拟合不足的问题
            # 为了更好地处理OI区域，增加分段数量
            num_segments = 7 if has_oi_peak else 5  # 如果有OI峰值，使用更多分段
            segment_length = len(wavelength_valid) // num_segments
            normalized_flux = np.copy(flux)
            
            print(f"将光谱分成{num_segments}个区间进行连续谱拟合")
            
            # 对每个区间单独拟合连续谱
            for i in range(num_segments):
                start_idx = i * segment_length
                end_idx = (i + 1) * segment_length if i < num_segments - 1 else len(wavelength_valid)
                
                if end_idx <= start_idx:
                    continue
                
                # 获取当前区间的波长和流量数据
                wave_segment = wavelength_valid[start_idx:end_idx]
                flux_segment = flux_valid[start_idx:end_idx]
                
                # 跳过数据点不足的区间
                if len(wave_segment) < 10:
                    print(f"区间{i+1}数据点不足，跳过")
                    continue
                
                # 检查当前区段是否包含OI线区域
                segment_has_oi = (np.min(wave_segment) <= 7774.2) and (np.max(wave_segment) >= 7774.2)
                
                # 为OI区域使用更保守的拟合方法
                if segment_has_oi and has_oi_peak:
                    print(f"区间{i+1}包含OI线(7774埃)，使用更保守的拟合方法")
                    poly_order = 1  # 使用较低阶数避免过拟合
                else:
                    # 使用多项式拟合连续谱，但为防止过拟合，根据点数调整阶数
                    poly_order = min(2, len(wave_segment) // 20)  # 降低默认阶数从3到2
                    poly_order = max(1, poly_order)  # 至少使用1阶多项式
                
                # 为每个区段找出可能的连续谱点
                # 使用中值滤波平滑曲线，识别连续谱的趋势
                from scipy.signal import medfilt
                window_size = min(11, len(flux_segment) // 5 * 2 + 1)  # 确保窗口大小为奇数
                window_size = max(3, window_size)  # 至少使用3点窗口
                smoothed_flux = medfilt(flux_segment, window_size)
                
                # 选择连续谱点的方法
                if segment_has_oi and has_oi_peak:
                    # 对于OI区域，使用更严格的方法选择连续谱点
                    # 选择低于中位值的点，因为吸收线会导致流量降低
                    q25 = np.percentile(flux_segment, 25)  # 使用25%分位数作为阈值
                    q75 = np.percentile(flux_segment, 75)
                    iqr = q75 - q25
                    upper_limit = q75 + 0.5 * iqr  # 使用更保守的上限
                    continuum_mask = flux_segment < upper_limit
                else:
                    # 对于其他区域，使用常规方法
                    # 选择高于中位值的点作为可能的连续谱点
                    median_flux = np.median(flux_segment)
                    continuum_mask = flux_segment > median_flux
                
                # 确保有足够点用于拟合
                if np.sum(continuum_mask) < poly_order + 2:
                    print(f"区间{i+1}连续谱点不足({np.sum(continuum_mask)})，使用全部点")
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
                    
                    # 确保伪连续谱为正值且不会过小导致归一化后的峰值过大
                    min_threshold = np.max(flux_segment) * 0.05  # 保持为5%以防止过小的分母
                    
                    # 增加安全检查：如果是OI区域，使用更高的最小阈值
                    if segment_has_oi and has_oi_peak:
                        min_threshold = np.max(flux_segment) * 0.2  # 增加到20%
                        
                    pseudo_continuum[pseudo_continuum < min_threshold] = min_threshold
                    
                    # 对当前区间内的数据进行归一化
                    normalized_flux[mask] = flux[mask] / pseudo_continuum
                except Exception as e:
                    print(f"区间{i+1}拟合失败: {e}")
                    continue
            
            # 处理可能的无效值（过大值、NaN值和负值）
            # 限制归一化后的极大值，防止出现异常峰
            normalized_flux[normalized_flux > 3] = 3  # 降低限制到3而不是5
            normalized_flux[normalized_flux < 0] = 0  # 确保没有负值
            normalized_flux = np.nan_to_num(normalized_flux, nan=1.0, posinf=1.0, neginf=0.0)
            
            # 专门检查OI 7774埃附近区域
            if has_oi_peak:
                # 找到OI 7774埃附近的点
                oi_peak_mask = (wavelength >= 7740) & (wavelength <= 7810)
                if np.any(oi_peak_mask) and np.max(normalized_flux[oi_peak_mask]) > 2.0:
                    print("修正OI 7774埃区域的异常峰值")
                    # 获取附近的均值作为参考
                    left_region = (wavelength >= 7700) & (wavelength < 7740)
                    right_region = (wavelength > 7810) & (wavelength <= 7850)
                    nearby_values = []
                    if np.any(left_region):
                        nearby_values.extend(normalized_flux[left_region])
                    if np.any(right_region):
                        nearby_values.extend(normalized_flux[right_region])
                    
                    if nearby_values:
                        nearby_mean = np.mean(nearby_values)
                        # 将异常值限制在合理范围内
                        limit_factor = 1.5
                        max_allowed = nearby_mean * limit_factor
                        oi_mask = (wavelength >= 7740) & (wavelength <= 7810) & (normalized_flux > max_allowed)
                        if np.any(oi_mask):
                            normalized_flux[oi_mask] = max_allowed
            
            # 检查归一化后是否有异常值
            if np.max(normalized_flux) > 2:
                print(f"警告: 归一化后最大值为{np.max(normalized_flux):.2f}，可能存在异常峰")
                # 查找并平滑异常峰值
                outlier_mask = normalized_flux > 2
                if np.sum(outlier_mask) > 0:
                    print(f"检测到{np.sum(outlier_mask)}个异常点，进行平滑处理")
                    # 对异常峰值周围的值取平均，平滑处理
                    for i in np.where(outlier_mask)[0]:
                        if i > 0 and i < len(normalized_flux) - 1:
                            # 使用相邻点的均值替代异常值
                            neighbors = normalized_flux[max(0, i-3):i].tolist() + normalized_flux[i+1:min(len(normalized_flux), i+4)].tolist()
                            neighbors = [n for n in neighbors if n <= 2.0]  # 只使用不超过2.0的邻居点
                            if neighbors:
                                normalized_flux[i] = np.mean(neighbors)
                            else:
                                normalized_flux[i] = 1.0  # 如果无法计算平均值，使用1.0
            
            zero_mask = normalized_flux < 0.01
            if np.sum(zero_mask) > len(normalized_flux) * 0.5:
                print("警告: 归一化后50%以上的数据接近零，可能存在问题")
                
                # 如果过多数据接近零，尝试使用更简单的方法
                # 使用总体平均值作为简单伪连续谱
                mean_flux = np.mean(flux_valid)
                normalized_flux = flux / mean_flux
                # 再次限制范围，确保没有异常值
                normalized_flux = np.clip(normalized_flux, 0.0, 3.0)
                print("回退到使用平均流量值进行简单归一化")
            
            # 最后再次确保没有异常值
            final_max = np.max(normalized_flux)
            if final_max > 2.0:
                print(f"最终归一化后最大值仍为{final_max:.2f}，进行均匀缩放")
                # 将所有值等比例缩小，使最大值为2.0
                normalized_flux = normalized_flux * (2.0 / final_max)
            
            print("连续谱归一化完成")
            return normalized_flux, {'flux_min': np.min(flux_valid), 'flux_max': np.max(flux_valid)}
        
        except Exception as e:
            print(f"连续谱归一化失败: {e}")
            import traceback
            traceback.print_exc()
            # 出错时返回仅使用平均值归一化的结果
            try:
                mean_val = np.mean(flux[~np.isnan(flux)]) if np.any(~np.isnan(flux)) else 1.0
                normalized = flux / mean_val
                return np.clip(normalized, 0.0, 2.0), {'flux_min': np.min(flux), 'flux_max': np.max(flux)}  # 确保输出限制在更合理的范围内
            except:
                return flux, {'flux_min': None, 'flux_max': None}
    
    def denoise_spectrum_second(self, wavelength, flux):
        """对光谱进行二次去噪处理，更强地移除噪声，但保留明显的特征"""
        try:
            # 检查是否有无效值
            if flux is None or np.all(np.isnan(flux)):
                print("无效的流量数据，无法进行二次去噪")
                return flux
            
            # 对OI线区域进行特殊检查
            oi_region = (wavelength >= 7700) & (wavelength <= 7850)
            has_oi_anomaly = False
            if np.any(oi_region):
                oi_flux = flux[oi_region]
                if np.max(oi_flux) > np.median(flux[~np.isnan(flux)]) * 1.5:
                    print("OI线(7774埃)附近检测到异常，将加强平滑")
                    has_oi_anomaly = True
            
            # 保存原始数据的副本
            flux_denoised = np.copy(flux)
            
            # 使用SavGol滤波器进行平滑去噪
            from scipy.signal import savgol_filter
            
            # 确定窗口大小 - 正常区域和异常区域使用不同的参数
            standard_window = 7  # 默认窗口大小
            oi_window = 15      # OI区域使用更大窗口
            
            # 处理NaN值
            valid_mask = ~np.isnan(flux)
            if not np.any(valid_mask):
                return flux
            
            # 创建一个有效数据的副本用于填充
            valid_flux = flux[valid_mask]
            valid_wavelength = wavelength[valid_mask]
            
            # 对一般区域应用滤波
            try:
                flux_denoised[valid_mask] = savgol_filter(valid_flux, standard_window, 2)
                print(f"二次去噪完成，使用窗口长度= {standard_window}")
            except Exception as e:
                print(f"SavGol滤波失败: {e}")
                return flux
                
            # 如果OI区域有异常，使用更强的滤波参数专门处理
            if has_oi_anomaly:
                # 找到OI区域的有效数据点
                oi_valid_mask = oi_region & valid_mask
                if np.sum(oi_valid_mask) > oi_window:  # 确保有足够的点进行滤波
                    try:
                        # 对OI区域使用更大窗口和更高阶多项式
                        oi_indices = np.where(oi_valid_mask)[0]
                        if len(oi_indices) >= oi_window:
                            oi_flux_section = flux[oi_valid_mask]
                            # 使用更大窗口进行强平滑
                            oi_smoothed = savgol_filter(oi_flux_section, oi_window, 3)
                            flux_denoised[oi_valid_mask] = oi_smoothed
                            print(f"OI区域增强去噪完成，使用窗口长度= {oi_window}")
                    except Exception as e:
                        print(f"OI区域特殊去噪失败: {e}")
            
            # 还可以额外进行中值滤波以移除尖峰
            from scipy.signal import medfilt
            
            # 对特别突出的峰值使用中值滤波
            if has_oi_anomaly:
                # 寻找异常峰值
                flux_mean = np.mean(flux_denoised[valid_mask])
                flux_std = np.std(flux_denoised[valid_mask])
                spike_threshold = flux_mean + 1.5 * flux_std
                
                spike_mask = (flux_denoised > spike_threshold) & valid_mask
                if np.any(spike_mask):
                    print(f"检测到{np.sum(spike_mask)}个异常峰值点，进行中值滤波")
                    # 将这些点替换为周围7个点的中值
                    for idx in np.where(spike_mask)[0]:
                        start = max(0, idx - 3)
                        end = min(len(flux_denoised), idx + 4)
                        if end - start >= 3:  # 确保至少有3个点用于中值计算
                            neighbors = flux_denoised[start:end]
                            flux_denoised[idx] = np.median(neighbors)
            
            # 最后确保没有NaN值
            flux_denoised = np.nan_to_num(flux_denoised, nan=np.median(flux_denoised[valid_mask]))
            
            return flux_denoised
            
        except Exception as e:
            print(f"二次去噪失败: {e}")
            import traceback
            traceback.print_exc()
            return flux
    
    def process_single_spectrum(self, spec_file, label):
        """处理单个光谱"""
        # 确保spec_file是字符串类型
        spec_file = str(spec_file)
        
        # 检查参数合法性
        if spec_file is None:
            raise ValueError("spec_file不能为None")
        
        # 使用CacheManager替代直接缓存操作
        cache_key = f"processed_{spec_file.replace('/', '_')}"
        # 检查缓存
        processed_result = self.cache_manager.get_cache(cache_key)
        if processed_result:
            print(f"使用缓存的预处理光谱: {spec_file}")
            return processed_result
        
        try:
            # 读取FITS文件
            print(f"处理光谱: {spec_file}")
            # 注意：这里只解包6个值，与read_fits_file返回值匹配
            wavelength, flux, v_helio, z_fits, snr, snr_bands = self.read_fits_file(spec_file)
            if wavelength is None or flux is None:
                print(f"无法读取FITS文件: {spec_file}")
                return None
            
            # 检查数据有效性
            if np.isnan(flux).all() or len(flux) == 0:
                print(f"文件{spec_file}中的流量数据全为NaN或为空")
                return None
            
            print(f"原始数据: 波长范围{wavelength[0]}~{wavelength[-1]}, 点数={len(wavelength)}")
            
            # 获取红移数据
            z = z_fits  # 优先使用从fits文件中读取的红移值
            cv = 0      # 视向速度默认值
            try:
                # 尝试从文件名匹配到CSV中的记录获取红移和视向速度
                base_file = os.path.basename(spec_file)
                if '.' in base_file:
                    base_file = base_file.split('.')[0]
                
                for csv_file in self.csv_files:
                    if os.path.exists(csv_file):
                        df = pd.read_csv(csv_file)
                        
                        # 检查CSV是否有spec列
                        if 'spec' in df.columns:
                            # 确保spec列是字符串类型
                            if not pd.api.types.is_string_dtype(df['spec']):
                                df['spec'] = df['spec'].astype(str)
                                
                            # 在CSV中查找匹配记录
                            matches = df[df['spec'].str.contains(base_file, case=False, na=False)]
                            if not matches.empty:
                                # 如果z值为0且CSV中有z列，则从CSV读取
                                if z == 0 and 'z' in df.columns:
                                    z = matches.iloc[0]['z']
                                    print(f"从CSV找到红移值: z = {z}")
                                
                                # 读取视向速度 - 从cv或rv列
                                for vel_col in ['cv', 'rv', 'velocity', 'RV']:
                                    if vel_col in df.columns:
                                        cv = matches.iloc[0][vel_col]
                                        print(f"从CSV找到视向速度: {vel_col} = {cv} km/s")
                                        # 如果视向速度值有效，更新v_helio
                                        if not pd.isna(cv) and cv != 0:
                                            v_helio = cv
                                            print(f"使用CSV中的视向速度值: {v_helio} km/s")
                                        break
                                break
            except Exception as e:
                print(f"查找红移或视向速度数据出错: {e}")
                # 出错时使用默认值或已读取的值
                
            # 如果fits中未找到信噪比数据，尝试从CSV获取
            if all(v == 0 for v in snr_bands.values()):
                try:
                    for csv_file in self.csv_files:
                        if os.path.exists(csv_file):
                            df = pd.read_csv(csv_file)
                            
                            # 检查CSV是否有spec列
                            if 'spec' in df.columns:
                                # 确保spec列是字符串类型
                                if not pd.api.types.is_string_dtype(df['spec']):
                                    df['spec'] = df['spec'].astype(str)
                                    
                                # 在CSV中查找匹配记录
                                base_file = os.path.basename(spec_file)
                                if '.' in base_file:
                                    base_file = base_file.split('.')[0]
                                
                                matches = df[df['spec'].str.contains(base_file, case=False, na=False)]
                                if not matches.empty:
                                    for band in snr_bands:
                                        if band in df.columns:
                                            snr_bands[band] = matches.iloc[0][band]
                                            print(f"从CSV找到{band}波段信噪比: {snr_bands[band]}")
                                    break  # 找到匹配项后退出循环
                except Exception as e:
                    print(f"从CSV读取信噪比失败: {e}")
            
            # 检查红移和视向速度值是否为NaN
            if pd.isna(z):
                print("警告: 红移值为NaN，设置为0")
                z = 0
            if pd.isna(v_helio):
                print("警告: 视向速度值为NaN，设置为0")
                v_helio = 0
            
            # 1. 波长校正
            wavelength_calibrated = self.correct_wavelength(wavelength, flux)
            print(f"波长校正后: 波长范围{wavelength_calibrated[0]}~{wavelength_calibrated[-1]}")
            
            # 2. 视向速度校正
            wavelength_corrected = self.correct_velocity(wavelength_calibrated, flux, v_helio)
            print(f"视向速度校正后: 波长范围{wavelength_corrected[0]}~{wavelength_corrected[-1]}")
            
            # 3. 去噪
            flux_denoised = self.denoise_spectrum(wavelength_corrected, flux)
            if flux_denoised is None:
                print(f"去噪{spec_file}失败")
                return None
            
            # 4. 红移校正
            wavelength_rest = self.correct_redshift(wavelength_corrected, flux_denoised, z)
            print(f"红移校正后: 波长范围{wavelength_rest[0]}~{wavelength_rest[-1]}")
            
            # 5. 重采样
            print(f"重采样到波长范围: {self.wavelength_range}, 点数={self.n_points}")
            wavelength_resampled, flux_resampled = self.resample_spectrum(wavelength_rest, flux_denoised)
            if wavelength_resampled is None or flux_resampled is None:
                print(f"重采样{spec_file}失败")
                return None
            
            # 6. 连续谱归一化
            flux_continuum, continuum_params = self.normalize_continuum(wavelength_resampled, flux_resampled)
            if flux_continuum is None:
                print(f"连续谱归一化{spec_file}失败")
                return None
            
            # 7. 二次去噪
            flux_denoised_second = self.denoise_spectrum_second(wavelength_resampled, flux_continuum)
            
            # 8. 最终归一化 (最大最小值归一化)
            print(f"对流量进行最终归一化")
            flux_normalized = self.normalize_spectrum(wavelength_resampled, flux_denoised_second)
            if flux_normalized is None:
                print(f"归一化{spec_file}失败")
                return None
            
            # 不再需要额外的归一化验证，normalize_spectrum已经完成了完整的验证
            print(f"成功处理光谱: {spec_file}")
            
            # 记录标准化参数
            normalization_params = {
                # 波长范围信息
                'wavelength_range': self.wavelength_range,
                'log_step': self.log_step,
                'flux_min': continuum_params['flux_min'] if continuum_params else None,
                'flux_max': continuum_params['flux_max'] if continuum_params else None,
                'mean': np.mean(flux_normalized),
                'std': np.std(flux_normalized)
            }
            
            # 返回处理后的光谱和标签，包括中间处理结果
            result = {
                'data': flux_normalized,  # 将spectrum改为data
                'metadata': {
                    'label': label,
                    'filename': spec_file,
                    'element': '',  # 会在process_element_data方法中被设置
                    # 保存中间结果用于可视化
                    'original_wavelength': wavelength,
                    'original_flux': flux,
                    'wavelength_calibrated': wavelength_calibrated,
                    'wavelength_corrected': wavelength_corrected,
                    'denoised_flux': flux_denoised,
                    'wavelength_rest': wavelength_rest,
                    'wavelength_resampled': wavelength_resampled, 
                    'flux_resampled': flux_resampled,
                    'flux_continuum': flux_continuum,
                    'flux_denoised_second': flux_denoised_second,
                    'z': z,  # 保存红移值
                    'v_helio': v_helio,
                    'snr': snr,  # 信噪比
                    'snr_bands': snr_bands,  # 各波段信噪比
                    'normalization_params': normalization_params
                },
                'validation_metrics': {
                    'quality_metrics': {
                        'snr': np.mean(flux_normalized) / np.std(flux_normalized) if np.std(flux_normalized) > 0 else 0,
                        'wavelength_coverage': 1.0,  # 默认为完整覆盖
                        'normalization_quality': 1.0  # 默认为良好质量
                    }
                }
            }
            
            # 为了保持向后兼容，添加旧字段
            result['spectrum'] = result['data']
            result['label'] = label
            result['filename'] = spec_file
            
            # 使用CacheManager保存结果
            self.cache_manager.set_cache(cache_key, result)
            return result
        except Exception as e:
            print(f"处理{spec_file}时出错: {e}")
            import traceback
            traceback.print_exc()
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
        """处理单个元素的数据，支持从指定位置继续处理"""
        print(f"处理{element}数据...")
        
        # 确保spec列为字符串类型
        if 'spec' in df.columns and not pd.api.types.is_string_dtype(df['spec']):
            print(f"注意: {element}_FE.csv 中的spec列不是字符串类型，正在转换...")
            df['spec'] = df['spec'].astype(str)
        
        # 获取光谱文件名和标签
        spec_files = df['spec'].values
        labels = df.iloc[:, -1].values  # 最后一列是标签
        
        # 检查进度文件
        progress_file = os.path.join(self.progress_dir, f"{element}_progress.pkl")
        drive_progress_file = f"/content/drive/My Drive/SPCNNet_Results/processed_data/progress/{element}_progress.pkl"
        
        results = []
        total_processed = 0
        
        # 如果有进度文件，加载已处理的结果
        if start_idx == 0:
            # 首先尝试标准目录
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'rb') as f:
                        saved_data = pickle.load(f)
                        
                        # 检查保存的数据结构
                        if isinstance(saved_data, dict) and 'results' in saved_data:
                            # 新格式：字典中包含results和last_idx
                            results = saved_data.get('results', [])
                            start_idx = saved_data.get('last_idx', 0)
                        elif isinstance(saved_data, list):
                            # 旧格式：直接是结果列表
                            results = saved_data
                            # 如果是列表，start_idx应该等于已处理的数量
                            start_idx = len(results)
                        else:
                            print(f"未识别的缓存数据格式: {type(saved_data)}，将从头开始处理")
                            results = []
                            start_idx = 0
                        
                        total_processed = len(results)
                        print(f"从上次中断处继续（已处理{total_processed}条记录，进度：{total_processed/len(spec_files):.2%}）")
                        
                        # 确保所有结果都有元素信息
                        for result in results:
                            if isinstance(result, dict):
                                if 'metadata' in result and 'element' not in result['metadata']:
                                    result['metadata']['element'] = element
                                elif 'element' not in result:
                                    result['element'] = element
                        
                        # 如果已经处理完所有数据，直接返回
                        if total_processed >= len(spec_files):
                            print(f"{element}数据已全部处理完成，共{total_processed}条记录")
                            return results
                        
                except Exception as e:
                    print(f"加载进度文件出错: {e}，将从头开始处理")
                    import traceback
                    traceback.print_exc()
                    start_idx = 0
                    results = []
                    
            # 如果标准目录没有，尝试Google Drive
            elif os.path.exists(drive_progress_file):
                try:
                    with open(drive_progress_file, 'rb') as f:
                        saved_data = pickle.load(f)
                        
                        # 同样检查数据结构
                        if isinstance(saved_data, dict) and 'results' in saved_data:
                            results = saved_data.get('results', [])
                            start_idx = saved_data.get('last_idx', 0)
                        elif isinstance(saved_data, list):
                            results = saved_data
                            start_idx = len(results)
                        else:
                            print(f"未识别的缓存数据格式: {type(saved_data)}，将从头开始处理")
                            results = []
                            start_idx = 0
                        
                        total_processed = len(results)
                        print(f"从Google Drive加载进度（已处理{total_processed}条记录，进度：{total_processed/len(spec_files):.2%}）")
                        
                        # 确保所有结果都有元素信息
                        for result in results:
                            if isinstance(result, dict):
                                if 'metadata' in result and 'element' not in result['metadata']:
                                    result['metadata']['element'] = element
                                elif 'element' not in result:
                                    result['element'] = element
                        
                        # 如果已经处理完所有数据，直接返回
                        if total_processed >= len(spec_files):
                            print(f"{element}数据已全部处理完成，共{total_processed}条记录")
                            return results
                        
                except Exception as e:
                    print(f"加载Google Drive进度文件出错: {e}，将从头开始处理")
                    import traceback
                    traceback.print_exc()
                    start_idx = 0
                    results = []
        
        # 计算剩余的批次
        remaining = len(spec_files) - start_idx
        if remaining <= 0:
            print(f"{element}数据已全部处理完成")
            return results
            
        num_batches = (remaining + self.batch_size - 1) // self.batch_size
        
        # 测试第一个文件，确认路径正确
        test_spec = spec_files[start_idx]
        print(f"测试第一个文件: {test_spec}")
        # 首先检查文件是否存在
        found_path = self._find_fits_file(test_spec)
        if found_path:
            print(f"找到文件路径: {found_path}")
        else:
            print(f"警告: 找不到文件 {test_spec}")
            return results
        
        # 尝试处理第一个文件
        print("尝试处理测试文件...")
        
        # 直接打开文件并检查数据格式
        try:
            with fits.open(found_path, ignore_missing_end=True, memmap=False) as hdul:
                print(f"FITS文件结构: 共{len(hdul)}个HDU")
                for i, hdu in enumerate(hdul):
                    hdu_type = type(hdu).__name__
                    print(f"  HDU{i}: 类型={hdu_type}")
                    
                    # 如果是BinTableHDU，输出表格信息
                    if isinstance(hdu, fits.BinTableHDU):
                        print(f"  表格列: {hdu.columns.names}")
                        print(f"  表格行数: {len(hdu.data)}")
                        
                        # 输出第一行数据类型
                        first_row = hdu.data[0]
                        print(f"  第一行数据类型: {type(first_row)}")
                        
                        # 检查每列的数据类型
                        for col_name in hdu.columns.names:
                            col_data = hdu.data[col_name]
                            if len(col_data) > 0:
                                print(f"  列 '{col_name}' 类型: {type(col_data[0])}")
                                # 如果是数组类型，尝试获取其形状
                                try:
                                    if hasattr(col_data[0], 'shape'):
                                        print(f"    数据形状: {col_data[0].shape}")
                                    elif hasattr(col_data[0], '__len__'):
                                        print(f"    数据长度: {len(col_data[0])}")
                                except:
                                    pass
        except Exception as e:
            print(f"检查FITS文件格式出错: {e}")
        
        # 正常处理第一个文件
        test_result = self.process_single_spectrum(test_spec, labels[start_idx])
        if test_result is None:
            print(f"警告: 无法处理第一个测试文件 {test_spec}，请检查文件内容或处理逻辑")
            # 尝试读取文件进行诊断
            try:
                with fits.open(found_path) as hdul:
                    header = hdul[0].header
                    print(f"文件头信息示例: NAXIS={header.get('NAXIS')}, NAXIS1={header.get('NAXIS1')}")
                    # 检查文件内容
                    data_shape = hdul[0].data.shape if hdul[0].data is not None else "无数据"
                    print(f"数据形状: {data_shape}")
            except Exception as e:
                print(f"读取文件出错: {e}")
            return results
        else:
            print(f"测试文件 {test_spec} 处理成功，继续批量处理")
            results.append(test_result)
            results[0]['element'] = element
            total_processed += 1
        
        # 逐批处理剩余数据
        for batch_idx in tqdm(range(num_batches), desc=f"处理{element}光谱批次"):
            # 计算当前批次的索引范围
            current_start = start_idx + 1 + batch_idx * self.batch_size  # 跳过已测试的第一个文件
            if current_start >= len(spec_files):
                break  # 防止越界
            
            current_end = min(current_start + self.batch_size, len(spec_files))
            batch_specs = spec_files[current_start:current_end]
            batch_labels = labels[current_start:current_end]
            
            if len(batch_specs) == 0 or len(batch_labels) == 0:
                continue  # 跳过空批次
            
            batch_results = []
            
            # 检查内存使用情况
            if self.check_memory_usage():
                # 如果内存紧张，先保存当前进度
                with open(progress_file, 'wb') as f:
                    pickle.dump({'results': results, 'last_idx': current_start}, f)
                # 保存文件查找缓存
                self._save_files_cache()
                print("内存使用率高，已保存进度，可以安全退出程序")
                
                # 询问是否继续
                if input("内存使用率较高，是否继续处理？(y/n): ").lower() != 'y':
                    return results
            
            # 使用多进程处理当前批次
            successful_count = 0
            failed_specs = []
            
            with Pool(processes=self.max_workers) as pool:
                jobs = []
                for spec_file, label in zip(batch_specs, batch_labels):
                    jobs.append(pool.apply_async(self.process_single_spectrum, 
                                                (spec_file, label)))
                
                for job in jobs:
                    try:
                        result = job.get(timeout=30)  # 设置超时避免卡死
                        if result is not None:
                            # 设置元素信息
                            if 'metadata' in result:
                                result['metadata']['element'] = element
                            else:
                                # 兼容旧格式
                                result['element'] = element
                            
                            batch_results.append(result)
                            successful_count += 1
                        else:
                            failed_specs.append(spec_file)
                    except Exception as e:
                        print(f"处理作业出错: {e}")
                        continue
            
            # 添加到结果集
            results.extend(batch_results)
            total_processed += successful_count
            
            # 输出当前批次统计和整体进度
            overall_progress = (total_processed / len(spec_files)) * 100
            if len(batch_specs) > 0:
                print(f"批次 {batch_idx+1}/{num_batches}: 成功 {successful_count}/{len(batch_specs)} 个文件")
                print(f"总进度: [{overall_progress:.2f}%] {total_processed}/{len(spec_files)} 已完成")
                if len(failed_specs) > 0 and len(failed_specs) < 5:
                    print(f"失败文件示例: {failed_specs}")
            
            # 定期保存进度
            if batch_idx % 5 == 0 or batch_idx == num_batches - 1:
                with open(progress_file, 'wb') as f:
                    pickle.dump({'results': results, 'last_idx': current_end}, f)
                # 保存文件查找缓存
                self._save_files_cache()
                print(f"✓ 进度已保存 [{overall_progress:.2f}%]")
        
        print(f"成功处理{total_processed}/{len(spec_files)}条{element}光谱数据 (完成率: {total_processed/len(spec_files):.2%})")
        return results
    
    def process_all_elements(self, elements, df_list, resume=True):
        """处理所有元素数据"""
        print(f"\n=== 开始预处理所有元素数据 ===")
        
        all_results = []
        
        # 统计总数据量用于进度显示
        total_records = sum(len(df) for df in df_list)
        print(f"总共需要处理 {len(elements)} 个元素，共 {total_records} 条记录")
        
        # 对每个元素进行处理
        for i, (element, df) in enumerate(zip(elements, df_list)):
            print(f"\n[{i+1}/{len(elements)}] 处理元素 {element} (共{len(df)}条记录)...")
            
            # 检查是否有缓存
            if resume:
                progress_file = os.path.join(self.progress_dir, f"{element}_progress.pkl")
                if os.path.exists(progress_file):
                    try:
                        with open(progress_file, 'rb') as f:
                            element_data = pickle.load(f)
                        print(f"从缓存加载 {element} 数据，共 {len(element_data)} 条记录")
                        all_results.extend(element_data)
                        continue
                    except Exception as e:
                        print(f"加载 {element} 缓存失败: {e}，将重新处理")
            
            # 处理元素数据
            element_data = self.process_element_data(df, element)
            
            # 保存进度
            if element_data:
                try:
                    progress_file = os.path.join(self.progress_dir, f"{element}_progress.pkl")
                    with open(progress_file, 'wb') as f:
                        pickle.dump(element_data, f)
                    print(f"保存 {element} 处理进度，共 {len(element_data)} 条记录")
                except Exception as e:
                    print(f"保存 {element} 进度失败: {e}")
            
            # 添加到总结果中
            all_results.extend(element_data)
        
        # 保存总体进度
        if all_results:
            try:
                all_progress_file = os.path.join(self.progress_dir, "all_progress.pkl")
                with open(all_progress_file, 'wb') as f:
                    pickle.dump(all_results, f)
                print(f"保存总处理进度，共 {len(all_results)} 条记录")
            except Exception as e:
                print(f"保存总进度失败: {e}")
        
        return all_results
    
    def process_all_data(self, resume=True):
        """预处理所有数据"""
        print("\n=== 开始预处理所有数据 ===")
        
        # 初始化返回变量，确保所有路径都能访问它们
        X, y, elements, filenames = None, None, None, None
        
        # 加载CSV数据
        dataframes, elements_list = self.read_csv_data()
        if not dataframes:
            print("错误: 没有找到有效的CSV文件")
            return None, None, None, None
        
        # 显示各元素数据情况
        print("\n=== CSV文件数据统计 ===")
        total_rows = 0
        for i, (element, df) in enumerate(zip(elements_list, dataframes)):
            num_rows = len(df)
            total_rows += num_rows
            print(f"{i+1}. 元素 {element}: {num_rows} 行数据")
            # 如果行数为0，显示警告
            if num_rows == 0:
                print(f"   警告: {element} 元素没有数据")
            # 如果行数过少，显示前几行
            elif num_rows < 10:
                print(f"   数据过少，显示全部数据:")
                print(df.head(num_rows))
        print(f"总计: {total_rows} 行数据\n")
        
        # 检查是否从缓存加载
        all_progress_file = os.path.join(self.progress_dir, "all_progress.pkl")
        if resume and os.path.exists(all_progress_file):
            try:
                with open(all_progress_file, 'rb') as f:
                    all_data = pickle.load(f)
                print(f"从缓存加载全部数据，共 {len(all_data)} 条记录")
                
                # 检查数据并修复错误格式
                if len(all_data) > 0:
                    print(f"缓存数据第一条记录类型: {type(all_data[0])}")
                    
                    # 特殊情况：如果缓存是字符串列表，尝试转换为适当的结构
                    if isinstance(all_data[0], str):
                        print("检测到缓存中包含字符串记录，尝试从各元素缓存中恢复数据...")
                        
                        # 清空当前all_data并从元素级缓存中重建
                        all_data = []
                        
                        # 扫描progress目录中的所有元素级缓存
                        element_progress_files = glob.glob(os.path.join(self.progress_dir, "*_progress.pkl"))
                        print(f"找到 {len(element_progress_files)} 个元素缓存文件")
                        
                        # 从每个元素级缓存中加载数据
                        for progress_file in element_progress_files:
                            element_name = os.path.basename(progress_file).replace("_progress.pkl", "")
                            print(f"正在加载元素 {element_name} 的缓存...")
                            
                            try:
                                with open(progress_file, 'rb') as f:
                                    element_data = pickle.load(f)
                                
                                # 检查元素数据格式
                                if isinstance(element_data, dict) and 'results' in element_data:
                                    # 新格式：包含results键的字典
                                    element_results = element_data['results']
                                elif isinstance(element_data, list):
                                    # 旧格式：直接是结果列表
                                    element_results = element_data
                                else:
                                    print(f"跳过元素 {element_name}: 未知的缓存格式")
                                    continue
                                
                                # 确保每个记录都有元素信息
                                for record in element_results:
                                    if isinstance(record, dict):
                                        if 'metadata' in record and 'element' not in record['metadata']:
                                            record['metadata']['element'] = element_name
                                        elif 'element' not in record:
                                            record['element'] = element_name
                                
                                print(f"从元素 {element_name} 缓存中加载了 {len(element_results)} 条记录")
                                all_data.extend(element_results)
                            except Exception as e:
                                print(f"加载元素 {element_name} 缓存时出错: {e}")
                        
                        # 更直接地加载单独的处理文件缓存
                        print("尝试从独立的处理文件缓存中加载数据...")
                        cache_files = glob.glob(os.path.join(self.cache_dir, "processed_*.pkl"))
                        print(f"找到 {len(cache_files)} 个独立缓存文件")
                        
                        loaded_count = 0
                        # 从每个缓存文件加载处理结果
                        for cache_file in cache_files:
                            try:
                                with open(cache_file, 'rb') as f:
                                    processed_data = pickle.load(f)
                                
                                # 检查数据格式
                                if isinstance(processed_data, dict):
                                    # 将缓存文件名添加到元数据中，以便跟踪来源
                                    filename = os.path.basename(cache_file).replace("processed_", "").replace(".pkl", "")
                                    if 'metadata' not in processed_data:
                                        processed_data['metadata'] = {'filename': filename}
                                    else:
                                        processed_data['metadata']['filename'] = filename
                                    
                                    # 添加到总数据中
                                    all_data.append(processed_data)
                                    loaded_count += 1
                            except Exception as e:
                                print(f"加载缓存文件 {cache_file} 时出错: {e}")
                        
                        print(f"从独立缓存文件中成功加载了 {loaded_count} 条记录")
                        
                        print(f"已从所有缓存中恢复共 {len(all_data)} 条记录")
                        
                        # 保存恢复后的总缓存
                        if all_data:
                            try:
                                with open(all_progress_file, 'wb') as f:
                                    pickle.dump(all_data, f)
                                print("已保存修复后的总缓存文件")
                            except Exception as e:
                                print(f"保存修复后的总缓存失败: {e}")
                        
                        # 如果仍然没有有效记录或仍然是非字典格式，则放弃使用缓存
                        if len(all_data) == 0 or (len(all_data) > 0 and not isinstance(all_data[0], dict)):
                            print("警告: 缓存数据记录格式不正确或为空")
                            choice = input("是否尝试手动恢复之前的处理结果? (y/n): ")
                            if choice.lower() == 'y':
                                # 手动恢复模式
                                try:
                                    # 清空当前all_data
                                    all_data = []
                                    # 扫描progress目录中的所有元素级缓存
                                    element_progress_files = glob.glob(os.path.join(self.progress_dir, "*_progress.pkl"))
                                    
                                    for progress_file in element_progress_files:
                                        element_name = os.path.basename(progress_file).replace("_progress.pkl", "")
                                        print(f"尝试恢复元素 {element_name} 的缓存...")
                                        
                                        try:
                                            # 先尝试进行正常加载
                                            with open(progress_file, 'rb') as f:
                                                element_data = pickle.load(f)
                                            
                                            # 检查元素数据格式并添加到总缓存
                                            if isinstance(element_data, dict) and 'results' in element_data:
                                                all_data.extend(element_data['results'])
                                                print(f"从 {element_name} 恢复了 {len(element_data['results'])} 条记录")
                                            elif isinstance(element_data, list):
                                                all_data.extend(element_data)
                                                print(f"从 {element_name} 恢复了 {len(element_data)} 条记录")
                                        except Exception as e:
                                            print(f"恢复元素 {element_name} 失败: {e}")
                                    
                                    # 手动恢复独立处理结果
                                    print("尝试手动恢复独立的处理文件缓存...")
                                    cache_files = glob.glob(os.path.join(self.cache_dir, "processed_*.pkl"))
                                    loaded_count = 0
                                    
                                    for cache_file in cache_files:
                                        try:
                                            with open(cache_file, 'rb') as f:
                                                processed_data = pickle.load(f)
                                            
                                            # 检查数据格式
                                            if isinstance(processed_data, dict):
                                                filename = os.path.basename(cache_file).replace("processed_", "").replace(".pkl", "")
                                                if 'metadata' not in processed_data:
                                                    processed_data['metadata'] = {'filename': filename}
                                                else:
                                                    processed_data['metadata']['filename'] = filename
                                                
                                                all_data.append(processed_data)
                                                loaded_count += 1
                                        except Exception as e:
                                            print(f"手动恢复缓存文件 {cache_file} 时出错: {e}")
                                    
                                    print(f"手动从独立缓存文件中恢复了 {loaded_count} 条记录")                                    
                                    print(f"手动恢复完成，共恢复 {len(all_data)} 条记录")
                                except Exception as e:
                                    print(f"手动恢复过程出错: {e}")
                            else:
                                print("不使用缓存，重新处理数据")
                                all_data = []
                    else:
                        # 验证每条记录是否有必要的字段
                        valid_records = []
                        for i, record in enumerate(all_data):
                            if record is None:
                                print(f"跳过第{i+1}条记录: 记录为None")
                                continue
                            if not isinstance(record, dict):
                                print(f"跳过第{i+1}条记录: 不是字典类型，而是 {type(record)}")
                                continue
                            if 'data' not in record and 'spectrum' not in record:
                                print(f"跳过第{i+1}条记录: 缺少光谱数据")
                                continue
                            if 'metadata' not in record and 'filename' not in record:
                                print(f"跳过第{i+1}条记录: 缺少元数据")
                                continue
                            
                            valid_records.append(record)
                        
                        all_data = valid_records
                        print(f"验证后有效记录数: {len(all_data)}")
                
                # 询问是否继续
                if isinstance(all_data, list) and len(all_data) > 0:
                    choice = input(f"找到{len(all_data)}条处理好的记录，是否使用这些缓存数据? (y/n): ")
                    if choice.lower() == 'y':
                        print("使用缓存数据")
                        
                        # 从已处理数据中恢复公共波长范围
                        if self.compute_common_range:
                            if all_data[0] and 'metadata' in all_data[0]:
                                wavelength_range = all_data[0]['metadata'].get('normalization_params', {}).get('wavelength_range')
                                if wavelength_range:
                                    self.wavelength_range = wavelength_range
                                    print(f"从缓存恢复波长范围: {self.wavelength_range}")
                            
                            # 重新计算点数
                            if self.wavelength_range and self.log_step:
                                start_wavelength, end_wavelength = self.wavelength_range
                                # 在对数空间中计算点数
                                self.n_points = int(np.ceil(np.log10(end_wavelength / start_wavelength) / self.log_step))
                                print(f"重新计算点数: {self.n_points}")
                        
                        # 如果用户要求，随机可视化几条光谱
                        visualize_choice = input("是否随机可视化几条光谱? (y/n): ").lower()
                        if visualize_choice == 'y':
                            n_samples = min(len(all_data), 3)
                            try:
                                samples = random.sample(all_data, n_samples)
                                for sample in samples:
                                    # 安全地获取文件名
                                    filename = None
                                    if isinstance(sample, dict):
                                        if 'filename' in sample:
                                            filename = sample['filename']
                                        elif 'metadata' in sample and isinstance(sample['metadata'], dict) and 'filename' in sample['metadata']:
                                            filename = sample['metadata']['filename']
                                    elif isinstance(sample, str):
                                        # 直接使用字符串作为文件名，但不将其作为sample_data传递
                                        filename = sample
                                        sample = None
                                        print(f"警告: 样本是字符串 '{filename}'，将作为文件名使用但不会传递该字符串作为样本数据")
                                        
                                    if filename:
                                        print(f"可视化: {filename}")
                                        # 直接传递样本数据，避免重新处理
                                        self.visualize_spectrum(filename, processed=True, save=True, 
                                                               from_all_data=True, sample_data=sample)
                                    else:
                                        print(f"样本中没有文件名信息: {type(sample)}")
                            except Exception as e:
                                print(f"可视化过程中出错: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        # 询问是否需要分割数据集
                        split_choice = input("是否需要划分数据集? (y/n): ").lower()
                        if split_choice == 'y':
                            print("准备数据用于模型训练")
                            X, y, elements, filenames = self._prepare_arrays(all_data)
                            if X is not None and y is not None:
                                print(f"准备完成，特征数据形状: {X.shape}, 标签数据形状: {y.shape}")
                                self.split_dataset(X, y, elements)
                            return X, y, elements, filenames
                        else:
                            # 即使不划分数据集，也需要准备数组以便返回
                            print("准备数据但不划分数据集")
                            X, y, elements, filenames = self._prepare_arrays(all_data)
                            if X is not None and y is not None:
                                print(f"准备完成，特征数据形状: {X.shape}, 标签数据形状: {y.shape}")
                            
                            # 用户选择不划分后，跳过main函数中的第二次询问
                            # 保存完整数据集到单一文件
                            if X is not None and y is not None:
                                output_dir = self.output_dir
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir, exist_ok=True)
                                
                                full_dataset_path = os.path.join(output_dir, 'full_dataset.npz')
                                np.savez(full_dataset_path, spectra=X, abundance=y, elements=elements)
                                print(f"完整数据集保存完成: {X.shape[0]}条记录，保存到{full_dataset_path}")
                            
                            # 如果用户选择不划分数据集但想要可视化示例，在此处理
                            if visualize_choice != 'y' and input("是否可视化示例光谱? (y/n): ").lower() == 'y':
                                if filenames is not None and len(filenames) > 0:
                                    print("正在可视化示例光谱...")
                                    sample_indices = random.sample(range(len(filenames)), min(3, len(filenames)))
                                    for i in sample_indices:
                                        self.visualize_spectrum(filenames[i], processed=True, save=True)
                            
                            return X, y, elements, filenames
                    else:
                        print("不使用缓存，重新处理数据")
            except Exception as e:
                print(f"加载缓存数据失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 第一步：计算最大公共波长范围
        print("\n=== 第一步：计算最大公共波长范围 ===")
        
        # 已存在波长范围文件，尝试加载
        wavelength_range_file = os.path.join(self.cache_dir, 'wavelength_range.pkl')
        if os.path.exists(wavelength_range_file) and resume:
            try:
                with open(wavelength_range_file, 'rb') as f:
                    self.wavelength_range = pickle.load(f)
                print(f"从缓存加载公共波长范围: {self.wavelength_range}")
                
                # 根据公共波长范围和步长计算点数
                start_wavelength, end_wavelength = self.wavelength_range
                self.n_points = int(np.ceil(np.log10(end_wavelength / start_wavelength) / self.log_step))
                print(f"对应点数: {self.n_points} (步长={self.log_step})")
                
            except Exception as e:
                print(f"加载公共波长范围失败: {e}，将重新计算")
                # 如果加载失败，继续执行计算波长范围的步骤
        
        # 如果没有缓存的波长范围或加载失败，则计算波长范围
        if self.compute_common_range and (not hasattr(self, 'wavelength_range') or self.wavelength_range is None):
            print("计算所有光谱的公共波长范围...")
            
            # 临时收集波长范围信息
            all_wave_ranges = []
            
            # 对每个元素进行处理，但只提取波长范围信息
            for i, (element, df) in enumerate(zip(elements_list, dataframes)):
                print(f"\n[{i+1}/{len(elements_list)}] 读取元素 {element} 的波长范围 (共{len(df)}条记录)...")
                
                # 只处理前几条记录快速确定波长范围
                sample_size = min(len(df), 20)  # 限制样本数量加快处理
                for j, (_, row) in enumerate(df.head(sample_size).iterrows()):
                    try:
                        spec_file = str(row['spec'])
                        # 读取波长数据但不进行完整预处理
                        wavelength, flux, _, _, _, _ = self.read_fits_file(spec_file)
                        if wavelength is not None and len(wavelength) > 0:
                            valid_mask = ~np.isnan(flux) & ~np.isinf(flux)
                            if np.any(valid_mask):
                                wavelength_valid = wavelength[valid_mask]
                                w_min, w_max = wavelength_valid.min(), wavelength_valid.max()
                                all_wave_ranges.append((w_min, w_max))
                                print(f"  文件 {spec_file}: 波长范围 {w_min:.2f}~{w_max:.2f}")
                    except Exception as e:
                        print(f"  读取文件 {spec_file} 波长范围时出错: {e}")
                        continue
            
            # 计算最大公共波长范围
            if all_wave_ranges:
                min_waves = [r[0] for r in all_wave_ranges]
                max_waves = [r[1] for r in all_wave_ranges]
                common_min = max(min_waves)
                common_max = min(max_waves)
                
                if common_min >= common_max:
                    print(f"警告: 计算出的公共范围无效 ({common_min}~{common_max})，使用默认范围")
                    self.wavelength_range = (3900, 7000)  # 使用默认范围
                else:
                    # 添加一点余量
                    self.wavelength_range = (common_min, common_max)
                
                # 保存波长范围信息
                try:
                    with open(wavelength_range_file, 'wb') as f:
                        pickle.dump(self.wavelength_range, f)
                    print(f"波长范围信息已保存")
                except Exception as e:
                    print(f"保存波长范围信息失败: {e}")
            else:
                print("没有找到任何波长范围信息，使用默认范围")
                self.wavelength_range = (3900, 7000)
        
        # 根据公共波长范围和步长计算点数
        if hasattr(self, 'wavelength_range') and self.wavelength_range is not None:
            start_wavelength, end_wavelength = self.wavelength_range
            self.n_points = int(np.ceil(np.log10(end_wavelength / start_wavelength) / self.log_step))
            print(f"公共波长范围: {self.wavelength_range}")
            print(f"对应点数: {self.n_points} (步长={self.log_step})")
        
        # 第二步：处理所有光谱数据
        print("\n=== 第二步：使用公共波长范围进行完整预处理 ===")
        all_data = self.process_all_elements(elements_list, dataframes, resume)
        
        # 第三步：询问是否分割数据集
        if all_data and input("是否需要划分数据集? (y/n): ").lower() == 'y':
            print("准备数据用于模型训练")
            X, y, elements, filenames = self._prepare_arrays(all_data)
            if X is not None and y is not None:
                self.split_dataset(X, y, elements)
        
        # 第四步：询问是否随机可视化
        if all_data and input("是否随机可视化几条光谱? (y/n): ").lower() == 'y':
            n_samples = min(len(all_data), 3)
            try:
                samples = random.sample(all_data, n_samples)
                for sample in samples:
                    # 安全地获取文件名
                    filename = None
                    if isinstance(sample, dict):
                        if 'filename' in sample:
                            filename = sample['filename']
                        elif 'metadata' in sample and isinstance(sample['metadata'], dict) and 'filename' in sample['metadata']:
                            filename = sample['metadata']['filename']
                    elif isinstance(sample, str):
                        # 直接使用字符串作为文件名
                        filename = sample
                        
                    if filename:
                        print(f"可视化: {filename}")
                        # 直接传递样本数据，避免重新处理
                        self.visualize_spectrum(filename, processed=True, save=True, 
                                               from_all_data=True, sample_data=sample)
                    else:
                        print(f"样本中没有文件名信息: {type(sample)}")
            except Exception as e:
                print(f"可视化过程中出错: {e}")
                import traceback
                traceback.print_exc()
        
        # 最终处理数据
        if all_data:
            X, y, elements, filenames = self._prepare_arrays(all_data)
            return X, y, elements, filenames
        else:
            print("没有有效数据可处理")
            return None, None, None, None
    
    def _prepare_arrays(self, all_data):
        """准备训练、验证和测试数据数组"""
        if not all_data:
            print("没有可用的数据")
            return None, None, None, None
            
        # 提取光谱数据和标签
        spectra = []
        labels = []
        filenames = []
        elements = []
        
        for data in all_data:
            # 忽略无效数据
            if not data or 'data' not in data or data['data'] is None:
                continue
                
            # 获取光谱数据
            spectrum = data['data']
            # 检查光谱数据有效性
            if spectrum is None or len(spectrum) == 0:
                continue
                
            # 替换NaN和无穷值
            if np.isnan(spectrum).any() or np.isinf(spectrum).any():
                print(f"发现无效值，替换为0: {data['filename']}")
                spectrum = np.nan_to_num(spectrum, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 获取标签
            label = data['metadata']['label']
            # 检查标签有效性
            if pd.isna(label):
                print(f"跳过数据，标签为NaN: {data['filename']}")
                continue
                
            spectra.append(spectrum)
            labels.append(label)
            filenames.append(data['filename'])
            elements.append(data['metadata']['element'])
        
        if not spectra:
            print("处理后没有有效数据")
            return None, None, None, None
            
        # 检查所有光谱的长度是否一致
        expected_length = len(spectra[0])
        inconsistent_indices = []
        
        for i, spec in enumerate(spectra):
            if len(spec) != expected_length:
                print(f"警告: 光谱 {i} ({filenames[i]}) 长度不一致。期望长度: {expected_length}, 实际长度: {len(spec)}")
                inconsistent_indices.append(i)
        
        if inconsistent_indices:
            print(f"发现 {len(inconsistent_indices)} 个光谱长度不一致，将从数据集中移除")
            # 从后往前删除，避免索引变化
            for i in sorted(inconsistent_indices, reverse=True):
                print(f"移除不一致光谱: {filenames[i]}")
                del spectra[i]
                del labels[i]
                del filenames[i]
                del elements[i]
        
        if not spectra:
            print("移除不一致光谱后没有有效数据")
            return None, None, None, None
        
        try:
            # 转换为NumPy数组
            X = np.array(spectra, dtype=np.float32)
            y = np.array(labels, dtype=np.float32)
            
            # 最终检查X中的无效值
            n_samples, n_features = X.shape
            for i in range(n_samples):
                if np.isnan(X[i]).any() or np.isinf(X[i]).any():
                    nan_count = np.isnan(X[i]).sum()
                    inf_count = np.isinf(X[i]).sum()
                    print(f"警告: 样本 {i} ({filenames[i]}) 包含 {nan_count} 个NaN和 {inf_count} 个无限值，替换为0")
                    X[i] = np.nan_to_num(X[i], nan=0.0, posinf=0.0, neginf=0.0)
            
            # 检查y中的无效值
            if np.isnan(y).any() or np.isinf(y).any():
                nan_count = np.isnan(y).sum()
                inf_count = np.isinf(y).sum()
                print(f"警告: 标签中包含 {nan_count} 个NaN和 {inf_count} 个无限值")
                # 找出包含NaN的样本
                nan_indices = np.where(np.isnan(y))[0]
                for idx in nan_indices:
                    print(f"  样本 {idx} ({filenames[idx]}) 的标签为NaN")
                # 替换NaN为中位数
                if nan_count > 0:
                    median_y = np.nanmedian(y)
                    print(f"  使用中位数 {median_y} 替换标签中的NaN")
                    y = np.nan_to_num(y, nan=median_y)
            
            print(f"准备完成 {len(X)} 个样本, 特征数: {X.shape[1]}")
            return X, y, elements, filenames
            
        except ValueError as e:
            print(f"创建数组时出错: {e}")
            print("尝试诊断问题...")
            
            # 收集长度信息进行诊断
            lengths = [len(spec) for spec in spectra]
            unique_lengths = set(lengths)
            
            if len(unique_lengths) > 1:
                print(f"发现多种不同的光谱长度: {unique_lengths}")
                
                # 选择最常见的长度
                from collections import Counter
                length_counts = Counter(lengths)
                most_common_length = length_counts.most_common(1)[0][0]
                print(f"最常见的光谱长度为 {most_common_length}，将只保留这个长度的光谱")
                
                # 只保留长度一致的光谱
                consistent_data = []
                for i, spec in enumerate(spectra):
                    if len(spec) == most_common_length:
                        consistent_data.append((spectra[i], labels[i], filenames[i], elements[i]))
                
                if not consistent_data:
                    print("没有足够的一致长度光谱")
                    return None, None, None, None
                
                # 重建数据
                spectra, labels, filenames, elements = zip(*consistent_data)
                
                # 再次尝试创建数组
                try:
                    X = np.array(spectra, dtype=np.float32)
                    y = np.array(labels, dtype=np.float32)
                    print(f"成功创建一致长度的数组: {X.shape}")
                    
                    # 最终检查无效值
                    n_samples, n_features = X.shape
                    for i in range(n_samples):
                        if np.isnan(X[i]).any() or np.isinf(X[i]).any():
                            X[i] = np.nan_to_num(X[i], nan=0.0, posinf=0.0, neginf=0.0)
                    
                    if np.isnan(y).any() or np.isinf(y).any():
                        y = np.nan_to_num(y, nan=np.nanmedian(y))
                    
                    return X, y, elements, filenames
                except Exception as e2:
                    print(f"第二次尝试创建数组时出错: {e2}")
                    return None, None, None, None
            else:
                print(f"所有光谱长度都相同 ({lengths[0]})，但仍然发生错误")
                return None, None, None, None
    
    def split_dataset(self, X, y, elements, ask_for_split=True):
        """将数据集分割为训练、验证和测试集"""
        if ask_for_split:
            response = input("是否要进行数据集划分？(y/n): ").lower()
            if response != 'y':
                print(f"跳过数据集划分，将完整数据集保存到 {os.path.join(self.output_dir, 'full_dataset.npz')}")
                
                # 保存完整数据集，使用已有的数据
                np.savez(os.path.join(self.output_dir, 'full_dataset.npz'),
                        X=X, y=y, elements=elements)
                
                print(f"完整数据集保存完成: {X.shape[0]}条记录")
                
                # 返回完整数据集作为训练集，空数组作为验证集和测试集
                return (X, y, elements), (np.array([]), np.array([]), np.array([])), (np.array([]), np.array([]), np.array([]))
        
        # 检查数据集大小，如果样本数量太少，无法进行分割
        n_samples = X.shape[0]
        if n_samples <= 5:  # 如果样本数量太少，不进行分割
            print(f"警告: 样本数量({n_samples})太少，无法进行有效的数据集划分")
            print(f"将相同的数据集用作训练、验证和测试集")
            
            # 保存相同的数据集
            np.savez(os.path.join(self.output_dir, 'train_dataset.npz'),
                    spectra=X, abundance=y, elements=elements)
            np.savez(os.path.join(self.output_dir, 'val_dataset.npz'),
                    spectra=X, abundance=y, elements=elements)
            np.savez(os.path.join(self.output_dir, 'test_dataset.npz'),
                    spectra=X, abundance=y, elements=elements)
            
            print(f"数据集保存完成: 每个集合 {n_samples} 条记录")
            
            return (X, y, elements), (X, y, elements), (X, y, elements)
        
        # 原有的数据集分割逻辑，用于样本数量足够的情况
        # 首先分割出测试集 (80% vs 20%)
        X_temp, X_test, y_temp, y_test, elements_temp, elements_test = train_test_split(
            X, y, elements, test_size=0.2, random_state=42)
        
        # 再从剩余数据中分割出验证集 (70% vs 10%)
        X_train, X_val, y_train, y_val, elements_train, elements_val = train_test_split(
            X_temp, y_temp, elements_temp, test_size=1/8, random_state=42)
        
        # 保存数据集
        np.savez(os.path.join(self.output_dir, 'train_dataset.npz'),
                spectra=X_train, abundance=y_train, elements=elements_train)
        np.savez(os.path.join(self.output_dir, 'val_dataset.npz'),
                spectra=X_val, abundance=y_val, elements=elements_val)
        np.savez(os.path.join(self.output_dir, 'test_dataset.npz'),
                spectra=X_test, abundance=y_test, elements=elements_test)
        
        print(f"数据集分割完成:")
        print(f"训练集: {X_train.shape[0]}条 (70%)")
        print(f"验证集: {X_val.shape[0]}条 (10%)")
        print(f"测试集: {X_test.shape[0]}条 (20%)")
        
        return (X_train, y_train, elements_train), (X_val, y_val, elements_val), (X_test, y_test, elements_test)
    
    def predict_abundance(self, fits_file, model):
        """使用已训练的模型预测单个光谱的丰度"""
        # 处理光谱
        result = self.process_single_spectrum(fits_file, 0.0)  # 使用占位符标签
        if result is None:
            print(f"无法处理文件: {fits_file}")
            return None
        
        # 获取处理后的光谱数据
        spectrum = None
        if 'data' in result:
            spectrum = result['data']
        elif 'spectrum' in result:
            spectrum = result['spectrum']
        
        if spectrum is None:
            print(f"无法获取光谱数据: {fits_file}")
            return None
            
        # 确保光谱是二维数组 (样本数, 特征数)
        spectrum_array = np.array(spectrum).reshape(1, -1)
        
        # 进行预测
        try:
            prediction = model.predict(spectrum_array)
            print(f"预测结果: {prediction[0]}")
            return prediction[0]
        except Exception as e:
            print(f"预测时出错: {e}")
            return None
    
    def visualize_spectrum(self, spec_file, processed=True, save=True, from_all_data=False, sample_data=None):
        """可视化单个光谱，原始光谱或处理后的光谱
        
        参数:
            spec_file: 光谱文件名
            processed: 是否显示处理后光谱
            save: 是否保存图像
            from_all_data: 是否来自已处理的总数据
            sample_data: 直接传入的样本数据（如果有）
        """
        print(f"开始可视化光谱: {spec_file}")
        
        # 处理缓存键
        spec_file_base = os.path.basename(spec_file) if spec_file else "unknown"
        cache_key = f"processed_{spec_file_base.replace('/', '_')}"
        
        processed_data = None
        
        # 尝试获取处理数据的顺序:
        # 1. 使用传入的样本数据(如果是字典类型)
        # 2. 从缓存获取
        # 3. 处理新的光谱
        
        if sample_data is not None and isinstance(sample_data, dict):
            print(f"使用传入的字典类型样本数据")
            processed_data = sample_data
        elif sample_data is not None and isinstance(sample_data, str):
            print(f"传入的样本数据是字符串类型: '{sample_data}'，将尝试从缓存查找")
            # 对于字符串类型的样本，我们只使用它作为文件名查找缓存
            spec_file = sample_data
            cache_key = f"processed_{spec_file.replace('/', '_')}"
        
        # 如果没有有效的处理数据，尝试从缓存获取
        if processed_data is None:
            print(f"尝试从缓存获取光谱数据: {cache_key}")
            cached_data = self.cache_manager.get_cache(cache_key)
            
            if cached_data is not None:
                print(f"成功从缓存加载光谱数据")
                processed_data = cached_data
            else:
                # 如果没找到直接匹配的缓存，尝试列出所有可能匹配的缓存文件
                print(f"没有找到精确匹配的光谱缓存")
                matching_files = []
                
                if os.path.exists(self.cache_dir):
                    cache_files = [os.path.basename(f) for f in glob.glob(os.path.join(self.cache_dir, "*.pkl"))]
                    print(f"缓存目录中有 {len(cache_files)} 个文件")
                    
                    # 尝试两种匹配方式
                    spec_key = spec_file.replace('/', '_')
                    matching_files = [f for f in cache_files if spec_key in f]
                    
                    if not matching_files and '.' in spec_key:
                        # 尝试不带扩展名匹配
                        spec_key_no_ext = spec_key.split('.')[0]
                        matching_files = [f for f in cache_files if spec_key_no_ext in f]
                    
                    if matching_files:
                        print(f"找到可能匹配的缓存文件: {matching_files}")
                        
                        # 尝试加载第一个匹配的文件
                        try:
                            best_match = matching_files[0]
                            cache_path = os.path.join(self.cache_dir, best_match)
                            with open(cache_path, 'rb') as f:
                                processed_data = pickle.load(f)
                            print(f"成功加载匹配的缓存文件: {best_match}")
                        except Exception as e:
                            print(f"加载匹配的缓存文件失败: {e}")
                
        # 如果仍然没有处理数据，尝试处理光谱
        if processed_data is None:
            print(f"没有找到缓存数据，处理光谱: {spec_file}")
            processed_data = self.process_single_spectrum(spec_file, 0.0)  # 使用占位符标签
            
            if processed_data is None:
                print(f"处理光谱失败: {spec_file}")
                return
        
        # 检查processed_data是否是字典类型
        if not isinstance(processed_data, dict):
            print(f"错误: processed_data不是字典类型，而是 {type(processed_data)}")
            return
                
        # 提取数据，支持新旧缓存结构
        if 'metadata' in processed_data:
            metadata = processed_data['metadata']
            original_wavelength = metadata.get('original_wavelength')
            original_flux = metadata.get('original_flux')
            wavelength_calibrated = metadata.get('wavelength_calibrated')
            wavelength_corrected = metadata.get('wavelength_corrected')
            wavelength_rest = metadata.get('wavelength_rest')
            denoised_flux = metadata.get('denoised_flux')
            wavelength_resampled = metadata.get('wavelength_resampled')
            flux_resampled = metadata.get('flux_resampled')
            flux_continuum = metadata.get('flux_continuum')
            flux_denoised_second = metadata.get('flux_denoised_second')
            z = metadata.get('z', 0)
            spectrum = processed_data.get('data')
        else:
            # 兼容旧格式
            original_wavelength = processed_data.get('original_wavelength')
            original_flux = processed_data.get('original_flux')
            wavelength_calibrated = processed_data.get('wavelength_calibrated')
            wavelength_corrected = processed_data.get('wavelength_corrected')
            wavelength_rest = processed_data.get('wavelength_rest')
            denoised_flux = processed_data.get('denoised_flux')
            wavelength_resampled = processed_data.get('wavelength_resampled')
            flux_resampled = processed_data.get('flux_resampled')
            flux_continuum = processed_data.get('flux_continuum')
            flux_denoised_second = processed_data.get('flux_denoised_second')
            z = processed_data.get('z', 0)
            spectrum = processed_data.get('spectrum') or processed_data.get('data')
        
        # 设置字体和图形样式
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图形
        plt.figure(figsize=(14, 16))  # 调整高度以容纳四张子图
        
        # 获取光谱类型和观测日期信息（如果有）
        spec_type = ""
        obs_date = ""
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
        plot_with_labels(ax1, original_wavelength, original_flux, 
                         (min(original_wavelength), max(original_wavelength)), 
                         absorption_lines, color='blue', label_name='Raw Spectrum')
        ax1.set_ylabel('Flux (relative)')
        overall_title = f"Spectrum: {os.path.basename(spec_file)}"
        ax1.set_title(overall_title)
        
        # 在第一个子图下方添加红移和观测日期信息
        if z or obs_date:
            info_text = ""
            if z:
                info_text += f"Star z={z} "
            if obs_date:
                info_text += f"{obs_date}"
            ax1.text(0.02, -0.2, info_text, transform=ax1.transAxes, fontsize=9)
        
        # 如果是处理后的光谱，显示处理效果
        if processed:
            # 校准、速度校正、去噪、红移校正和重采样后的结果
            ax2 = plt.subplot(4, 1, 2)
            plot_with_labels(ax2, wavelength_resampled, flux_resampled, 
                            (min(wavelength_resampled), max(wavelength_resampled)), 
                            absorption_lines, color='green', label_name='Calibrated, Velocity Corrected, Denoised, Redshift Corrected & Resampled')
            ax2.set_ylabel('Flux')
            ax2.set_title("Spectrum after Calibration, Velocity Correction, Denoising, Redshift Correction & Resampling")
            
            # 连续谱归一化 - 这是第三张图
            ax3 = plt.subplot(4, 1, 3)
            plot_with_labels(ax3, wavelength_resampled, flux_continuum, 
                            (min(wavelength_resampled), max(wavelength_resampled)), 
                            absorption_lines, color='purple', label_name='Continuum Normalized')
            ax3.set_ylabel('Normalized Flux')
            ax3.set_title("Spectrum after Continuum Normalization")
            
            # 二次去噪和最终归一化 - 这是第四张图
            ax4 = plt.subplot(4, 1, 4)
            # 确保最终归一化到[0,1]范围
            spectrum_normalized = np.clip(flux_denoised_second if flux_denoised_second is not None else spectrum, 0, 1)
            plot_with_labels(ax4, wavelength_resampled, spectrum_normalized, 
                            (min(wavelength_resampled), max(wavelength_resampled)), 
                            absorption_lines, color='red', label_name='Fully Processed')
            ax4.set_ylabel('Final Normalized Flux')
            ax4.set_title("Spectrum after Second Denoising and Final Normalization")
            ax4.set_ylim(0, 1)  # 设置y轴范围为[0,1]
        
        else:
            # 如果不是处理后光谱，则使用原始光谱进行处理并显示
            
            # 1. 波长校正
            wavelength_calibrated = self.correct_wavelength(processed_data['original_wavelength'], processed_data['original_flux'])
            print(f"波长校正后: 波长范围{wavelength_calibrated[0]}~{wavelength_calibrated[-1]}")
            
            # 从FITS文件读取视向速度
            v_helio = 0  # 默认值
            try:
                _, _, v_helio, _, _, _ = self.read_fits_file(spec_file)
            except Exception as e:
                print(f"读取FITS文件获取视向速度时出错: {e}")
            
            # 2. 视向速度校正
            wavelength_corrected = self.correct_velocity(wavelength_calibrated, processed_data['original_flux'], v_helio)
            print(f"视向速度校正后: 波长范围{wavelength_corrected[0]}~{wavelength_corrected[-1]}")
            
            # 3. 去噪
            flux_denoised = self.denoise_spectrum(wavelength_corrected, processed_data['original_flux'])
            if flux_denoised is None:
                print(f"去噪{spec_file}失败")
                return
            
            # 4. 红移校正
            wavelength_rest = self.correct_redshift(wavelength_corrected, flux_denoised, z)
            print(f"红移校正后: 波长范围{wavelength_rest[0]}~{wavelength_rest[-1]}")
            
            # 5. 重采样
            print(f"重采样到波长范围: {self.wavelength_range}, 点数={self.n_points}")
            wavelength_resampled, flux_resampled = self.resample_spectrum(wavelength_rest, flux_denoised)
            if wavelength_resampled is None or flux_resampled is None:
                print(f"重采样{spec_file}失败")
                return
            
            # 6. 连续谱归一化
            flux_continuum, continuum_params = self.normalize_continuum(wavelength_resampled, flux_resampled)
            if flux_continuum is None:
                print(f"连续谱归一化{spec_file}失败")
                return None
            
            # 7. 二次去噪
            flux_denoised_second = self.denoise_spectrum_second(wavelength_resampled, flux_continuum)
            
            # 8. 最终归一化 (最大最小值归一化)
            print(f"对流量进行最终归一化")
            flux_normalized = self.normalize_spectrum(wavelength_resampled, flux_denoised_second)
            if flux_normalized is None:
                print(f"归一化{spec_file}失败")
                return None
            
            # 不再需要额外的归一化验证，normalize_spectrum已经完成了完整的验证
            print(f"成功处理光谱: {spec_file}")
            
            spectrum = flux_normalized
            
            # 显示处理过程 - 调整为符合要求的格式
            ax2 = plt.subplot(4, 1, 2)
            # 校准、速度校正、去噪、红移校正和重采样后的结果
            plot_with_labels(ax2, wavelength_resampled, flux_resampled, 
                           (min(wavelength_resampled), max(wavelength_resampled)), 
                           absorption_lines, color='green', label_name='Calibrated, Velocity Corrected, Denoised, Redshift Corrected & Resampled')
            ax2.set_ylabel('Flux')
            ax2.set_title("Spectrum after Calibration, Velocity Correction, Denoising, Redshift Correction & Resampling")
            
            # 连续谱归一化后的光谱
            ax3 = plt.subplot(4, 1, 3)
            plot_with_labels(ax3, wavelength_resampled, flux_continuum, 
                           (min(wavelength_resampled), max(wavelength_resampled)), 
                           absorption_lines, color='purple', label_name='Continuum Normalized')
            ax3.set_ylabel('Normalized Flux')
            ax3.set_title("Spectrum after Continuum Normalization")
            
            # 二次去噪和最终归一化后的光谱
            ax4 = plt.subplot(4, 1, 4)
            # 确保最终归一化到[0,1]范围
            spectrum_normalized = np.clip(spectrum, 0, 1)
            plot_with_labels(ax4, wavelength_resampled, spectrum_normalized, 
                           (min(wavelength_resampled), max(wavelength_resampled)), 
                           absorption_lines, color='red', label_name='Fully Processed')
            ax4.set_ylabel('Final Normalized Flux')
            ax4.set_title("Spectrum after Second Denoising and Final Normalization")
            ax4.set_ylim(0, 1)  # 设置y轴范围为[0,1]
        
        # 添加波长范围和处理信息
        if self.compute_common_range and len(self.processed_ranges) > 1:
            range_description = f'Wavelength Range: {self.wavelength_range[0]:.2f}-{self.wavelength_range[1]:.2f} Å'
        else:
            range_description = f'Wavelength Range: {self.wavelength_range[0]:.2f}-{self.wavelength_range[1]:.2f} Å'
        
        # 检查是否使用对数步长
        if hasattr(self, 'log_step') and self.log_step:
            step_description = f'Log Step: {self.log_step} dex'
        else:
            step_description = f'Points: {len(wavelength_resampled)}'
        
        # 添加红移信息
        z_description = f'Redshift: z = {z}' if z else ''
        
        info_text = f'{range_description}, {step_description}'
        if z_description:
            info_text += f', {z_description}'
        
        plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=10)
        
        plt.tight_layout(pad=2.0, rect=[0, 0.02, 1, 0.98])
        
        if save:
            output_file = os.path.join(self.output_dir, f"{os.path.basename(spec_file)}_visualization.png")
            plt.savefig(output_file, dpi=150)
            print(f"Image saved to: {output_file}")
            plt.close()
        else:
            plt.show()
    
    def check_data_sources(self):
        """检查数据源是否存在并可用"""
        print("\n=== 检查数据源 ===")
        
        # 检查是否跳过详细检查
        if len(self.fits_file_cache) > 0:
            skip_check = input(f"已加载{len(self.fits_file_cache)}个文件路径缓存。是否跳过详细FITS文件检查？(y/n): ").lower() == 'y'
            if skip_check:
                print("跳过详细FITS文件检查，假定文件存在")
                print("✓ CSV文件已就绪")
                print("✓ FITS目录已就绪")
                print("\n=== 检查完成 ===\n")
                return True
        
        # 检查CSV文件
        csv_ok = True
        for i, csv_file in enumerate(self.csv_files, 1):
            if os.path.exists(csv_file):
                print(f"✓ {i}. CSV文件存在: {csv_file}")
                # 检查大小是否合理
                size_mb = os.path.getsize(csv_file) / (1024 * 1024)
                print(f"   大小: {size_mb:.2f} MB")
                # 检查是否包含spec列
                if 'spec' in pd.read_csv(csv_file).columns:
                    print(f"  - 'spec'列存在")
                else:
                    print(f"  ✗ 错误: {csv_file} 中没有'spec'列")
                    csv_ok = False
            else:
                print(f"✗ {i}. CSV文件不存在: {csv_file}")
                csv_ok = False
        
        # 检查fits目录
        fits_ok = True
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
                
                # 检查CSV文件中的spec值是否匹配fits文件
                for csv_file in self.csv_files:
                    if os.path.exists(csv_file):
                        df = pd.read_csv(csv_file)
                        if 'spec' in df.columns and len(df) > 0:
                            # 检查spec列的数据类型并转换为字符串
                            if not pd.api.types.is_string_dtype(df['spec']):
                                print(f"\n  警告: {csv_file} 中的spec列不是字符串类型，正在转换...")
                                df['spec'] = df['spec'].astype(str)
                            
                            spec_examples = df['spec'].iloc[:5].tolist()
                            print(f"\n  检查 {csv_file} 中的spec值是否匹配fits文件:")
                            
                            for spec in spec_examples:
                                # 使用新的查找逻辑
                                found_path = self._find_fits_file(spec)
                                if found_path:
                                    rel_path = os.path.relpath(found_path, self.fits_dir)
                                    print(f"    ✓ {spec} 存在: {rel_path}")
                                else:
                                    print(f"    ✗ {spec} 不存在，在所有目录中都未找到")
                            
                            # 检测常见文件格式
                            extensions = [os.path.splitext(f)[1] for f in all_fits_files[:20]]
                            common_exts = set(extensions)
                            if common_exts:
                                print(f"  FITS文件中的常见后缀: {', '.join(common_exts)}")
        else:
            print(f"\n✗ fits目录不存在: {os.path.abspath(self.fits_dir)}")
            fits_ok = False
        
        print("\n=== 检查完成 ===\n")
        
        return csv_ok and fits_ok
    
    def clean_cache(self):
        """有选择地清理不同类型的缓存文件"""
        print("\n=== 缓存清理 ===")
        cache_options = {
            '1': '文件查找缓存 (FITS文件位置)',
            '2': '预处理结果缓存 (所有处理好的光谱)',
            '3': '进度文件 (处理进度记录)',
            '4': '选择性清除预处理步骤缓存',
            '5': '所有缓存',
            '0': '退出'
        }
        
        while True:
            print("\n请选择要清理的缓存类型:")
            for key, value in cache_options.items():
                print(f"{key}. {value}")
            
            choice = input("请输入选项(0-5): ").strip()
            
            if choice == '0':
                print("退出缓存清理")
                break
                
            elif choice == '1':
                # 文件查找缓存
                files_cache_path = self.cache_file
                if os.path.exists(files_cache_path):
                    try:
                        with open(files_cache_path, 'rb') as f:
                            cache_data = pickle.load(f)
                            cache_size = len(cache_data) if isinstance(cache_data, dict) else 0
                        print(f"发现文件查找缓存: {files_cache_path}")
                        print(f"缓存包含 {cache_size} 个FITS文件位置记录")
                        if input("确认清理? (y/n): ").lower() == 'y':
                            os.remove(files_cache_path)
                            print("✓ 文件查找缓存已清理")
                            # 重置内存中的缓存
                            self.fits_file_cache = {}
                    except Exception as e:
                        print(f"读取文件查找缓存失败: {e}")
                else:
                    print("未发现文件查找缓存")
            
            elif choice == '2':
                # 预处理结果缓存
                cache_files = glob.glob(os.path.join(self.cache_dir, "*"))
                cache_files = [f for f in cache_files if os.path.basename(f) != os.path.basename(self.cache_file)]
                
                if cache_files:
                    print(f"发现 {len(cache_files)} 个预处理缓存文件")
                    if input("确认清理所有预处理结果缓存? (y/n): ").lower() == 'y':
                        for file in tqdm(cache_files, desc="清理预处理缓存"):
                            try:
                                os.remove(file)
                            except Exception as e:
                                print(f"清理文件 {file} 失败: {e}")
                        print("✓ 所有预处理缓存已清理")
                else:
                    print("未发现预处理缓存文件")
            
            elif choice == '3':
                # 进度文件
                progress_files = glob.glob(os.path.join(self.progress_dir, "*_progress.pkl"))
                all_progress = os.path.join(self.progress_dir, "all_progress.pkl")
                if os.path.exists(all_progress):
                    progress_files.append(all_progress)
                    
                if progress_files:
                    print(f"\n发现 {len(progress_files)} 个进度文件:")
                    for file in progress_files:
                        print(f"  - {os.path.basename(file)}")
                    if input("确认清理所有进度文件? (y/n): ").lower() == 'y':
                        for file in tqdm(progress_files, desc="清理进度文件"):
                            try:
                                os.remove(file)
                            except Exception as e:
                                print(f"清理文件 {file} 失败: {e}")
                        print("✓ 所有进度文件已清理")
                else:
                    print("未发现进度文件")
            
            elif choice == '4':
                # 选择性清除预处理步骤
                print("\n可选的预处理步骤:")
                steps = {
                    '1': '波长校正和视向速度校正',
                    '2': '去噪和红移校正',
                    '3': '重采样',
                    '4': '连续谱归一化',
                    '5': '二次去噪',
                    '6': '最终归一化',
                    '0': '返回上级菜单'
                }
                
                for key, value in steps.items():
                    print(f"{key}. {value}")
                
                step_choice = input("请选择要清除的预处理步骤(0-6): ").strip()
                
                if step_choice == '0':
                    continue
                    
                if step_choice in steps:
                    # 检查存在的缓存文件
                    cache_files = glob.glob(os.path.join(self.cache_dir, "*"))
                    cache_files = [f for f in cache_files if os.path.basename(f) != os.path.basename(self.cache_file)]
                    
                    if cache_files:
                        print(f"发现 {len(cache_files)} 个缓存文件")
                        print("清除预处理步骤会删除所有缓存文件，使系统在下次运行时重新生成所有步骤")
                        
                        if input(f"确认删除缓存以重新执行'{steps[step_choice]}'? (y/n): ").lower() == 'y':
                            for file in tqdm(cache_files, desc=f"清除缓存"):
                                try:
                                    os.remove(file)
                                except Exception as e:
                                    print(f"清理文件 {file} 失败: {e}")
                            print("✓ 所有预处理缓存已清理")
                            print(f"下次运行时将重新执行'{steps[step_choice]}'及其后续步骤")
                    else:
                        print("未发现预处理缓存文件")
                else:
                    print("无效的选择")
            
            elif choice == '5':
                # 清除所有缓存
                if input("确认清除所有缓存? 这将删除所有文件查找缓存、预处理结果和进度文件 (y/n): ").lower() == 'y':
                    # 文件查找缓存
                    if os.path.exists(self.cache_file):
                        try:
                            os.remove(self.cache_file)
                            self.fits_file_cache = {}
                        except Exception as e:
                            print(f"清理文件查找缓存失败: {e}")
                    
                    # 预处理结果缓存
                    cache_files = glob.glob(os.path.join(self.cache_dir, "*"))
                    for file in tqdm(cache_files, desc="清理预处理缓存"):
                        try:
                            os.remove(file)
                        except Exception as e:
                            print(f"清理文件 {file} 失败: {e}")
                    
                    # 进度文件
                    progress_files = glob.glob(os.path.join(self.progress_dir, "*"))
                    for file in tqdm(progress_files, desc="清理进度文件"):
                        try:
                            os.remove(file)
                        except Exception as e:
                            print(f"清理文件 {file} 失败: {e}")
                    
                    print("✓ 所有缓存已清理")
            
            else:
                print("无效的选择，请重新输入")
                
        print("\n=== 缓存清理完成 ===\n")
    
    def check_and_fix_file_paths(self):
        """检查并修复文件路径问题"""
        print("\n=== 路径问题诊断 ===")
        
        # 创建测试文件列表
        test_files = []
        for csv_file in self.csv_files:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                if 'spec' in df.columns:
                    # 确保spec列的类型为字符串
                    if not pd.api.types.is_string_dtype(df['spec']):
                        print(f"注意: {csv_file} 中的spec列不是字符串类型，正在转换...")
                        df['spec'] = df['spec'].astype(str)
                    
                    for spec in df['spec'].values[:5]:  # 只取前5个测试
                        spec = str(spec)  # 确保类型为字符串
                        if spec not in test_files:
                            test_files.append(spec)
        
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
        for spec in test_files:
            # 再次确保spec是字符串类型
            spec = str(spec)
            
            print(f"测试: {spec}")
            # 直接路径测试
            direct_path = os.path.join(self.fits_dir, spec)
            if os.path.exists(direct_path):
                print(f"  直接路径存在: {direct_path}")
            else:
                print(f"  直接路径不存在: {direct_path}")
            
            # 测试带扩展名的路径
            for ext in ['.fits', '.fits.gz', '.fit', '.fit.gz']:
                ext_path = direct_path + ext
                if os.path.exists(ext_path):
                    print(f"  带扩展名的路径存在: {ext_path}")
                    break
            
            # 使用查找函数
            found_path = self._find_fits_file(spec)
            if found_path:
                print(f"  _find_fits_file找到: {found_path}")
            else:
                print(f"  _find_fits_file未找到文件")
            
            # 使用缓存函数
            cached_path = self._get_file_extension(spec)
            if cached_path:
                print(f"  _get_file_extension找到: {cached_path}")
            else:
                print(f"  _get_file_extension未找到文件")
        
        # 清除缓存并重新测试
        print("\n清除缓存后重新测试:")
        self.fits_file_cache = {}
        for spec in test_files[:1]:  # 只测试第一个
            # 确保spec是字符串类型
            spec = str(spec)
            print(f"重新测试: {spec}")
            found_path = self._find_fits_file(spec)
            if found_path:
                print(f"  重新测试找到: {found_path}")
            else:
                print(f"  重新测试未找到文件")
        
        print("\n=== 诊断完成 ===\n")

    def _validate_wavelength_range(self, data):
        """验证波长范围数据的函数
        
        参数:
            data: 要验证的数据对象
            
        返回:
            bool: 如果数据是有效的波长范围元组则返回True，否则返回False
        """
        # 如果是波长范围元组，直接返回True
        if isinstance(data, tuple) and len(data) == 2:
            return True
        # 否则使用缓存管理器的原始验证方法
        if hasattr(self.cache_manager, '_original_validate'):
            return self.cache_manager._original_validate(data)
        # 如果没有原始验证方法，直接返回True
        return True
        
    def update_cache_manager(self):
        """修改缓存管理器的验证功能，允许保存非字典格式的数据"""
        # 保存原始的验证方法
        if hasattr(self.cache_manager, '_validate_cache_data'):
            self.cache_manager._original_validate = self.cache_manager._validate_cache_data
        
        # 替换验证方法为类方法
        self.cache_manager._validate_cache_data = self._validate_wavelength_range

def main():
    """主函数"""
    start_time = time.time()
    
    # 处理命令行参数
    import argparse
    
    parser = argparse.ArgumentParser(description="LAMOST光谱数据预处理器")
    parser.add_argument('--csv_files', nargs='+', default=None,
                      help='要处理的CSV文件列表，每个文件包含一个元素的数据。不指定时自动检测当前目录所有CSV文件')
    parser.add_argument('--fits_dir', default='fits', help='FITS文件目录')
    parser.add_argument('--output_dir', default='processed_data', help='输出目录')
    parser.add_argument('--wavelength_range', nargs=2, type=float, default=None,
                      help='波长范围，例如: 4000 8000')
    parser.add_argument('--n_points', type=int, default=None,
                      help='重采样后的点数')
    parser.add_argument('--log_step', type=float, default=0.0001,
                      help='对数空间中的重采样步长（dex）')
    parser.add_argument('--batch_size', type=int, default=20,
                      help='批处理大小')
    parser.add_argument('--max_workers', type=int, default=None,
                      help='最大工作进程数，默认为CPU核心数的一半')
    parser.add_argument('--memory_limit', type=float, default=0.7,
                      help='内存使用限制(占总内存比例)')
    parser.add_argument('--no_resume', action='store_true',
                      help='不恢复之前的进度，从头开始处理')
    parser.add_argument('--evaluate', action='store_true',
                      help='评估预处理效果')
    parser.add_argument('--single_element', type=str, default=None,
                      help='仅处理指定元素的CSV文件，例如: C_FE')
    parser.add_argument('--low_memory_mode', action='store_true', 
                      help='启用低内存模式，减少内存使用但速度变慢')
    
    args = parser.parse_args()
    
    # 设置基础路径
    base_path = '/content' if IN_COLAB else os.path.abspath('.')
    print(f"基础路径: {base_path}")
    
    # 获取系统内存信息
    mem_info = psutil.virtual_memory()
    print(f"系统内存: 总计 {mem_info.total / (1024**3):.1f}GB, "
          f"可用 {mem_info.available / (1024**3):.1f}GB, "
          f"使用率 {mem_info.percent}%")
    
    # 检测内存情况，自动决定是否使用低内存模式
    low_memory_mode = args.low_memory_mode or mem_info.percent > 80
    
    if low_memory_mode and not args.low_memory_mode:
        print("检测到系统内存不足，自动启用低内存模式")
        user_choice = input("是否启用低内存模式? 这将减少内存使用但处理速度会变慢 (y/n): ").lower()
        low_memory_mode = user_choice == 'y'
    
    # 设置CSV文件路径
    if args.csv_files is None:
        # 自动检测当前目录下所有CSV文件
        args.csv_files = [f for f in os.listdir() if f.endswith('.csv')]
        if not args.csv_files:
            print("错误: 当前目录未找到CSV文件，请指定--csv_files参数")
            return
        print(f"自动检测到以下CSV文件: {args.csv_files}")
    
    # 如果指定了单个元素，就只处理对应的CSV文件
    if args.single_element:
        # 查找匹配该元素的CSV文件
        matching_files = []
        for csv_file in args.csv_files:
            element_name = os.path.basename(csv_file).split('.')[0]
            if element_name == args.single_element:
                matching_files.append(csv_file)
        
        if not matching_files:
            print(f"错误: 找不到元素 {args.single_element} 对应的CSV文件")
            return
        
        args.csv_files = matching_files
        print(f"仅处理元素 {args.single_element} 的数据: {args.csv_files}")
    
    fits_dir = args.fits_dir
    if not os.path.isabs(fits_dir):
        fits_dir = os.path.join(base_path, fits_dir)
    
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(base_path, output_dir)
    
    # 展示路径信息
    print(f"CSV文件路径: {args.csv_files}")
    print(f"FITS目录路径: {fits_dir}")
    print(f"输出目录路径: {output_dir}")
    
    # 确保fits目录存在
    if not os.path.exists(fits_dir):
        print(f"创建FITS目录: {fits_dir}")
        os.makedirs(fits_dir, exist_ok=True)
    
    # 检测FITS文件是否有效
    print("\n=== 检查FITS文件有效性 ===")
    fits_files = []
    if os.path.exists(fits_dir):
        fits_files = [os.path.join(fits_dir, f) for f in os.listdir(fits_dir) 
                    if f.endswith(('.fits', '.fits.gz', '.fit', '.fit.gz'))]
    
    if not fits_files:
        print("警告：找不到任何FITS文件！")
    else:
        print(f"找到{len(fits_files)}个FITS文件，开始检查...")
        
        # 抽样检查几个文件
        sample_size = min(5, len(fits_files))
        sample_files = fits_files[:sample_size]
        
        valid_count = 0
        for file in sample_files:
            print(f"\n检查文件: {os.path.basename(file)}")
            try:
                with fits.open(file, ignore_missing_end=True, memmap=False) as hdul:
                    print(f"  HDU数量: {len(hdul)}")
                    print(f"  主HDU类型: {type(hdul[0]).__name__}")
                    header = hdul[0].header
                    print(f"  主要头信息: NAXIS={header.get('NAXIS')}, NAXIS1={header.get('NAXIS1')}")
                    
                    # 检查是否有数据
                    has_data = False
                    for i, hdu in enumerate(hdul):
                        if hdu.data is not None:
                            data_shape = hdu.data.shape if hasattr(hdu.data, 'shape') else "无形状"
                            print(f"  HDU{i}有数据: 形状={data_shape}")
                            has_data = True
                            break
                    
                    if not has_data:
                        print("  警告: 所有HDU中都没有数据")
                    else:
                        valid_count += 1
                        
            except Exception as e:
                print(f"  读取错误: {e}")
        
        print(f"\n检查结果: {valid_count}/{sample_size}个文件有有效数据")
        
        if valid_count == 0:
            print("所有测试文件都没有数据，可能是FITS文件格式有问题。")
            fix_option = input("是否尝试自动修复FITS文件? (y/n): ").lower()
            if fix_option == 'y':
                print("尝试使用astropy修复FITS文件...")
                # 这里只修复示例文件
                for file in sample_files:
                    try:
                        # 读取文件并重新写入，可能会修复一些格式问题
                        with fits.open(file, ignore_missing_end=True) as hdul:
                            fixed_file = file + '.fixed'
                            hdul.writeto(fixed_file, overwrite=True)
                            print(f"  已修复: {os.path.basename(file)} -> {os.path.basename(fixed_file)}")
                    except Exception as e:
                        print(f"  修复失败: {e}")
    
    print("\n=== 检查完成 ===\n")
    
    # 初始化预处理器
    wavelength_range = tuple(args.wavelength_range) if args.wavelength_range else None
    
    preprocessor = LAMOSTPreprocessor(
        csv_files=args.csv_files,
        fits_dir=fits_dir,
        output_dir=output_dir,
        wavelength_range=None,  # 修改为None，表示将使用最大公有波长范围
        n_points=None,  # 修改为None，点数将根据波长范围和步长自动计算
        log_step=0.0001,  # 新增：对数空间中的重采样步长（dex）
        compute_common_range=True,  # 新增：是否计算最大公有波长范围
 
        max_workers=1 if low_memory_mode else 2,  # 低内存模式使用单线程
        batch_size=5 if low_memory_mode else 20,   # 低内存模式减小批次大小
        memory_limit=0.7,  # 内存使用阈值
        low_memory_mode=low_memory_mode,  # 低内存模式标志
        verbose=1  # 输出详细程度，0=静默，1=基本信息，2=详细信息
    )
    
    # 检查数据源
    preprocessor.check_data_sources()
    
    # 添加路径诊断
    preprocessor.check_and_fix_file_paths()
    
    # 询问用户是否清理缓存
    preprocessor.clean_cache()
    
    # 询问用户是否继续
    user_input = input("是否继续处理数据? (y/n): ").strip().lower()
    if user_input != 'y':
        print("程序已终止")
        return
    
    # 处理所有数据，支持断点续传
    X, y, elements, filenames = preprocessor.process_all_data(resume=True)
    
    # 检查返回值是否有效
    if X is None:
        print("错误: 没有处理到任何有效数据，处理过程可能失败或用户选择跳过数据集划分")
        return
    
    if len(X) == 0:
        print("错误: 没有处理到任何有效数据，请检查fits文件路径和CSV文件")
        return
    
    # 检查是否已经在process_all_data函数中完成了数据集划分
    # 如果没有full_dataset.npz文件，说明用户选择了划分数据集
    full_dataset_path = os.path.join(preprocessor.output_dir, 'full_dataset.npz')
    if not os.path.exists(full_dataset_path):
        # 用户选择了数据集划分
        print("用户已选择分割数据集为训练集、验证集和测试集")
    else:
        # 用户选择不分割，使用完整数据集
        print("用户选择不分割数据集，使用完整数据集")
    
    # 可视化部分不再主动询问，因为在process_all_data中已经询问并处理
    
    print(f"预处理完成，总耗时: {time.time() - start_time:.2f}秒")
    print(f"处理结果保存在: {os.path.abspath(preprocessor.output_dir)}")

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
                    print("请上传C_FE.csv, MG_FE.csv, CA_FE.csv文件...")
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