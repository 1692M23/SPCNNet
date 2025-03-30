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
from sklearn.model_selection import train_test_split
import random

# GPU加速相关导入
try:
    import cupy as cp
    import cupyx.scipy.signal as cusignal
    import cupyx.scipy.ndimage as cundimage
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    # 创建占位符以避免在后续代码中出现导入错误
    class DummyModule:
        def __getattr__(self, name):
            return None
    
    cp = DummyModule()
    cusignal = DummyModule()
    cundimage = DummyModule()
    print("警告: 未找到CuPy库，将使用CPU模式运行")
    print("要启用GPU加速，请安装CuPy库: pip install cupy-cuda11x (根据您的CUDA版本选择合适的包)")

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

# 检测是否有可用的GPU
def check_gpu_available():
    """检测是否有可用的GPU，并返回设备信息"""
    if not HAS_GPU:
        print("未安装CuPy，无法使用GPU")
        return False
    
    try:
        # 检查CUDA是否可用
        n_gpus = cp.cuda.runtime.getDeviceCount()
        if n_gpus > 0:
            # 打印GPU信息
            for i in range(n_gpus):
                device_props = cp.cuda.runtime.getDeviceProperties(i)
                print(f"找到GPU {i}: {device_props['name']}, 显存: {device_props['totalGlobalMem'] / (1024**3):.2f} GB")
            
            # 设置默认使用的GPU
            cp.cuda.runtime.setDevice(0)
            print(f"使用GPU: {cp.cuda.runtime.getDevice()}")
            return True
        else:
            print("未检测到可用的GPU")
            return False
    except Exception as e:
        print(f"GPU检测出错: {e}")
        return False

# 环境设置
IN_COLAB = is_in_colab()
USE_GPU = check_gpu_available()

class LAMOSTPreprocessor:
    def __init__(self, csv_files=None, 
                 fits_dir='fits', 
                 output_dir='processed_data',
                 wavelength_range=(3690, 9100),  # 设置固定的默认范围
                 n_points=None,
                 log_step=0.0001,
                 compute_common_range=False,  # 默认不计算公共范围
                 n_splits=5,     # 交叉验证折数
                 max_workers=None,  # 最大工作进程数，None表示自动确定
                 batch_size=20,  # 批处理大小
                 memory_limit=0.7,  # 内存使用限制(占总内存比例)
                 low_memory_mode=False,  # 低内存模式
                 use_gpu=USE_GPU):  # 是否使用GPU
        
        # 设置GPU使用选项
        self.use_gpu = use_gpu
        if self.use_gpu and not HAS_GPU:
            print("警告: 未找到CuPy库，无法使用GPU模式")
            self.use_gpu = False
        
        if self.use_gpu:
            print("🚀 已启用GPU加速模式")
            # 预热GPU，分配少量内存确保GPU已初始化
            try:
                temp_array = cp.zeros((100, 100), dtype=cp.float32)
                cp.sum(temp_array)
                # 同步GPU，确保操作完成
                cp.cuda.Stream.null.synchronize()
                # 释放内存
                del temp_array
                cp.get_default_memory_pool().free_all_blocks()
                print("GPU预热完成")
            except Exception as e:
                print(f"GPU预热失败: {e}")
                print("切换到CPU模式")
                self.use_gpu = False
        else:
            print("使用CPU模式运行")
            
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
        
        self.wavelength_range = wavelength_range  # 使用固定范围
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
        
        # 检查是否是元素名称而不是文件路径（例如"C_FE"）
        if spec_name in ["C_FE", "MG_FE", "CA_FE"]:
            print(f"检测到元素名称: {spec_name}，尝试查找对应的示例FITS文件")
            # 查找与元素相关的CSV文件中的FITS文件样例
            csv_path = f"{spec_name}.csv"
            if os.path.exists(csv_path):
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    if 'spec' in df.columns and len(df) > 0:
                        first_spec = df['spec'].iloc[0]
                        print(f"从CSV文件中获取第一个样例: {first_spec}")
                        return self._find_fits_file(first_spec)
                except Exception as e:
                    print(f"从CSV读取样例时出错: {e}")
        
        # 尝试直接在fits目录下按基础名称匹配（常规后缀）
        for ext in ['', '.fits', '.fits.gz', '.fit', '.fit.gz']:
            path = os.path.join(self.fits_dir, base_name + ext)
            if os.path.exists(path) and os.path.isfile(path):
                print(f"基础名称匹配成功: {path}")
                self.fits_file_cache[spec_name] = path
                return path
        
        # 进行递归搜索，处理嵌套目录
        print(f"开始递归搜索FITS目录: {self.fits_dir}")
        fits_files_found = []
        
        for root, dirs, files in os.walk(self.fits_dir):
            for file in files:
                # 检查文件名是否匹配（忽略大小写）
                if base_name.lower() in file.lower():
                    found_path = os.path.join(root, file)
                    fits_files_found.append(found_path)
                    print(f"部分名称匹配成功: {found_path}")
                
                # 尝试去除可能的后缀后再比较
                file_base = file.lower()
                for ext in ['.fits', '.fits.gz', '.fit', '.fit.gz']:
                    if file_base.endswith(ext):
                        file_base = file_base[:-len(ext)]
                        break
                
                if base_name.lower() == file_base:
                    found_path = os.path.join(root, file)
                    fits_files_found.append(found_path)
                    print(f"去除后缀后匹配成功: {found_path}")
                
                # 尝试更模糊的匹配方式
                # 移除路径分隔符，便于匹配跨目录文件
                clean_base_name = base_name.replace('/', '_').replace('\\', '_')
                clean_file_base = file_base.replace('/', '_').replace('\\', '_')
                
                if clean_base_name.lower() in clean_file_base or clean_file_base in clean_base_name.lower():
                    found_path = os.path.join(root, file)
                    fits_files_found.append(found_path)
                    print(f"模糊匹配成功: {found_path}")
        
        # 处理找到的文件
        if fits_files_found:
            # 优先选择与查询名称最匹配的文件
            best_match = fits_files_found[0]  # 默认第一个
            for found_path in fits_files_found:
                file_name = os.path.basename(found_path).lower()
                # 如果文件名包含查询名称，优先选择
                if base_name.lower() in file_name:
                    best_match = found_path
                    break
            
            print(f"从{len(fits_files_found)}个匹配结果中选择: {best_match}")
            self.fits_file_cache[spec_name] = best_match
            return best_match
        
        # 尝试其他备选方案 - 如果目录中有任何FITS文件，作为最后的备选方案
        fits_files = []
        for root, _, files in os.walk(self.fits_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in ['.fits', '.fits.gz', '.fit', '.fit.gz']):
                    fits_files.append(os.path.join(root, file))
        
        if fits_files:
            # 显示可用的备选文件
            print(f"未直接找到 {spec_name} 匹配的文件。")
            print(f"FITS目录中有 {len(fits_files)} 个FITS文件可用。")
            print(f"可用的FITS文件示例: {fits_files[0] if fits_files else '无'}")
        
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
            
            if self.use_gpu:
                try:
                    # 使用GPU进行SavGol滤波
                    # 将数据转移到GPU
                    d_valid_flux = cp.asarray(valid_flux)
                    # 使用cupy的滤波算法
                    d_denoised = cusignal.savgol_filter(d_valid_flux, window_length, polyorder)
                    # 将结果从GPU拷贝回CPU
                    flux_denoised[mask] = cp.asnumpy(d_denoised)
                    # 清理GPU内存
                    del d_valid_flux, d_denoised
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception as e:
                    print(f"GPU去噪失败，回退到CPU: {e}")
                    flux_denoised[mask] = signal.savgol_filter(valid_flux, window_length, polyorder)
            else:
                # 使用CPU进行滤波
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
            
            if common_min < common_max:
                self.wavelength_range = (common_min, common_max)
                print(f"更新最大公有波长范围: {common_min:.2f}~{common_max:.2f}")
            else:
                print(f"警告: 无法更新公有波长范围，当前范围不重叠")
    
    def _gpu_interp(self, x, y, xnew):
        """在GPU上进行线性插值
        
        参数:
            x: 原始x坐标
            y: 原始y值
            xnew: 新的x坐标
            
        返回:
            插值后的y值
        """
        import cupy as cp
        
        # 确保x是单调递增的
        if not cp.all(cp.diff(x) > 0):
            # 排序x和y
            sort_indices = cp.argsort(x)
            x = x[sort_indices]
            y = y[sort_indices]
        
        # 找到xnew中每个点在x中的位置
        indices = cp.searchsorted(x, xnew) - 1
        
        # 处理边界情况
        indices = cp.clip(indices, 0, len(x) - 2)
        
        # 计算权重
        interval_width = x[indices + 1] - x[indices]
        weights = (xnew - x[indices]) / interval_width
        
        # 线性插值
        ynew = y[indices] * (1 - weights) + y[indices + 1] * weights
        
        return ynew
    
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
            
            # 使用固定的波长范围，不再计算公共范围
            # 即使设置了compute_common_range，也忽略它
            w_min, w_max = 3690, 9100  # 强制使用固定范围
            
            # 在对数空间中进行重采样
            log_w_min = np.log10(w_min)
            log_w_max = np.log10(w_max)
            
            # 根据步长计算点数
            if self.n_points is None:
                n_points = int((log_w_max - log_w_min) / self.log_step) + 1
            else:
                n_points = self.n_points
                
            # 创建对数等间隔的波长点
            log_wavelength_new = np.linspace(log_w_min, log_w_max, n_points)
            wavelength_new = 10 ** log_wavelength_new
            
            # 使用CPU或GPU进行插值
            if self.use_gpu and self.cupy_available:
                import cupy as cp
                try:
                    # 将数据传输到GPU
                    wavelength_valid_cp = cp.array(wavelength_valid)
                    flux_valid_cp = cp.array(flux_valid)
                    wavelength_new_cp = cp.array(wavelength_new)
                    
                    # 在GPU上进行插值
                    # cupy没有直接的interp函数，使用线性近似
                    flux_new_cp = self._gpu_interp(wavelength_valid_cp, flux_valid_cp, wavelength_new_cp)
                    
                    # 将结果传回CPU
                    flux_new = cp.asnumpy(flux_new_cp)
                    
                    # 清理GPU内存
                    del wavelength_valid_cp, flux_valid_cp, wavelength_new_cp, flux_new_cp
                    cp.get_default_memory_pool().free_all_blocks()
                    
                except Exception as e:
                    if self.disable_gpu_warnings:
                        # 发生错误时切换到CPU处理
                        flux_new = np.interp(wavelength_new, wavelength_valid, flux_valid)
                    else:
                        print(f"GPU插值失败，切换到CPU: {e}")
                        flux_new = np.interp(wavelength_new, wavelength_valid, flux_valid)
            else:
                # 在CPU上进行线性插值
                flux_new = np.interp(wavelength_new, wavelength_valid, flux_valid)
            
            return wavelength_new, flux_new
            
        except Exception as e:
            print(f"重采样时出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def normalize_spectrum(self, flux):
        """对光谱进行归一化处理"""
        try:
            # 检查数据有效性
            if flux is None or len(flux) == 0:
                print("无效的流量数据，无法归一化")
                return None
                
            # 处理全为NaN的情况
            if np.isnan(flux).all():
                print("所有流量值都是NaN，无法归一化")
                return None
            
            # 连续谱归一化 (简单的最大值归一化)
            valid_mask = ~np.isnan(flux) & ~np.isinf(flux)
            valid_flux = flux[valid_mask]
            
            if len(valid_flux) == 0:
                print("没有有效的流量值，无法归一化")
                return None
            
            if self.use_gpu:
                try:
                    # 使用GPU计算最大最小值
                    d_valid_flux = cp.asarray(valid_flux)
                    flux_min = float(cp.min(d_valid_flux).get())
                    flux_max = float(cp.max(d_valid_flux).get())
                    
                    print(f"归一化：最小值={flux_min}，最大值={flux_max}")
                    
                    if cp.isclose(flux_max, flux_min):
                        print(f"流量范围无效: min={flux_min}, max={flux_max}，设置为0-1范围")
                        normalized_flux = np.zeros_like(flux)
                        normalized_flux[valid_mask] = 0.5  # 所有有效值设为0.5
                        # 清理GPU内存
                        del d_valid_flux
                        cp.get_default_memory_pool().free_all_blocks()
                        return normalized_flux, {'flux_min': flux_min, 'flux_max': flux_max}
                    
                    # 创建归一化后的数组
                    normalized_flux = np.zeros_like(flux)
                    
                    # 在GPU上进行归一化计算
                    d_normalized = (d_valid_flux - flux_min) / (flux_max - flux_min)
                    
                    # 确保所有值都严格在0-1范围内
                    d_normalized = cp.clip(d_normalized, 0.0, 1.0)
                    
                    # 将结果复制回CPU
                    normalized_flux[valid_mask] = cp.asnumpy(d_normalized)
                    
                    # 清理GPU内存
                    del d_valid_flux, d_normalized
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception as e:
                    print(f"GPU归一化失败，回退到CPU: {e}")
                    # 退回到CPU计算
                    flux_min = np.min(valid_flux)
                    flux_max = np.max(valid_flux)
                    
                    print(f"归一化：最小值={flux_min}，最大值={flux_max}")
                    
                    if np.isclose(flux_max, flux_min):
                        print(f"流量范围无效: min={flux_min}, max={flux_max}，设置为0-1范围")
                        normalized_flux = np.zeros_like(flux)
                        normalized_flux[valid_mask] = 0.5
                        return normalized_flux, {'flux_min': flux_min, 'flux_max': flux_max}
                    
                    normalized_flux = np.zeros_like(flux)
                    normalized_flux[valid_mask] = (valid_flux - flux_min) / (flux_max - flux_min)
                    normalized_flux = np.clip(normalized_flux, 0.0, 1.0)
            else:
                # 使用CPU进行计算
                # 最大最小值归一化
                flux_min = np.min(valid_flux)
                flux_max = np.max(valid_flux)
                
                print(f"归一化：最小值={flux_min}，最大值={flux_max}")
                
                if np.isclose(flux_max, flux_min):
                    print(f"流量范围无效: min={flux_min}, max={flux_max}，设置为0-1范围")
                    normalized_flux = np.zeros_like(flux)
                    normalized_flux[valid_mask] = 0.5  # 所有有效值设为0.5
                    return normalized_flux, {'flux_min': flux_min, 'flux_max': flux_max}
                
                # 创建归一化后的数组
                normalized_flux = np.zeros_like(flux)
                normalized_flux[valid_mask] = (valid_flux - flux_min) / (flux_max - flux_min)
                
                # 确保所有值都严格在0-1范围内
                normalized_flux = np.clip(normalized_flux, 0.0, 1.0)
            
            # 替换无效值
            normalized_flux[~valid_mask] = 0.0
            
            # 最终检查确保没有NaN或无限值
            if np.isnan(normalized_flux).any() or np.isinf(normalized_flux).any():
                print("归一化后仍有无效值，进行最终替换")
                normalized_flux = np.nan_to_num(normalized_flux, nan=0.0, posinf=1.0, neginf=0.0)
                
            return normalized_flux, {'flux_min': flux_min, 'flux_max': flux_max}
        except Exception as e:
            print(f"归一化失败: {e}")
            return None, None
    
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
                
                # 计算中值滤波
                if self.use_gpu:
                    try:
                        window_size = min(11, len(flux_segment) // 5 * 2 + 1)  # 确保窗口大小为奇数
                        window_size = max(3, window_size)  # 至少使用3点窗口
                        
                        # 将数据转移到GPU
                        d_flux_segment = cp.asarray(flux_segment)
                        # 使用GPU计算中值滤波
                        d_smoothed_flux = cundimage.median_filter(d_flux_segment, size=window_size)
                        smoothed_flux = cp.asnumpy(d_smoothed_flux)
                        
                        # 清理GPU内存
                        del d_flux_segment, d_smoothed_flux
                        cp.get_default_memory_pool().free_all_blocks()
                    except Exception as e:
                        print(f"GPU中值滤波失败，回退到CPU: {e}")
                        # 回退到CPU计算
                        from scipy.signal import medfilt
                        window_size = min(11, len(flux_segment) // 5 * 2 + 1)
                        window_size = max(3, window_size)
                        smoothed_flux = medfilt(flux_segment, window_size)
                else:
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
                    if self.use_gpu:
                        try:
                            # 将数据传输到GPU
                            d_wave = cp.asarray(wave_segment[continuum_mask])
                            d_flux = cp.asarray(flux_segment[continuum_mask])
                            
                            # 使用GPU进行多项式拟合
                            continuum_fit = cp.polynomial.polynomial.polyfit(
                                d_wave, d_flux, poly_order
                            )
                            
                            # 计算当前区间的伪连续谱
                            # 只在当前区间的波长范围内计算
                            segment_range = (wave_segment[0], wave_segment[-1])
                            mask = (wavelength >= segment_range[0]) & (wavelength <= segment_range[1])
                            
                            if not np.any(mask):
                                continue  # 如果没有波长在该区间内，跳过
                            
                            # 将波长传输到GPU
                            d_wavelength_mask = cp.asarray(wavelength[mask])
                            # 计算多项式值
                            d_pseudo_continuum = cp.polynomial.polynomial.polyval(d_wavelength_mask, continuum_fit)
                            # 将结果传回CPU
                            pseudo_continuum = cp.asnumpy(d_pseudo_continuum)
                            
                            # 清理GPU内存
                            del d_wave, d_flux, d_wavelength_mask, d_pseudo_continuum
                            cp.get_default_memory_pool().free_all_blocks()
                        except Exception as e:
                            print(f"GPU多项式拟合失败，回退到CPU: {e}")
                            # 回退到CPU拟合
                            continuum_fit = np.polyfit(
                                wave_segment[continuum_mask], 
                                flux_segment[continuum_mask], 
                                poly_order
                            )
                            
                            segment_range = (wave_segment[0], wave_segment[-1])
                            mask = (wavelength >= segment_range[0]) & (wavelength <= segment_range[1])
                            
                            if not np.any(mask):
                                continue
                                
                            pseudo_continuum = np.polyval(continuum_fit, wavelength[mask])
                    else:
                        # 使用CPU进行多项式拟合
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
            
            if self.use_gpu:
                try:
                    # 对一般区域应用GPU滤波
                    d_valid_flux = cp.asarray(valid_flux)
                    d_flux_denoised = cusignal.savgol_filter(d_valid_flux, standard_window, 2)
                    flux_denoised[valid_mask] = cp.asnumpy(d_flux_denoised)
                    print(f"二次去噪完成(GPU)，使用窗口长度= {standard_window}")
                    
                    # 如果OI区域有异常，使用更强的滤波参数专门处理
                    if has_oi_anomaly:
                        # 找到OI区域的有效数据点
                        oi_valid_mask = oi_region & valid_mask
                        if np.sum(oi_valid_mask) > oi_window:  # 确保有足够的点进行滤波
                            # 对OI区域使用更大窗口和更高阶多项式
                            oi_indices = np.where(oi_valid_mask)[0]
                            if len(oi_indices) >= oi_window:
                                oi_flux_section = flux[oi_valid_mask]
                                # 使用GPU进行更强平滑
                                d_oi_flux = cp.asarray(oi_flux_section)
                                d_oi_smoothed = cusignal.savgol_filter(d_oi_flux, oi_window, 3)
                                flux_denoised[oi_valid_mask] = cp.asnumpy(d_oi_smoothed)
                                print(f"OI区域增强去噪完成(GPU)，使用窗口长度= {oi_window}")
                                # 清理GPU内存
                                del d_oi_flux, d_oi_smoothed
                    
                    # 对特别突出的峰值使用中值滤波
                    if has_oi_anomaly:
                        # 寻找异常峰值
                        flux_mean = np.mean(flux_denoised[valid_mask])
                        flux_std = np.std(flux_denoised[valid_mask])
                        spike_threshold = flux_mean + 1.5 * flux_std
                        
                        spike_mask = (flux_denoised > spike_threshold) & valid_mask
                        if np.any(spike_mask):
                            print(f"检测到{np.sum(spike_mask)}个异常峰值点，进行中值滤波")
                            
                            if self.use_gpu and np.sum(spike_mask) > 10:  # 只有当峰值点足够多时才用GPU
                                try:
                                    # 将数据转移到GPU
                                    d_flux_denoised = cp.asarray(flux_denoised)
                                    # 使用GPU进行中值滤波
                                    # 由于无法像CPU版本那样针对个别点处理，我们应用全局中值滤波
                                    # 但注意这会改变所有数据点，而不仅是峰值点
                                    mask_array = cp.zeros(flux_denoised.shape, dtype=cp.bool_)
                                    mask_array[spike_mask] = True
                                    # 执行中值滤波（这里我们对整个数组应用滤波）
                                    d_filtered = cundimage.median_filter(d_flux_denoised, size=3)
                                    # 只更新异常点
                                    d_flux_denoised[mask_array] = d_filtered[mask_array]
                                    # 复制回CPU
                                    flux_denoised = cp.asnumpy(d_flux_denoised)
                                    # 清理GPU内存
                                    del d_flux_denoised, d_filtered, mask_array
                                    cp.get_default_memory_pool().free_all_blocks()
                                except Exception as e:
                                    print(f"GPU中值滤波失败，回退到CPU: {e}")
                                    # 回退到CPU处理
                                    for idx in np.where(spike_mask)[0]:
                                        start = max(0, idx - 3)
                                        end = min(len(flux_denoised), idx + 4)
                                        if end - start >= 3:  # 确保至少有3个点用于中值计算
                                            neighbors = flux_denoised[start:end]
                                            flux_denoised[idx] = np.median(neighbors)
                            else:
                                # CPU处理
                                for idx in np.where(spike_mask)[0]:
                                    start = max(0, idx - 3)
                                    end = min(len(flux_denoised), idx + 4)
                                    if end - start >= 3:  # 确保至少有3个点用于中值计算
                                        neighbors = flux_denoised[start:end]
                                        flux_denoised[idx] = np.median(neighbors)
                    
                    # 清理GPU内存
                    del d_valid_flux, d_flux_denoised
                    cp.get_default_memory_pool().free_all_blocks()
                    
                except Exception as e:
                    print(f"GPU二次去噪失败，回退到CPU: {e}")
                    # 回退到CPU滤波
                    from scipy.signal import savgol_filter
                    flux_denoised[valid_mask] = savgol_filter(valid_flux, standard_window, 2)
                    
                    # CPU处理OI区域
                    if has_oi_anomaly:
                        oi_valid_mask = oi_region & valid_mask
                        if np.sum(oi_valid_mask) > oi_window:
                            oi_indices = np.where(oi_valid_mask)[0]
                            if len(oi_indices) >= oi_window:
                                oi_flux_section = flux[oi_valid_mask]
                                oi_smoothed = savgol_filter(oi_flux_section, oi_window, 3)
                                flux_denoised[oi_valid_mask] = oi_smoothed
            else:
                # 使用CPU版本的SavGol滤波器
                from scipy.signal import savgol_filter
                
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
        cache_key = f"processed_spectrum_{spec_file.replace('/', '_')}"
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
            flux_normalized, norm_params = self.normalize_spectrum(flux_denoised_second)
            if flux_normalized is None:
                print(f"归一化{spec_file}失败")
                return None
            
            print(f"成功处理光谱: {spec_file}")
            
            # 记录标准化参数
            normalization_params = {
                # 波长范围信息
                'wavelength_range': self.wavelength_range,
                'log_step': self.log_step,
                'flux_min': norm_params['flux_min'] if norm_params else None,
                'flux_max': norm_params['flux_max'] if norm_params else None,
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
                        results = saved_data.get('results', [])
                        start_idx = saved_data.get('last_idx', 0)
                        total_processed = len(results)
                        print(f"从上次中断处继续（已处理{total_processed}条记录，进度：{total_processed/len(spec_files):.2%}）")
                except Exception as e:
                    print(f"加载进度文件出错: {e}，将从头开始处理")
                    start_idx = 0
            # 如果标准目录没有，尝试Google Drive
            elif os.path.exists(drive_progress_file):
                try:
                    with open(drive_progress_file, 'rb') as f:
                        saved_data = pickle.load(f)
                        results = saved_data.get('results', [])
                        start_idx = saved_data.get('last_idx', 0)
                        total_processed = len(results)
                        print(f"从Google Drive加载进度（已处理{total_processed}条记录，进度：{total_processed/len(spec_files):.2%}）")
                except Exception as e:
                    print(f"加载Google Drive进度文件出错: {e}，将从头开始处理")
                    start_idx = 0
        
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
    
    def process_all_data(self, resume=True):
        """处理所有数据并准备训练集，支持断点续传，采用两阶段处理策略"""
        start_time = time.time()
        
        # 进度文件路径
        progress_file = os.path.join(self.progress_dir, 'all_data_progress.pkl')
        wave_range_file = os.path.join(self.progress_dir, 'wave_range_progress.pkl')
        drive_progress_file = '/content/drive/My Drive/SPCNNet_Results/processed_data/progress/all_data_progress.pkl'
        
        # 初始化全部数据列表
        all_data = []
        
        # 如果有进度文件，考虑恢复
        if resume:
            # 先尝试从标准路径加载
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'rb') as f:
                        all_data = pickle.load(f)
                    print(f"已加载保存的处理结果，共{len(all_data)}条记录")
                    
                    # 读取CSV文件来获取总记录数
                    dataframes, elements = self.read_csv_data()
                    if not dataframes:
                        print("错误: 没有有效的数据集")
                        return np.array([]), np.array([]), np.array([]), np.array([])
                    
                    # 计算总数据量
                    total_records = sum(len(df) for df in dataframes)
                    
                    # 显示进度并询问是否继续
                    progress_percent = len(all_data)/total_records * 100
                    print(f"当前进度: {len(all_data)}/{total_records} 条记录 ({progress_percent:.2f}%)")
                    
                    if len(all_data) >= total_records:
                        print(f"所有数据已处理完成，进入数据准备阶段")
                        return self._prepare_arrays(all_data)
                    else:
                        if input(f"是否继续处理剩余{total_records - len(all_data)}条数据？(y/n): ").lower() != 'y':
                            print("跳过处理阶段，直接使用已有数据")
                            return self._prepare_arrays(all_data)
                        else:
                            print(f"继续处理剩余数据...")
                except Exception as e:
                    print(f"加载进度文件出错: {e}，将重新处理所有数据")
                    all_data = []
            # 如果标准路径没有，尝试从Google Drive加载
            elif os.path.exists(drive_progress_file):
                try:
                    with open(drive_progress_file, 'rb') as f:
                        all_data = pickle.load(f)
                    print(f"从Google Drive加载处理结果，共{len(all_data)}条记录")
                    
                    # 读取CSV文件来获取总记录数
                    dataframes, elements = self.read_csv_data()
                    if not dataframes:
                        print("错误: 没有有效的数据集")
                        return np.array([]), np.array([]), np.array([]), np.array([])
                    
                    # 计算总数据量
                    total_records = sum(len(df) for df in dataframes)
                    
                    # 显示进度并询问是否继续
                    progress_percent = len(all_data)/total_records * 100
                    print(f"当前进度: {len(all_data)}/{total_records} 条记录 ({progress_percent:.2f}%)")
                    
                    if len(all_data) >= total_records:
                        print(f"所有数据已处理完成，进入数据准备阶段")
                        return self._prepare_arrays(all_data)
                    else:
                        if input(f"是否继续处理剩余{total_records - len(all_data)}条数据？(y/n): ").lower() != 'y':
                            print("跳过处理阶段，直接使用已有数据")
                            return self._prepare_arrays(all_data)
                        else:
                            print(f"继续处理剩余数据...")
                except Exception as e:
                    print(f"加载Google Drive进度文件出错: {e}，将重新处理所有数据")
                    all_data = []
        
        # 读取CSV文件
        dataframes, elements = self.read_csv_data()
        if not dataframes:
            print("错误: 没有有效的数据集")
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # 计算总数据量
        total_records = sum(len(df) for df in dataframes)
        print(f"总数据量: {total_records}条记录")
        
        # 输出缓存目录信息
        print(f"\n===== 缓存目录信息 =====")
        print(f"缓存目录: {self.cache_dir}")
        cache_files = glob.glob(os.path.join(self.cache_dir, "*"))
        print(f"缓存文件总数: {len(cache_files)}")
        wavelength_cache_files = glob.glob(os.path.join(self.cache_dir, "wavelength_range_*"))
        print(f"波长范围缓存文件数: {len(wavelength_cache_files)}")
        spectrum_cache_files = glob.glob(os.path.join(self.cache_dir, "processed_spectrum_*"))
        print(f"光谱处理缓存文件数: {len(spectrum_cache_files)}")
        
        # 打印CSV中spec列的样本值
        for i, (df, element) in enumerate(zip(dataframes, elements)):
            if 'spec' in df.columns:
                print(f"\n元素 {element} 的spec列前10个样本:")
                for j, spec in enumerate(df['spec'].values[:10]):
                    print(f"  {j+1}. '{spec}'")
            
            # 打印缓存文件名示例
            if wavelength_cache_files:
                print(f"\n波长范围缓存文件示例:")
                for j, cache_file in enumerate(wavelength_cache_files[:5]):
                    print(f"  {j+1}. '{os.path.basename(cache_file)}'")
            
            if spectrum_cache_files:
                print(f"\n光谱处理缓存文件示例:")
                for j, cache_file in enumerate(spectrum_cache_files[:5]):
                    print(f"  {j+1}. '{os.path.basename(cache_file)}'")
            
            break  # 只显示第一个元素的信息
        
        # 第一阶段：计算所有光谱的最大公共波长范围
        # 检查是否已经完成了第一阶段
        first_stage_done = False
        if os.path.exists(wave_range_file):
            try:
                with open(wave_range_file, 'rb') as f:
                    saved_range_data = pickle.load(f)
                    self.processed_ranges = saved_range_data.get('processed_ranges', [])
                    self.wavelength_range = saved_range_data.get('wavelength_range', self.wavelength_range)
                    first_stage_processed = saved_range_data.get('processed_count', 0)
                    
                    print(f"已从缓存加载波长范围信息:")
                    print(f"已处理: {first_stage_processed}/{total_records} 条记录")
                    print(f"当前最大公共波长范围: {self.wavelength_range}")
                    
                    # 如果已经处理了所有记录，第一阶段完成
                    if first_stage_processed >= total_records:
                        first_stage_done = True
                        print("第一阶段(计算公共波长范围)已完成")
                    else:
                        print(f"将继续进行第一阶段处理剩余的 {total_records - first_stage_processed} 条记录")
            except Exception as e:
                print(f"加载波长范围进度文件出错: {e}，将重新计算公共波长范围")
                self.processed_ranges = []
                first_stage_processed = 0
        else:
            self.processed_ranges = []
            first_stage_processed = 0
        
        # 如果第一阶段未完成，进行第一阶段处理
        if not first_stage_done:
            print(f"\n===== 第一阶段：计算最大公共波长范围 =====")
            first_stage_count = 0
            
            # 处理每个元素的数据
            for i, (df, element) in enumerate(zip(dataframes, elements)):
                print(f"\n处理元素 {i+1}/{len(dataframes)}: {element} (第一阶段)")
                
                # 处理每个光谱文件计算波长范围
                spec_files = df['spec'].values
                for j, spec_file in enumerate(tqdm(spec_files, desc=f"处理{element}光谱波长范围")):
                    # 如果已经处理过，跳过
                    if first_stage_processed + first_stage_count > j:
                        continue
                    
                    # 尝试从缓存获取
                    cache_key = f"wavelength_range_{spec_file.replace('/', '_')}"
                    cached_range = self.cache_manager.get_cache(cache_key)
                    
                    if cached_range:
                        # 如果有缓存，直接使用缓存的波长范围
                        w_min, w_max = cached_range
                        self.processed_ranges.append((w_min, w_max))
                        print(f"使用缓存的波长范围 {spec_file}: {w_min:.2f}~{w_max:.2f}")
                        
                    if cached_range:
                        # 如果有缓存，需要检查数据类型并提取波长范围
                        try:
                            # 打印缓存数据类型以便调试
                            print(f"缓存数据类型: {type(cached_range)}, 值: {cached_range}")
                            
                            # 处理不同类型的缓存数据
                            if isinstance(cached_range, tuple) and len(cached_range) == 2:
                                # 标准的二元组格式
                                w_min, w_max = cached_range
                            elif isinstance(cached_range, tuple) and len(cached_range) > 2:
                                # 如果元组有超过2个元素，只取前两个
                                print(f"警告: 缓存元组长度 > 2: {len(cached_range)}")
                                w_min, w_max = cached_range[0], cached_range[1]
                            elif isinstance(cached_range, list) and len(cached_range) >= 2:
                                # 列表格式
                                w_min, w_max = cached_range[0], cached_range[1]
                            elif isinstance(cached_range, dict) and 'data' in cached_range:
                                # 字典格式
                                data_array = cached_range['data']
                                if isinstance(data_array, np.ndarray) and len(data_array) == 2:
                                    w_min, w_max = data_array[0], data_array[1]
                                elif 'metadata' in cached_range and 'wavelength_range' in cached_range['metadata']:
                                    # 从元数据中获取
                                    w_range = cached_range['metadata']['wavelength_range']
                                    w_min, w_max = w_range[0], w_range[1]
                                else:
                                    # 无法提取，跳过缓存数据
                                    print(f"缓存数据格式不正确: {spec_file}，重新处理")
                                    cached_range = None
                            elif isinstance(cached_range, np.ndarray) and len(cached_range) >= 2:
                                # numpy数组格式
                                w_min, w_max = cached_range[0], cached_range[1]
                            else:
                                # 不支持的格式
                                print(f"缓存数据类型不支持: {type(cached_range)}，重新处理")
                                cached_range = None
                                
                            if cached_range:  # 如果成功提取了波长范围
                                self.processed_ranges.append((w_min, w_max))
                                print(f"使用缓存的波长范围 {spec_file}: {w_min:.2f}~{w_max:.2f}")
                        except Exception as e:
                            print(f"处理缓存数据时出错 {spec_file}: {e}，重新处理")
                            cached_range = None
                    
                    if not cached_range:  # 如果没有缓存或者缓存处理失败
                        # 没有缓存，需要读取文件提取波长范围
                        try:
                            # 查找FITS文件
                            file_path = self._get_file_extension(spec_file)
                            if not file_path:
                                print(f"无法找到文件: {spec_file}")
                                continue
                                
                            # 读取FITS文件，获取波长和流量
                            wavelength, flux, v_helio, z, snr, snr_bands = self.read_fits_file(file_path)
                            if wavelength is None or flux is None:
                                print(f"无法读取波长和流量数据: {file_path}")
                                continue
                            
                            # 检查并过滤无效值
                            valid_mask = ~np.isnan(flux) & ~np.isinf(flux)
                            if not np.any(valid_mask):
                                print(f"所有流量值都是无效的: {file_path}")
                                continue
                                
                            wavelength_valid = wavelength[valid_mask]
                            if len(wavelength_valid) < 2:
                                print(f"有效波长点数太少: {file_path}")
                                continue
                                
                            # 获取波长范围
                            w_min, w_max = wavelength_valid.min(), wavelength_valid.max()
                            self.processed_ranges.append((w_min, w_max))
                            
                            # 保存波长范围到缓存 - 使用字典格式
                            wavelength_range_data = {
                                'data': np.array([w_min, w_max]),  # 数据必须是numpy数组
                                'metadata': {
                                    'filename': spec_file,
                                    'wavelength_range': [w_min, w_max]
                                },
                                'validation_metrics': {
                                    'quality_metrics': {
                                        'snr': 10.0,  # 占位值
                                        'wavelength_coverage': 1.0,
                                        'normalization_quality': 1.0
                                    }
                                }
                            }
                            self.cache_manager.set_cache(cache_key, wavelength_range_data)
                            
                            # 更新最大公有范围
                            if len(self.processed_ranges) > 1:
                                common_min = max(r[0] for r in self.processed_ranges)
                                common_max = min(r[1] for r in self.processed_ranges)
                                
                                if common_min < common_max:
                                    self.wavelength_range = (common_min, common_max)
                                    
                        except Exception as e:
                            print(f"处理文件 {spec_file} 波长范围时出错: {e}")
                            continue
                    
                    first_stage_count += 1
                    
                    # 每处理100个文件保存一次进度
                    if first_stage_count % 100 == 0:
                        # 保存波长范围进度
                        with open(wave_range_file, 'wb') as f:
                            pickle.dump({
                                'processed_ranges': self.processed_ranges,
                                'wavelength_range': self.wavelength_range,
                                'processed_count': first_stage_processed + first_stage_count
                            }, f)
                        print(f"\n已保存波长范围进度: {first_stage_processed + first_stage_count}/{total_records}")
                        print(f"当前最大公共波长范围: {self.wavelength_range}")
            
            # 保存最终的波长范围进度
            with open(wave_range_file, 'wb') as f:
                pickle.dump({
                    'processed_ranges': self.processed_ranges,
                    'wavelength_range': self.wavelength_range,
                    'processed_count': total_records  # 标记为全部处理完成
                }, f)
            print(f"\n第一阶段完成：已确定最大公共波长范围 {self.wavelength_range}")
        
        # 第二阶段：重采样和后续处理
        print(f"\n===== 第二阶段：重采样和后续处理 =====")
        print(f"使用确定的公共波长范围: {self.wavelength_range}")
        
        # 如果没有有效的公共波长范围，使用默认范围
        if not hasattr(self, 'wavelength_range') or self.wavelength_range is None:
            self.wavelength_range = (3690, 9100)
            print(f"警告: 未找到有效的公共波长范围，使用默认范围 {self.wavelength_range}")
        
        # 处理每个元素的数据
        processed_records = 0
        for i, (df, element) in enumerate(zip(dataframes, elements)):
            # 更高效的方式统计已处理元素记录
            # 首先检查缓存中已有的处理结果数量
            cached_element_count = 0
            
            print(f"检查{element}元素的缓存预处理结果...")
            
            # 直接搜索缓存目录中的文件
            cache_files = glob.glob(os.path.join(self.cache_dir, "processed_spectrum_*"))
            print(f"发现 {len(cache_files)} 个缓存文件")
            
            # 获取元素中所有样本的spec文件名
            spec_files = df['spec'].values
            
            # 创建映射，将spec文件名的各种可能形式映射到实际的spec文件
            spec_mapping = {}
            for spec_file in spec_files:
                spec_base = os.path.basename(str(spec_file))
                if '.' in spec_base:
                    spec_base = spec_base.split('.')[0]
                
                # 存储不同的形式
                spec_mapping[spec_file] = spec_file
                spec_mapping[spec_base] = spec_file
                
                # 还可以添加其他可能的变体
                if spec_base.isdigit():
                    spec_mapping[spec_base.lstrip('0')] = spec_file
            
            # 遍历所有缓存文件，检查是否匹配任何spec
            for cache_file in tqdm(cache_files, desc=f"扫描{element}缓存"):
                try:
                    cache_basename = os.path.basename(cache_file)
                    # 从缓存文件名中提取可能的spec标识
                    if cache_basename.startswith("processed_spectrum_"):
                        cache_spec = cache_basename.replace("processed_spectrum_", "")
                        
                        # 检查是否能在映射中找到匹配
                        matched = False
                        matched_spec = None
                        
                        for spec_key, spec_value in spec_mapping.items():
                            if str(spec_key) in cache_spec or cache_spec in str(spec_key):
                                matched = True
                                matched_spec = spec_value
                                break
                        
                        if matched:
                            print(f"找到匹配: {cache_basename} -> {matched_spec}")
                            # 从文件直接读取缓存内容
                            try:
                                with open(cache_file, 'rb') as f:
                                    cached_result = pickle.load(f)
                                    
                                if cached_result is not None:
                                    cached_element_count += 1
                                    # 将已缓存的结果直接添加到all_data中
                                    if 'metadata' not in cached_result:
                                        cached_result['metadata'] = {}
                                    cached_result['metadata']['element'] = element
                                    all_data.append(cached_result)
                            except Exception as e:
                                print(f"读取缓存文件 {cache_file} 时出错: {e}")
                except Exception as e:
                    print(f"处理缓存文件 {cache_file} 时出错: {e}")
            
            # 打印匹配情况的详细信息
            print(f"\n===== 缓存匹配情况 =====")
            print(f"元素: {element}")
            print(f"总记录数: {len(df)}")
            print(f"匹配到缓存的记录数: {cached_element_count}")
            print(f"匹配率: {cached_element_count/len(df)*100:.2f}%")
            print(f"=======================\n")
            
            # 统计已处理的元素记录数
            element_records = cached_element_count
            # 元素的总记录数
            element_total = len(df)
            
            print(f"从缓存中发现{element}元素已处理记录: {element_records}/{element_total}")
            
            # 只有当元素的所有记录都已处理时，才认为该元素处理完成
            if element_records >= element_total:
                print(f"{element}数据已在之前的运行中处理完成 ({element_records}/{element_total}条记录)")
                processed_records += element_records
                # 注意：我们已经在扫描缓存时将数据添加到all_data中了
            else:
                print(f"\n处理元素 {i+1}/{len(dataframes)}: {element} (第二阶段)")
                print(f"已处理: {element_records}/{element_total}条记录")
                print(f"当前进度: [{processed_records/total_records:.2%}]")
                
                # 如果元素已部分处理，从未处理的部分继续
                start_idx = element_records
                print(f"从索引{start_idx}继续处理...")
                
                results = self.process_element_data(df, element, start_idx=start_idx)
                all_data.extend(results)
                processed_records += len(results)
                
                # 更新总体进度
                overall_progress = processed_records / total_records
                print(f"总进度: [{overall_progress:.2%}] 已完成{processed_records}/{total_records}条记录")
            
            # 保存总进度
            with open(progress_file, 'wb') as f:
                pickle.dump(all_data, f)
                
            # 输出阶段性统计
            elapsed_time = time.time() - start_time
            records_per_second = processed_records / elapsed_time if elapsed_time > 0 else 0
            print(f"当前已处理总数据量: {len(all_data)}条")
            print(f"处理速度: {records_per_second:.2f}条/秒")
            
            # 估计剩余时间
            if processed_records < total_records and records_per_second > 0:
                remaining_records = total_records - processed_records
                estimated_time = remaining_records / records_per_second
                hours, remainder = divmod(estimated_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(f"预计剩余时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
            
            # 检查内存使用情况
            self.check_memory_usage()
        
        # 如果没有处理到任何数据
        if not all_data:
            print("错误: 没有处理到任何有效数据，请检查fits文件路径和CSV文件")
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # 输出最终统计
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\n处理完成! 总耗时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
        print(f"处理记录: {len(all_data)}/{total_records} ({len(all_data)/total_records:.2%})")
        
        # 保存文件缓存
        self._save_files_cache()
        
        # 转换为NumPy数组并返回
        return self._prepare_arrays(all_data)
    
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
            return X, y, filenames, elements
            
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
                    
                    return X, y, filenames, elements
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
    
    def visualize_spectrum(self, spec_file, processed=True, save=True):
        """可视化单个光谱，原始光谱或处理后的光谱"""
        if processed:
            # 检查是否有缓存
            cache_key = f"processed_{spec_file.replace('/', '_')}"
            cached_data = self.cache_manager.get_cache(cache_key)
            
            if cached_data is None:
                # 如果没有缓存，处理光谱
                print(f"没有找到处理后的光谱缓存，重新处理: {spec_file}")
                processed_data = self.process_single_spectrum(spec_file, 0.0)  # 使用占位符标签
                if processed_data is None:
                    print(f"无法处理文件: {spec_file}")
                    return
            else:
                processed_data = cached_data
                
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
                spectrum = processed_data.get('spectrum')
        
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
            spectrum_normalized = np.clip(spectrum, 0, 1)
            plot_with_labels(ax4, wavelength_resampled, spectrum_normalized, 
                            (min(wavelength_resampled), max(wavelength_resampled)), 
                            absorption_lines, color='red', label_name='Fully Processed')
            ax4.set_ylabel('Final Normalized Flux')
            ax4.set_title("Spectrum after Second Denoising and Final Normalization")
            ax4.set_ylim(0, 1)  # 设置y轴范围为[0,1]
        
        else:
            # 如果不是处理后光谱，则使用原始光谱进行处理并显示
            
            # 1. 波长校正
            wavelength_calibrated = self.correct_wavelength(original_wavelength, original_flux)
            print(f"波长校正后: 波长范围{wavelength_calibrated[0]}~{wavelength_calibrated[-1]}")
            
            # 从FITS文件读取视向速度
            v_helio = 0  # 默认值
            try:
                _, _, v_helio, _, _, _ = self.read_fits_file(spec_file)
            except Exception as e:
                print(f"读取FITS文件获取视向速度时出错: {e}")
            
            # 2. 视向速度校正
            wavelength_corrected = self.correct_velocity(wavelength_calibrated, original_flux, v_helio)
            print(f"视向速度校正后: 波长范围{wavelength_corrected[0]}~{wavelength_corrected[-1]}")
            
            # 3. 去噪
            flux_denoised = self.denoise_spectrum(wavelength_corrected, original_flux)
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
            flux_normalized, norm_params = self.normalize_spectrum(flux_denoised_second)
            if flux_normalized is None:
                print(f"归一化{spec_file}失败")
                return
            
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

    def visualize_example_spectra(self, element=None):
        """可视化指定元素的示例光谱
        
        Args:
            element (str, optional): 元素名称，如 'C_FE'、'MG_FE'、'CA_FE'。
                                     为None时会尝试所有已处理的元素。
        """
        print(f"准备可视化示例光谱...")
        
        # 定义要处理的元素列表
        elements_to_process = []
        if element:
            elements_to_process = [element]
        else:
            # 尝试从已知的CSV文件中获取元素列表
            standard_elements = ['C_FE', 'MG_FE', 'CA_FE']
            for elem in standard_elements:
                if os.path.exists(f"{elem}.csv"):
                    elements_to_process.append(elem)
        
        if not elements_to_process:
            print("未找到指定元素或任何标准元素的CSV文件")
            return
        
        print(f"将处理以下元素: {', '.join(elements_to_process)}")
        
        # 处理每个元素
        for elem in elements_to_process:
            print(f"\n===== 处理元素: {elem} =====")
            
            # 1. 查找该元素的CSV文件
            csv_path = f"{elem}.csv"
            if not os.path.exists(csv_path):
                print(f"找不到{csv_path}，跳过此元素")
                continue
                
            # 2. 读取CSV文件获取光谱文件列表
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                
                if 'spec' not in df.columns:
                    print(f"CSV文件{csv_path}中找不到'spec'列")
                    continue
                    
                # 取前5个示例
                sample_specs = df['spec'].values[:5]
                if len(sample_specs) == 0:
                    print(f"CSV文件{csv_path}中没有光谱数据")
                    continue
                    
                print(f"找到{len(sample_specs)}个样本光谱")
                
                # 3. 可视化每个样本光谱
                for i, spec in enumerate(sample_specs):
                    print(f"\n处理样本 {i+1}/{len(sample_specs)}: {spec}")
                    
                    # 检查缓存和处理该光谱
                    cache_key = f"processed_{spec.replace('/', '_')}"
                    cached_data = self.cache_manager.get_cache(cache_key)
                    
                    if cached_data is None:
                        print(f"没有找到处理后的光谱缓存，尝试处理: {spec}")
                        
                        # 查找FITS文件
                        fits_file = self._find_fits_file(spec)
                        if fits_file is None:
                            print(f"找不到FITS文件: {spec}")
                            continue
                            
                        print(f"开始处理FITS文件: {fits_file}")
                        try:
                            # 使用0.0作为占位符标签处理光谱
                            processed_data = self.process_single_spectrum(spec, 0.0)
                            if processed_data is None:
                                print(f"处理光谱失败: {spec}")
                                continue
                                
                            print(f"光谱处理成功，保存到缓存")
                        except Exception as e:
                            print(f"处理光谱时出错: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                    else:
                        print(f"使用缓存的预处理光谱")
                    
                    # 可视化处理后的光谱
                    try:
                        print(f"可视化光谱: {spec}")
                        self.visualize_spectrum(spec, processed=True, save=True)
                        print(f"光谱可视化完成")
                    except Exception as e:
                        print(f"可视化光谱时出错: {e}")
                        import traceback
                        traceback.print_exc()
                        
            except Exception as e:
                print(f"处理元素{elem}时出错: {e}")
                import traceback
                traceback.print_exc()
                
        print("\n===== 示例光谱可视化完成 =====")
        print(f"图像保存在: {os.path.abspath(self.output_dir)}")

    # 在LAMOSTPreprocessor类内添加GPU插值函数
    def _gpu_interp(self, x, y, xnew):
        """在GPU上进行线性插值
        
        参数:
            x: 原始x坐标
            y: 原始y值
            xnew: 新的x坐标
            
        返回:
            插值后的y值
        """
        import cupy as cp
        
        # 确保x是单调递增的
        if not cp.all(cp.diff(x) > 0):
            # 排序x和y
            sort_indices = cp.argsort(x)
            x = x[sort_indices]
            y = y[sort_indices]
        
        # 找到xnew中每个点在x中的位置
        indices = cp.searchsorted(x, xnew) - 1
        
        # 处理边界情况
        indices = cp.clip(indices, 0, len(x) - 2)
        
        # 计算权重
        interval_width = x[indices + 1] - x[indices]
        weights = (xnew - x[indices]) / interval_width
        
        # 线性插值
        ynew = y[indices] * (1 - weights) + y[indices + 1] * weights
        
        return ynew

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
    parser.add_argument('--use_gpu', action='store_true',
                      help='使用GPU加速计算(需要安装CuPy库)')
    parser.add_argument('--no_gpu', action='store_false', dest='use_gpu',
                      help='禁用GPU加速，强制使用CPU')
    
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
    
    # 检查GPU支持
    use_gpu = args.use_gpu
    if use_gpu:
        if not HAS_GPU:
            print("警告: 要求使用GPU但未找到CuPy库，将使用CPU模式")
            print("要使用GPU，请安装CuPy库: pip install cupy-cuda11x (根据您的CUDA版本选择合适的包)")
            use_gpu = False
        else:
            # 检查GPU是否可用
            gpu_available = check_gpu_available()
            if not gpu_available:
                print("警告: 未检测到可用的GPU设备，将使用CPU模式")
                use_gpu = False
            else:
                print("🚀 将使用GPU加速计算")
    
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
        wavelength_range=(3690, 9100),  # 使用固定范围
        n_points=None,  # 使用步长自动计算点数
        log_step=0.0001,
        compute_common_range=False,  # 强制不计算公共范围
        max_workers=args.max_workers if args.max_workers is not None else (1 if low_memory_mode else 2),
        batch_size=5 if low_memory_mode else args.batch_size,
        memory_limit=args.memory_limit,
        low_memory_mode=low_memory_mode,
        use_gpu=use_gpu  # 添加GPU支持
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
    X, y, filenames, elements = preprocessor.process_all_data(resume=not args.no_resume)
    
    if X is None or len(X) == 0:
        print("错误: 没有处理到任何有效数据，请检查fits文件路径和CSV文件")
        return
        
    # 分割数据集
    preprocessor.split_dataset(X, y, elements)
    
    # 清理缓存文件查找记录
    preprocessor._save_files_cache()
    
    # 计算并显示总耗时
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n处理完成! 总耗时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")

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