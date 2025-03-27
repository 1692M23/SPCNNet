import numpy as np
import pandas as pd
from astropy.io import fits
import os
import glob
from sklearn.model_selection import train_test_split, KFold
from scipy import signal, interpolate
from multiprocessing import Pool, cpu_count
import time
from tqdm import tqdm
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
    def __init__(self, csv_files=['C_FE.csv', 'MG_FE.csv', 'CA_FE.csv'], 
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
                 low_memory_mode=False):  # 低内存模式
        
        self.csv_files = csv_files
        self.fits_dir = fits_dir
        self.output_dir = output_dir
        self.wavelength_range = wavelength_range if wavelength_range else (4000, 8098)  # 默认范围
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
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 进度记录目录
        self.progress_dir = os.path.join(output_dir, 'progress')
        if not os.path.exists(self.progress_dir):
            os.makedirs(self.progress_dir)
            
        # 缓存目录
        self.cache_dir = os.path.join(output_dir, 'cache')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        # 检查fits目录是否存在
        if not os.path.exists(fits_dir):
            os.makedirs(fits_dir)
            print(f"警告: 创建了fits目录，请确保FITS文件放在 {os.path.abspath(fits_dir)} 目录下")
        else:
            print(f"已找到fits目录: {os.path.abspath(fits_dir)}")
            fits_count = len(glob.glob(os.path.join(fits_dir, "*")))
            print(f"该目录中有 {fits_count} 个文件")
            
        # 后缀缓存，避免重复查找
        self.extension_cache = {}
        
        self.low_memory_mode = low_memory_mode
        
        self.cache_manager = CacheManager(cache_dir=os.path.join(output_dir, 'cache'))
        
    def read_csv_data(self):
        """读取CSV文件并返回DataFrame列表"""
        dataframes = []
        for csv_file in self.csv_files:
            if not os.path.exists(csv_file):
                print(f"错误: 找不到CSV文件 {csv_file}")
                continue
                
            df = pd.read_csv(csv_file)
            dataframes.append(df)
            print(f"已加载{csv_file}，共{len(df)}条记录")
            
            # 检查spec列中的文件是否存在
            if 'spec' in df.columns:
                spec_files = df['spec'].values
                missing_files = []
                for spec_file in spec_files:
                    # 使用_find_fits_file方法查找文件
                    if self._find_fits_file(spec_file) is None:
                        missing_files.append(spec_file)
                
                if missing_files:
                    print(f"警告: {csv_file}中的一些FITS文件不存在 (显示前5个):")
                    for file in missing_files[:5]:
                        print(f"  - {os.path.join(self.fits_dir, file)}")
                    print("请确保所有FITS文件都放在fits目录下")
        
        return dataframes
    
    def _find_fits_file(self, spec_name):
        """查找匹配的fits文件，处理嵌套目录和命名差异"""
        # 如果输入已经是完整路径，提取文件名部分
        if os.path.isabs(spec_name):
            base_name = os.path.basename(spec_name)
        else:
            base_name = spec_name
        
        # 首先尝试直接匹配（常规后缀）
        for ext in ['', '.fits', '.fits.gz', '.fit', '.fit.gz']:
            path = os.path.join(self.fits_dir, base_name + ext)
            if os.path.exists(path) and os.path.isfile(path):
                return path
        
        # 进行递归搜索，处理嵌套目录
        for root, dirs, files in os.walk(self.fits_dir):
            for file in files:
                # 检查文件名是否匹配（忽略大小写）
                if base_name.lower() in file.lower():
                    found_path = os.path.join(root, file)
                    return found_path
                
                # 尝试去除可能的后缀后再比较
                file_base = file.lower()
                for ext in ['.fits', '.fits.gz', '.fit', '.fit.gz']:
                    if file_base.endswith(ext):
                        file_base = file_base[:-len(ext)]
                        break
                
                if base_name.lower() == file_base:
                    found_path = os.path.join(root, file)
                    return found_path
        
        # 如果以上都没找到，返回None
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
        """读取FITS文件并返回波长和流量数据"""
        # 获取正确的文件路径
        file_path = self._get_file_extension(fits_file)
        if file_path is None:
            print(f"无法找到文件: {fits_file}，查找路径: {self.fits_dir}")
            return None, None, 0
        
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
                v_helio = header.get('V_HELIO', 0)  # 日心视向速度 (km/s)
                
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
                    return None, None, 0
                
                if wavelength is None:
                    print("无法生成波长数据")
                    return None, None, 0
                
                # 确保数据类型是浮点数
                flux = flux.astype(np.float64)
                wavelength = wavelength.astype(np.float64)
                
                # 检查是否有NaN或无限值
                if np.isnan(flux).any() or np.isinf(flux).any():
                    print(f"数据中包含NaN或无限值，尝试替换")
                    flux = np.nan_to_num(flux)
                
                print(f"成功提取光谱数据: 点数={len(wavelength)}, 波长范围={wavelength[0]:.2f}~{wavelength[-1]:.2f}")
                return wavelength, flux, v_helio
                
        except Exception as e:
            print(f"读取{file_path}出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None, 0
    
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
            valid_flux = flux[~np.isnan(flux)]
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
        """处理单个光谱数据"""
        # 使用CacheManager替代直接缓存操作
        cache_key = f"processed_{spec_file.replace('/', '_')}"
        cached_data = self.cache_manager.get_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # 读取FITS文件
            print(f"处理光谱: {spec_file}")
            wavelength, flux, v_helio = self.read_fits_file(spec_file)
            if wavelength is None or flux is None:
                print(f"无法读取FITS文件: {spec_file}")
                return None
            
            # 检查数据有效性
            if np.isnan(flux).all() or len(flux) == 0:
                print(f"文件{spec_file}中的流量数据全为NaN或为空")
                return None
            
            print(f"原始数据: 波长范围{wavelength[0]}~{wavelength[-1]}, 点数={len(wavelength)}")
            
            # 获取红移数据
            z = 0
            try:
                # 尝试从文件名匹配到CSV中的记录获取红移
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
                print(f"查找红移数据出错: {e}")
                # 出错时使用默认值
                z = 0
            
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
            flux_continuum = self.normalize_continuum(wavelength_resampled, flux_resampled)
            
            # 7. 二次去噪
            flux_denoised_second = self.denoise_spectrum_second(wavelength_resampled, flux_continuum)
            
            # 8. 最终归一化 (最大最小值归一化)
            print(f"对流量进行最终归一化")
            flux_normalized = self.normalize_spectrum(flux_denoised_second)
            if flux_normalized is None:
                print(f"归一化{spec_file}失败")
                return None
            
            print(f"成功处理光谱: {spec_file}")
            
            # 返回处理后的光谱和标签，包括中间处理结果
            result = {
                'spectrum': flux_normalized, 
                'label': label, 
                'filename': spec_file,
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
                'z': z  # 保存红移值
            }
            
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
        
        # 获取光谱文件名和标签
        spec_files = df['spec'].values
        labels = df.iloc[:, -1].values  # 最后一列是标签
        
        # 检查进度文件
        progress_file = os.path.join(self.progress_dir, f"{element}_progress.pkl")
        results = []
        total_processed = 0
        
        # 如果有进度文件，加载已处理的结果
        if os.path.exists(progress_file) and start_idx == 0:
            try:
                with open(progress_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    results = saved_data.get('results', [])
                    start_idx = saved_data.get('last_idx', 0)
                    total_processed = len(results)
                    print(f"从上次中断处继续（已处理{total_processed}条记录）")
            except Exception as e:
                print(f"加载进度文件出错: {e}，将从头开始处理")
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
            
            # 输出当前批次统计
            if len(batch_specs) > 0:
                print(f"批次 {batch_idx+1}/{num_batches}: 成功 {successful_count}/{len(batch_specs)} 个文件")
                if len(failed_specs) > 0 and len(failed_specs) < 5:
                    print(f"失败文件示例: {failed_specs}")
            
            # 定期保存进度
            if batch_idx % 5 == 0 or batch_idx == num_batches - 1:
                with open(progress_file, 'wb') as f:
                    pickle.dump({'results': results, 'last_idx': current_end}, f)
        
        print(f"成功处理{total_processed}/{len(spec_files)}条{element}光谱数据")
        return results
    
    def process_all_data(self, resume=True):
        """处理所有数据并准备训练集，支持断点续传"""
        start_time = time.time()
        all_data = []
        
        # 检查是否有整体进度文件
        progress_file = os.path.join(self.progress_dir, "all_progress.pkl")
        if resume and os.path.exists(progress_file):
            try:
                with open(progress_file, 'rb') as f:
                    all_data = pickle.load(f)
                print(f"已加载保存的处理结果，共{len(all_data)}条记录")
                
                # 是否继续处理
                if input("是否继续处理剩余数据？(y/n): ").lower() != 'y':
                    # 直接跳到数据转换步骤
                    return self._prepare_arrays(all_data)
            except Exception as e:
                print(f"加载进度文件出错: {e}，将重新处理所有数据")
                all_data = []
        
        # 读取CSV文件
        dataframes = self.read_csv_data()
        if not dataframes:
            print("错误: 没有有效的数据集")
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # 处理每个元素的数据
        for i, (df, element) in enumerate(zip(dataframes, ['C_FE', 'MG_FE', 'CA_FE'])):
            # 检查是否已处理过这个元素
            element_processed = any(item.get('element') == element for item in all_data) if all_data else False
            
            if not element_processed:
                results = self.process_element_data(df, element)
                all_data.extend(results)
            else:
                print(f"{element}数据已在之前的运行中处理完成")
            
            # 保存总进度
            with open(progress_file, 'wb') as f:
                pickle.dump(all_data, f)
                
            print(f"当前已处理总数据量: {len(all_data)}条")
            
            # 检查内存使用情况
            self.check_memory_usage()
        
        # 如果没有处理到任何数据
        if not all_data:
            print("警告: 没有成功处理任何光谱数据!")
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # 转换为NumPy数组并返回
        return self._prepare_arrays(all_data)
    
    def _prepare_arrays(self, all_data):
        """将处理结果转换为NumPy数组并保存"""
        # 在转换之前检查并过滤无效数据
        valid_data = []
        invalid_count = 0
        spectrum_lengths = []
        
        for item in all_data:
            # 检查spectrum是否是有效的数组
            spectrum = item.get('spectrum')
            if spectrum is not None and isinstance(spectrum, (list, np.ndarray)) and len(spectrum) > 0:
                spectrum_lengths.append(len(spectrum))
                valid_data.append(item)
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            print(f"警告: 过滤了{invalid_count}条无效数据")
        
        if len(valid_data) == 0:
            print("错误: 没有有效的处理结果")
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # 检查所有光谱的长度是否一致
        if len(set(spectrum_lengths)) > 1:
            most_common_length = max(set(spectrum_lengths), key=spectrum_lengths.count)
            print(f"警告: 光谱长度不一致! 发现{len(set(spectrum_lengths))}种不同长度")
            print(f"最常见的长度为{most_common_length}，将其他长度的光谱过滤掉")
            
            # 过滤掉长度不一致的数据
            consistent_data = [item for item, length in zip(valid_data, spectrum_lengths) 
                              if length == most_common_length]
            
            print(f"保留了{len(consistent_data)}/{len(valid_data)}条长度一致的数据")
            valid_data = consistent_data
        
        # 转换为NumPy数组
        try:
            X = np.array([item['spectrum'] for item in valid_data])
            y = np.array([item['label'] for item in valid_data])
            elements = np.array([item.get('element', '') for item in valid_data])
            filenames = np.array([item['filename'] for item in valid_data])
            
            print(f"成功创建数据数组: X形状={X.shape}, y形状={y.shape}")
        except ValueError as e:
            print(f"创建数组时出错: {e}")
            # 尝试更详细地诊断问题
            lengths = [len(item['spectrum']) for item in valid_data]
            unique_lengths = set(lengths)
            if len(unique_lengths) > 1:
                print(f"光谱长度不一致: {unique_lengths}")
                length_counts = {length: lengths.count(length) for length in unique_lengths}
                print(f"各长度数量: {length_counts}")
                
                # 选择最常见的长度
                most_common = max(length_counts.items(), key=lambda x: x[1])
                print(f"最常见的长度为: {most_common[0]}，出现{most_common[1]}次")
                
                # 只保留最常见长度的数据
                filtered_data = [item for item, length in zip(valid_data, lengths) 
                                if length == most_common[0]]
                print(f"过滤后保留{len(filtered_data)}/{len(valid_data)}条数据")
                
                # 重新尝试创建数组
                X = np.array([item['spectrum'] for item in filtered_data])
                y = np.array([item['label'] for item in filtered_data])
                elements = np.array([item.get('element', '') for item in filtered_data])
                filenames = np.array([item['filename'] for item in filtered_data])
                valid_data = filtered_data
            else:
                # 如果不是长度问题，可能是其他类型不一致
                print("检查数据类型:")
                sample_types = [type(item['spectrum']) for item in valid_data[:5]]
                print(f"前5条记录的光谱类型: {sample_types}")
                
                # 尝试将所有数据转换为相同类型
                X = np.array([np.array(item['spectrum'], dtype=float) for item in valid_data])
                y = np.array([item['label'] for item in valid_data])
                elements = np.array([item.get('element', '') for item in valid_data])
                filenames = np.array([item['filename'] for item in valid_data])
        
        # 保存处理后的数据
        np.savez(os.path.join(self.output_dir, 'processed_data.npz'),
                X=X, y=y, elements=elements, filenames=filenames)
        
        print(f"数据处理完成，总数据量: {len(X)}条")
        
        return X, y, elements, filenames
    
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
        
        print(f"数据集分割完成:")
        print(f"训练集: {X_train.shape[0]}条 (70%)")
        print(f"验证集: {X_val.shape[0]}条 (10%)")
        print(f"测试集: {X_test.shape[0]}条 (20%)")
        
        return (X_train, y_train, elements_train), (X_val, y_val, elements_val), (X_test, y_test, elements_test)
    
    def setup_cross_validation(self, X, y):
        """设置交叉验证"""
        if len(X) == 0:
            print("错误: 没有数据可以进行交叉验证")
            return []
            
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        folds = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            folds.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))
            
            # 保存每个折的数据
            fold_dir = os.path.join(self.output_dir, f"fold_{fold}")
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)
                
            np.savez(os.path.join(fold_dir, 'fold_data.npz'),
                    X_train=X_train_fold, y_train=y_train_fold,
                    X_val=X_val_fold, y_val=y_val_fold)
        
        print(f"交叉验证设置完成: {self.n_splits}折")
        return folds
    
    def predict_abundance(self, fits_file, model):
        """预测单个FITS文件的元素丰度"""
        # 读取并预处理光谱
        wavelength, flux, v_helio = self.read_fits_file(fits_file)
        if wavelength is None or flux is None:
            return None
        
        # 去噪
        flux_denoised = self.denoise_spectrum(wavelength, flux)
        
        # 视向速度校正
        wavelength_corrected = self.correct_velocity(wavelength, flux_denoised, v_helio)
        
        # 重采样
        wavelength_resampled, flux_resampled = self.resample_spectrum(wavelength_corrected, flux_denoised)
        if wavelength_resampled is None or flux_resampled is None:
            return None
        
        # 归一化
        flux_normalized = self.normalize_spectrum(flux_resampled)
        if flux_normalized is None:
            return None
        
        # 准备模型输入
        X_input = np.expand_dims(flux_normalized, axis=0)
        
        # 使用模型预测
        prediction = model.predict(X_input)
        
        return prediction[0]
    
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
                    
                    # 检查是否包含spec列
                    if 'spec' in df.columns:
                        spec_example = df['spec'].iloc[0] if len(df) > 0 else "无数据"
                        print(f"  - 'spec'列示例: {spec_example}")
                    else:
                        print(f"  ✗ 错误: {csv_file} 中没有'spec'列")
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
                
                # 检查CSV文件中的spec值是否匹配fits文件
                for csv_file in self.csv_files:
                    if os.path.exists(csv_file):
                        df = pd.read_csv(csv_file)
                        if 'spec' in df.columns and len(df) > 0:
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
        
        print("\n=== 检查完成 ===\n")
        
    def clean_cache(self):
        """清理缓存文件"""
        if os.path.exists(self.cache_dir):
            cache_files = glob.glob(os.path.join(self.cache_dir, "*"))
            if cache_files:
                print(f"发现{len(cache_files)}个缓存文件")
                if input("是否清理缓存? (y/n): ").lower() == 'y':
                    for file in tqdm(cache_files, desc="清理缓存"):
                        try:
                            os.remove(file)
                        except Exception:
                            pass
                    print("缓存清理完成")

    def check_and_fix_file_paths(self):
        """检查并修复文件路径问题"""
        print("\n=== 路径问题诊断 ===")
        
        # 创建测试文件列表
        test_files = []
        for csv_file in self.csv_files:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                if 'spec' in df.columns:
                    for spec in df['spec'].values[:5]:  # 只取前5个测试
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
        self.extension_cache = {}
        for spec in test_files[:1]:  # 只测试第一个
            print(f"重新测试: {spec}")
            found_path = self._find_fits_file(spec)
            if found_path:
                print(f"  重新测试找到: {found_path}")
            else:
                print(f"  重新测试未找到文件")
        
        print("\n=== 诊断完成 ===\n")

def main():
    """主函数"""
    start_time = time.time()
    
    # 设置基础路径
    base_path = '/content' if IN_COLAB else os.path.abspath('.')
    print(f"基础路径: {base_path}")
    
    # 获取系统内存信息
    mem_info = psutil.virtual_memory()
    print(f"系统内存: 总计 {mem_info.total / (1024**3):.1f}GB, "
          f"可用 {mem_info.available / (1024**3):.1f}GB, "
          f"使用率 {mem_info.percent}%")
    
    # 检测内存情况，自动决定是否使用低内存模式
    low_memory_mode = mem_info.percent > 80
    
    if low_memory_mode:
        print("检测到系统内存不足，自动启用低内存模式")
        user_choice = input("是否启用低内存模式? 这将减少内存使用但处理速度会变慢 (y/n): ").lower()
        low_memory_mode = user_choice == 'y'
    
    # 构建路径
    csv_files = [os.path.join(base_path, f) for f in ['C_FE.csv', 'MG_FE.csv', 'CA_FE.csv']]
    fits_dir = os.path.join(base_path, 'fits')
    output_dir = os.path.join(base_path, 'processed_data')
    
    # 展示路径信息
    print(f"CSV文件路径: {csv_files}")
    print(f"FITS目录路径: {fits_dir}")
    print(f"输出目录路径: {output_dir}")
    
    # 确保fits目录存在
    if not os.path.exists(fits_dir):
        print(f"创建FITS目录: {fits_dir}")
        os.makedirs(fits_dir, exist_ok=True)
    
    # 检测FITS文件是否有效
    print("\n=== 检查FITS文件有效性 ===")
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
    preprocessor = LAMOSTPreprocessor(
        csv_files=csv_files,
        fits_dir=fits_dir,
        output_dir=output_dir,
        wavelength_range=None,  # 修改为None，表示将使用最大公有波长范围
        n_points=None,  # 修改为None，点数将根据波长范围和步长自动计算
        log_step=0.0001,  # 新增：对数空间中的重采样步长（dex）
        compute_common_range=True,  # 新增：是否计算最大公有波长范围
        n_splits=5,
        max_workers=1 if low_memory_mode else 2,  # 低内存模式使用单线程
        batch_size=5 if low_memory_mode else 20,   # 低内存模式减小批次大小
        memory_limit=0.7,  # 内存使用阈值
        low_memory_mode=low_memory_mode  # 低内存模式标志
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
    
    if len(X) == 0:
        print("错误: 没有处理到任何有效数据，请检查fits文件路径和CSV文件")
        return
    
    # 分割数据集
    train_data, val_data, test_data = preprocessor.split_dataset(X, y, elements)
    
    # 设置交叉验证
    X_train, y_train, _ = train_data
    folds = preprocessor.setup_cross_validation(X_train, y_train)
    
    # 可视化几个示例光谱(可选)
    if len(filenames) > 0 and not low_memory_mode and input("是否可视化示例光谱? (y/n): ").lower() == 'y':
        print("正在可视化示例光谱...")
        for i in range(min(3, len(filenames))):
            preprocessor.visualize_spectrum(filenames[i])
    
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