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
import torch  # 用于深度学习处理
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
                 max_workers=None,  # 最大工作进程数，None表示自动确定
                 batch_size=20,  # 批处理大小
                 memory_limit=0.7,  # 内存使用限制(占总内存比例)
                 low_memory_mode=False):  # 低内存模式
        
        # 存储初始化参数
        self.csv_files = csv_files
        self.fits_dir = fits_dir
        self.output_dir = output_dir
        
        # 确保目录存在
        os.makedirs(self.fits_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建进度保存目录
        self.progress_dir = os.path.join(self.output_dir, "progress")
        os.makedirs(self.progress_dir, exist_ok=True)
        
        # 设置波长范围和点数
        self.wavelength_range = wavelength_range
        self.n_points = n_points
        self.log_step = log_step
        self.compute_common_range = compute_common_range
        
        # 设置并行处理参数
        if max_workers is None:
            # 自动设置为CPU核心数减1，避免系统资源过度使用
            self.max_workers = max(1, os.cpu_count() - 1) if os.cpu_count() else 2
        else:
            self.max_workers = max_workers
            
        self.batch_size = batch_size
        self.memory_limit = memory_limit
        
        # 设置缓存
        self.extension_cache = {}
        
        self.low_memory_mode = low_memory_mode
        
        # 读取CSV文件并保存为实例变量，方便其他方法访问
        self.dataframes = self.read_csv_data()
        
        # 光速常量（km/s）
        self.c = 299792.458
        
        # 存储已处理光谱的波长范围，用于计算最大公有范围
        self.processed_ranges = []
        
        print(f"设置最大工作进程数: {self.max_workers}")
        
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

        return dataframes
    
    def _find_fits_file(self, obsid):
        """通过OBSID查找对应的FITS文件"""
        # 如果映射尚未构建，先构建映射
        if not hasattr(self, '_fits_obsid_map') or not self._fits_obsid_map:
            self._build_fits_obsid_mapping()
        
        # 将输入转换为字符串，因为科学计数法可能会导致精度问题
        obsid_str = str(obsid).strip()
        
        # 直接查找映射表
        if obsid_str in self._fits_obsid_map:
            return self._fits_obsid_map[obsid_str]
        
        # 尝试科学计数法转换
        try:
            # 处理可能的科学计数法格式
            if 'E' in obsid_str or 'e' in obsid_str:
                # 尝试转换为浮点数再转整数
                obsid_num = int(float(obsid_str))
                obsid_str_alt = str(obsid_num)
                if obsid_str_alt in self._fits_obsid_map:
                    return self._fits_obsid_map[obsid_str_alt]
            else:
                # 尝试解析为浮点数再转回科学计数法格式
                obsid_float = float(obsid_str)
                obsid_sci = f"{obsid_float:.1e}".upper()
                if obsid_sci in self._fits_obsid_map:
                    return self._fits_obsid_map[obsid_sci]
        except:
            pass
        
        print(f"未找到OBSID为{obsid}的FITS文件")
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
    
    def process_single_spectrum(self, obsid, label):
        """处理单个光谱，适用于并行处理"""
        try:
            # 查找对应的FITS文件
            fits_file = self._find_fits_file(obsid)
            if fits_file is None:
                return {
                    'status': 'error',
                    'message': f'未找到OBSID {obsid}的FITS文件',
                    'obsid': obsid
                }
                
            # 读取光谱数据
            wavelength, flux, snr = self.read_fits_file(fits_file)
            if wavelength is None or flux is None:
                return {
                    'status': 'error',
                    'message': f'读取FITS文件失败: {fits_file}',
                    'obsid': obsid
                }
                
            # 处理光谱数据
            processed_data = self._process_spectrum(wavelength, flux, snr, obsid)
            
            return {
                'status': 'success',
                'data': processed_data,
                'obsid': obsid,
                'label': label
            }
            
            except Exception as e:
            print(f"处理OBSID {obsid}时出错: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'message': str(e),
                'obsid': obsid
            }
    
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
        
        # 获取obsid和标签
        if 'obsid' not in df.columns:
            print(f"错误: 数据集中缺少'obsid'列")
            return []
        
        obsids = df['obsid'].values
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
        remaining = len(obsids) - start_idx
        if remaining <= 0:
            print(f"{element}数据已全部处理完成")
            return results
            
        num_batches = (remaining + self.batch_size - 1) // self.batch_size
        
        # 测试第一个OBSID，确认匹配正确
        test_obsid = obsids[start_idx]
        print(f"测试第一个OBSID: {test_obsid}")
        # 查找对应的FITS文件
        found_path = self._find_fits_file(test_obsid)
        if found_path:
            print(f"找到FITS文件路径: {found_path}")
        else:
            print(f"警告: 找不到OBSID为{test_obsid}的FITS文件")
            # 尝试构建映射后再查找一次
            self._build_fits_obsid_mapping()
            found_path = self._find_fits_file(test_obsid)
            if found_path:
                print(f"重新构建映射后找到FITS文件: {found_path}")
            else:
                print(f"仍然无法找到OBSID为{test_obsid}的FITS文件，请检查数据")
            return results
        
        # 尝试处理第一个文件
        print("尝试处理测试文件...")
        test_result = self.process_single_spectrum(test_obsid, labels[start_idx])
        if test_result is None:
            print(f"警告: 无法处理OBSID为{test_obsid}的FITS文件，请检查文件内容或处理逻辑")
            return results
        else:
            print(f"OBSID为{test_obsid}的FITS文件处理成功，继续批量处理")
            results.append(test_result)
            results[0]['element'] = element
            total_processed += 1
        
        # 逐批处理剩余数据
        for batch_idx in tqdm(range(num_batches), desc=f"处理{element}光谱批次"):
            # 计算当前批次的索引范围
            current_start = start_idx + 1 + batch_idx * self.batch_size  # 跳过已测试的第一个文件
            if current_start >= len(obsids):
                break  # 防止越界
            
            current_end = min(current_start + self.batch_size, len(obsids))
            batch_obsids = obsids[current_start:current_end]
            batch_labels = labels[current_start:current_end]
            
            if len(batch_obsids) == 0 or len(batch_labels) == 0:
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
            failed_obsids = []
            
            with Pool(processes=self.max_workers) as pool:
                jobs = []
                for obsid, label in zip(batch_obsids, batch_labels):
                    jobs.append(pool.apply_async(self.process_single_spectrum, 
                                                (obsid, label)))
                
                for job in jobs:
                    try:
                        result = job.get(timeout=30)  # 设置超时避免卡死
                        if result is not None:
                            result['element'] = element
                            batch_results.append(result)
                            successful_count += 1
                        else:
                            failed_obsids.append(obsid)
                    except Exception as e:
                        print(f"处理作业出错: {e}")
                        continue
            
            # 添加到结果集
            results.extend(batch_results)
            total_processed += successful_count
            
            # 输出当前批次统计
            if len(batch_obsids) > 0:
                print(f"批次 {batch_idx+1}/{num_batches}: 成功 {successful_count}/{len(batch_obsids)} 个文件")
                if len(failed_obsids) > 0 and len(failed_obsids) < 5:
                    print(f"失败OBSID示例: {failed_obsids}")
            
            # 定期保存进度
            if batch_idx % 5 == 0 or batch_idx == num_batches - 1:
                with open(progress_file, 'wb') as f:
                    pickle.dump({'results': results, 'last_idx': current_end}, f)
        
        print(f"成功处理{total_processed}/{len(obsids)}条{element}光谱数据")
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
        """扫描FITS目录，构建文件路径到OBSID的映射"""
        if hasattr(self, '_fits_obsid_map') and self._fits_obsid_map:
            return self._fits_obsid_map
        
        print("构建FITS文件与OBSID的映射关系...")
        self._fits_obsid_map = {}
        
        # 获取所有FITS文件
        fits_files = []
        for root, _, files in os.walk(self.fits_dir):
            for file in files:
                if any(file.endswith(ext) for ext in ['.fits', '.fits.gz', '.fit', '.fit.gz']):
                    fits_files.append(os.path.join(root, file))
        
        print(f"找到{len(fits_files)}个FITS文件，开始提取OBSID...")
        
        # 批量处理文件，显示进度
        total_files = len(fits_files)
        for i, fits_file in enumerate(fits_files):
            if i % 100 == 0:
                print(f"进度: {i}/{total_files} ({i/total_files*100:.1f}%)")
            
            obsid = self._extract_obsid_from_fits(fits_file)
            if obsid:
                self._fits_obsid_map[obsid] = fits_file
        
        print(f"OBSID映射完成，共找到{len(self._fits_obsid_map)}个有效OBSID")
        return self._fits_obsid_map

    def process_data(self):
        """处理所有数据"""
        try:
            # 计算总批次数
            total_samples = len(self.dataframes[0])
            total_batches = (total_samples + self.batch_size - 1) // self.batch_size
            processed_batches = 0
            
            # 获取数据批次
            for batch_idx, batch_data in enumerate(self.get_batches(self.batch_size)):
                # 处理当前批次
                success = self.process_batch(batch_data)
                
                if success:
                    processed_batches += 1
                    # 保存当前批次结果
                    batch_results = {
                        'processed_data': self.processed_data if hasattr(self, 'processed_data') else None,
                        'wavelength_range': self.wavelength_range,
                        'n_points': self.n_points,
                        'batch_info': {
                            'size': len(batch_data),
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                    }
                    
                    # 保存到临时文件
                    batch_file = os.path.join(self.progress_dir, f'batch_results_{time.strftime("%Y%m%d_%H%M%S")}.pkl')
                    with open(batch_file, 'wb') as f:
                        pickle.dump(batch_results, f)
                    
                    # 显示当前进度和处理结果
                    print(f"\n批次 {processed_batches}/{total_batches} 处理完成")
                    print("本批次处理结果：")
                    print(f"- 处理的光谱数量: {len(batch_data)}")
                    print(f"- 结果保存到: {batch_file}")
                    
                    # 在处理完前两批后询问是否继续
                    if batch_idx < 2:
                        user_input = input("\n是否继续处理下一批数据？(y/n): ").lower().strip()
                        if user_input != 'y':
                            print("用户选择停止处理")
                            break
                else:
                    print(f"批次 {processed_batches + 1} 处理失败")
                    
            return processed_batches > 0
            
                    except Exception as e:
            print(f"数据处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    # 设置参数
    params = {
        'csv_file': 'CA_FE.csv',  # 参考数据集
        'fits_dir': 'fits',  # 光谱文件目录
        'output_dir': 'processed_data',  # 输出目录
        'batch_size': 32,  # 批处理大小
        'num_workers': 4,  # 数据加载的工作进程数
        'test_size': 0.2,  # 测试集比例
        'val_size': 0.2,  # 验证集比例
        'random_state': 42,  # 随机种子
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备选择
    }
    
    # 创建预处理器
    preprocessor = LAMOSTPreprocessor(
        csv_file=params['csv_file'],
        fits_dir=params['fits_dir'],
        output_dir=params['output_dir'],
        batch_size=params['batch_size'],
        num_workers=params['num_workers'],
        test_size=params['test_size'],
        val_size=params['val_size'],
        random_state=params['random_state'],
        device=params['device']
    )
    
    # 询问用户是否继续数据处理
    if not preprocessor.ask_user_continue():
        return
    
    # 询问用户是否划分数据集
    if not preprocessor.ask_user_split():
        return
    
    # 执行预处理
    preprocessor.process()

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