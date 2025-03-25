import os
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

# 导入预处理模块
from preprocessdata import LAMOSTPreprocessor

# 设置日志
logger = logging.getLogger('multi_element_processor')

class MultiElementProcessor:
    def __init__(self, fits_dir=None, cache_dir=None):
        self.fits_dir = fits_dir
        self.preprocessor = LAMOSTPreprocessor()
        self.obsid_fits_mapping = {}
        self.cached_spectra = {}
        
    def read_csv_data(self, csv_file, element_columns=None):
        """读取包含多元素丰度的CSV文件"""
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 验证obsid列是否存在
        if 'obsid' not in df.columns:
            raise ValueError(f"CSV文件必须包含obsid列，现有列: {df.columns}")
        
        # 将obsid转为字符串类型
        df['obsid'] = df['obsid'].astype(str)
        
        # 如果未指定元素列，尝试自动检测
        if element_columns is None:
            # 尝试查找形如X_FE的列名（X/Fe相对丰度）
            element_columns = [col for col in df.columns if col.endswith('_FE')]
            if not element_columns:
                raise ValueError("未找到元素丰度列，请明确指定element_columns参数")
        
        # 验证元素列是否存在
        for col in element_columns:
            if col not in df.columns:
                raise ValueError(f"找不到元素丰度列: {col}")
        
        # 保留需要的列：obsid和元素丰度列
        result_df = df[['obsid'] + element_columns].copy()
        
        # 记录每个元素的有效数据数量
        valid_counts = {}
        for col in element_columns:
            valid_mask = ~result_df[col].isna() & (result_df[col] != -9999)
            valid_counts[col] = valid_mask.sum()
            logger.info(f"元素{col}有效数据数量: {valid_counts[col]}/{len(result_df)}")
        
        return result_df
    
    def build_element_dataset(self, csv_data, element_column):
        """为单个元素构建数据集"""
        # 获取有效数据掩码
        valid_mask = ~csv_data[element_column].isna() & (csv_data[element_column] != -9999)
        valid_indices = np.where(valid_mask)[0]
        
        # 初始化结果数组
        X = []
        y = []
        processed_indices = []
        
        # 处理每个有效样本
        for idx in valid_indices:
            obsid = csv_data.iloc[idx]['obsid']
            abundance = csv_data.iloc[idx][element_column]
            
            # 处理光谱数据
            spectrum_data = self.get_spectrum(obsid)
            if spectrum_data is not None:
                X.append(spectrum_data)
                y.append(abundance)
                processed_indices.append(idx)
        
        return np.array(X), np.array(y), processed_indices
    
    def get_spectrum(self, obsid):
        """获取处理后的光谱数据，支持缓存"""
        # 检查缓存
        if obsid in self.cached_spectra:
            return self.cached_spectra[obsid]
        
        try:
            # 处理光谱
            processed_data = self.preprocessor.process_single_spectrum(obsid, None)
            if processed_data and 'spectrum' in processed_data:
                spectrum = processed_data['spectrum']
                # 缓存结果
                self.cached_spectra[obsid] = spectrum
                return spectrum
            else:
                logger.warning(f"处理OBSID为{obsid}的光谱失败")
                return None
        except Exception as e:
            logger.warning(f"处理OBSID为{obsid}的光谱时出错: {str(e)}")
            return None
    
    def prepare_datasets(self, csv_file, element_columns=None, test_size=0.2, val_size=0.1, random_state=42):
        """准备多个元素的训练、验证和测试数据集"""
        # 读取CSV数据
        csv_data = self.read_csv_data(csv_file, element_columns)
        
        # 如果未指定元素列，使用所有检测到的元素列
        if element_columns is None:
            element_columns = [col for col in csv_data.columns if col.endswith('_FE')]
        
        # 准备每个元素的数据集
        datasets = {}
        for element in element_columns:
            logger.info(f"准备{element}元素数据集...")
            X, y, _ = self.build_element_dataset(csv_data, element)
            
            if len(X) == 0:
                logger.warning(f"元素{element}没有有效数据，跳过")
                continue
                
            # 划分数据集
            # 首先分离测试集
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)
            
            # 然后从剩余数据中分离验证集
            val_size_adjusted = val_size / (1 - test_size)  # 调整验证集比例
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state)
            
            datasets[element] = {
                'train': (X_train, y_train),
                'val': (X_val, y_val),
                'test': (X_test, y_test)
            }
            
            logger.info(f"元素{element}数据集准备完成: 训练集{len(X_train)}样本, "
                       f"验证集{len(X_val)}样本, 测试集{len(X_test)}样本")
        
        return datasets