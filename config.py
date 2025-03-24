#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置文件：包含模型训练、数据处理和输出相关的所有配置参数
"""

import os
import torch
import numpy as np

# 数据路径配置
data_paths = {
    'train_data': 'data/processed/train_dataset.npz',
    'val_data': 'data/processed/val_dataset.npz',
    'test_data': 'data/processed/test_dataset.npz',
    'raw_data': 'data/raw/spectra.npz',
    'C_FE_csv': 'data/processed/C_FE_metadata.csv',
    'MG_FE_csv': 'data/processed/MG_FE_metadata.csv',
    'CA_FE_csv': 'data/processed/CA_FE_metadata.csv'
}

# 数据处理配置
data_config = {
    'random_seed': 42,
    'train_ratio': 0.7,
    'val_ratio': 0.1,
    'test_ratio': 0.2,
    'normalize': True,
    'standardize': True,
    'remove_outliers': True,
    'outlier_threshold': 3.0,  # 标准差的倍数
    'wavelength_range': [3800, 9000],  # 光谱波长范围
    'resample_dim': 3000,              # 光谱重采样维度
    'denoise': True,                   # 是否去噪
    'denoise_params': [
        {'window': 5, 'polyorder': 2},  # 第一次去噪参数
        {'window': 7, 'polyorder': 3}   # 第二次去噪参数
    ],
    'denoise_window': 11,              # 去噪窗口大小 (Savitzky-Golay) - 旧参数，保留兼容性
    'denoise_polyorder': 2,            # 去噪多项式阶数 - 旧参数，保留兼容性
    'normalization_method': 'median',  # 归一化方法 (median, minmax, standard)
        'n_splits': 5,                     # K折交叉验证的K值
    'shuffle': True,                   # 是否打乱数据
        'wavelength_grids': {
            'u': np.linspace(3000, 4000, 1000),  # u波段波长范围
        'g': np.linspace(4000, 5000, 1000),  # g波段波长范围
        'r': np.linspace(5000, 6000, 1000),  # r波段波长范围
        'i': np.linspace(6000, 7000, 1000),  # i波段波长范围
        'z': np.linspace(7000, 8000, 1000),  # z波段波长范围
    }
}

# 模型配置
model_config = {
    'input_dim': 7214,  # 光谱维度
    'num_filters': 32,  # 初始卷积层通道数
    'num_blocks': 2,    # 每个残差块组的块数
    'model_dir': 'models'  # 模型保存目录
}

# 训练配置
training_config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'num_epochs': 100,
    'early_stopping_patience': 15,
    'batch_size': 32,
    'shuffle': True,
    'elements': ['Fe', 'Mg', 'Si', 'Ca', 'Ti', 'Ni', 'Cr', 'Mn', 'V', 'Co']  # 要预测的元素列表
}

# 超参数调优配置
tuning_config = {
    'param_grid': {
        'learning_rate': [5e-5, 1e-4, 5e-4],
        'batch_size': [32, 64, 128],
        'weight_decay': [1e-6, 1e-5, 1e-4]
    },
    'cv_folds': 3,                     # 交叉验证折数
    'early_stopping_patience': 10      # 调优时的早停耐心值
}

# 输出配置
output_config = {
    'log_dir': 'logs',
    'results_dir': 'results',
    'plots_dir': 'plots',
    'predictions_dir': 'predictions'
}

# MIST等时线配置
mist_isochrones = {
    'enabled': True,
    'ages': [1, 3, 5, 10],  # Gyr
    'colors': ['red', 'green', 'blue', 'purple'],
    'data_dir': './mist_data/' if os.path.exists('./mist_data/') else None
}

# 元素丰度预测配置
element_config = {
    'C_FE': {
        'range': [-1.0, 1.0],  # 典型值范围
        'unit': 'dex',
        'display_name': '[C/Fe]',
        'description': '碳丰度比'
    },
    'MG_FE': {
        'range': [-0.5, 0.8],  # 典型值范围
        'unit': 'dex',
        'display_name': '[Mg/Fe]',
        'description': '镁丰度比'
    },
    'CA_FE': {
        'range': [-0.5, 0.8],  # 典型值范围
        'unit': 'dex',
        'display_name': '[Ca/Fe]',
        'description': '钙丰度比'
    }
}

# 创建所需目录
required_directories = [
    'data/processed',
    'data/raw',
    model_config['model_dir'],
    output_config['log_dir'],
    output_config['results_dir'],
    output_config['plots_dir'],
    output_config['predictions_dir']
]

for directory in required_directories:
    os.makedirs(directory, exist_ok=True)
    print(f"已创建目录: {directory}")

# 数据预处理配置
preprocessing_config = {
    'wavelength_range': (4000, 7000),  # 波长范围
    'normalize': True,                  # 是否进行标准化
    'denoise': True,                    # 是否进行去噪
    'denoise_params': {
        'window_size': 5,
        'sigma': 1.0
    }
}

# 评估配置
evaluation_config = {
    'metrics': ['mae', 'mse', 'rmse', 'r2', 'dex'],  # 评估指标
    'plot_types': ['predictions', 'residuals', 'error_distribution'],  # 可视化类型
    'save_predictions': True  # 是否保存预测结果
}

# 将所有配置合并到一个字典中
CONFIG = {
    'data_paths': data_paths,
    'data_config': data_config,
    'model_config': model_config,
    'training_config': training_config,
    'tuning_config': tuning_config,
    'output_config': output_config,
    'mist_isochrones': mist_isochrones,
    'element_config': element_config,
    'preprocessing_config': preprocessing_config,
    'evaluation_config': evaluation_config
} 