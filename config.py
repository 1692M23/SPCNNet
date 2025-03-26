#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置文件：包含模型训练、数据处理和输出相关的所有配置参数
"""

import os
import torch
import numpy as np
import logging
import json
import time
import pickle

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('config.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('config')

# 数据路径配置
DATA_CONFIG = {
    # 数据集路径
    'train_data': 'processed_data/train_dataset.npz',
    'val_data': 'processed_data/val_dataset.npz',
    'test_data': 'processed_data/test_dataset.npz',
    'reference_data': 'processed_data/reference_dataset.npz',
    # metadata元数据，按不同丰度存放额外信息，用于训练途中和评估(绘图等)，为预处理代码运行后生成
    'C_FE_csv': 'processed_data/C_FE_metadata.csv',
    'MG_FE_csv': 'processed_data/MG_FE_metadata.csv',
    'CA_FE_csv': 'processed_data/CA_FE_metadata.csv',
    'fits_dir': 'fits',  # FITS文件目录
    'output_dir': 'processed_data'  # 处理后数据输出目录
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
    'wavelength_range': None,  # 设为None，表示使用最大公有波长范围
    'resample_dim': 3000,              # 光谱重采样维度
    'denoise': True,                   # 是否去噪
    'denoise_params': {
        'first_pass': {'window': 5, 'polyorder': 2},  # 第一次去噪参数
        'second_pass': {'window': 7, 'polyorder': 3}  # 第二次去噪参数
    },
    'normalization_method': 'median',  # 归一化方法 (median, minmax, standard)
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
    'model_dir': 'models',  # 模型保存目录
    'input_size': 3907,  # 输入特征维度
    'output_size': 1,  # 输出维度
    'hidden_size': 256,  # 隐藏层大小
    'num_layers': 2,  # 层数
    'dropout': 0.1,  # 降低dropout率
    'num_ensemble_models': 3,  # 集成模型数量
    'dropout_rate': 0.5  # 默认dropout率
}

# 训练配置
training_config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 32,
    'num_epochs': 100,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'early_stopping_patience': 10,
    'elements': ['C_FE', 'MG_FE', 'CA_FE'],  # 要预测的元素列表
    'batch_processing': {
        'enabled': True,  # 是否启用批处理
        'save_results_per_epoch': True,  # 是否每个epoch保存结果
        'save_plots': True,  # 是否保存图表
        'metrics_to_track': ['train_loss', 'val_loss', 'mae', 'rmse', 'r2'],  # 要跟踪的指标
    }
}

# 超参数调优配置
tuning_config = {
    # 传统网格搜索参数 (作为备选)
    'param_grid': {
        'learning_rate': [5e-5, 1e-4, 5e-4],
        'batch_size': [32, 64, 128],
        'weight_decay': [1e-6, 1e-5, 1e-4]
    },
    'early_stopping_patience': 10,     # 调优时的早停耐心值
    
    # 两阶段优化配置
    'two_stage': {
        'enabled': True,               # 是否启用两阶段优化
        'max_evals_stage1': 30,        # 第一阶段最大评估次数
        'max_evals_stage2': 20,        # 第二阶段最大评估次数
        'stage1_epochs': 50,           # 第一阶段每次评估的最大训练轮数
        'stage2_epochs': 100,          # 第二阶段每次评估的最大训练轮数
    },
    
    # 参数搜索空间
    'search_space': {
        'stage1': {                    # 第一阶段广泛搜索
            'lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            'weight_decay': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
            'batch_size': [16, 32, 64, 128],
            'dropout_rate': [0.2, 0.3, 0.4, 0.5, 0.6],
            'patience': [5, 10, 15, 20],
        },
        'stage2_range': {             # 第二阶段精细调整范围 (相对于第一阶段最佳值)
            'lr': [0.5, 2.0],         # 学习率范围系数 [最佳值*0.5, 最佳值*2.0]
            'weight_decay': [0.5, 2.0], 
            'dropout_rate': 0.1,      # 上下浮动范围
        }
    },
    
    # 批量超参数优化配置
    'batch_hyperopt': {
        'enabled': True,              # 是否启用批量超参数优化
        'batch_size': 1000,           # 每批数据大小
        'batches_per_round': 2,       # 每轮处理的批次数
        'max_batches': 10,            # 最大批次数
        'save_intermediates': True,   # 是否保存中间结果
        'generate_two_sets': True,    # 是否生成两组最佳参数
    },
    
    # 调优结果保存
    'save_results': True,             # 是否保存调优结果
    'results_dir': 'results/hyperopt', # 保存目录
    'save_trials': True,              # 是否保存完整的trials对象（大文件）
    'save_plots': True,               # 是否保存调优过程可视化
}

# 评估配置
evaluation_config = {
    'metrics': ['mae', 'mse', 'rmse', 'r2', 'dex'],  # 评估指标
    'plot_types': ['predictions', 'residuals', 'error_distribution'],  # 可视化类型
    'save_predictions': True,  # 是否保存预测结果
    'batch_processing': {
        'enabled': True,  # 是否启用批处理评估
        'batch_size': 32,  # 评估批次大小
        'save_per_batch': True,  # 是否每批次保存结果
        'metrics_to_track': ['mae', 'rmse', 'r2', 'scatter'],  # 要跟踪的指标
    }
}

# 预测配置
prediction_config = {
    'batch_processing': {
        'enabled': True,  # 是否启用批处理预测
        'batch_size': 32,  # 预测批次大小
        'save_per_batch': True,  # 是否每批次保存结果
        'metrics_to_track': ['mean', 'std', 'min', 'max'],  # 要跟踪的统计指标
    }
}

# 基线模型配置
baseline_config = {
    'models': {
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'objective': 'reg:squarederror'
        },
        'lightgbm': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'objective': 'regression'
        }
    },
    'batch_processing': {
        'enabled': True,
        'batch_size': 1000,
        'batches_per_round': 2
    },
    'results_dir': 'results/baseline'
}

# 输出配置
output_config = {
    'log_dir': 'logs',  # 日志目录
    'results_dir': 'results',  # 结果目录
    'plots_dir': 'plots',  # 图表目录
    'predictions_dir': 'results/predictions',  # 预测结果目录
    'evaluation_dir': 'results/evaluation',  # 评估结果目录
    'hyperopt_dir': 'results/hyperopt',  # 超参数优化结果目录
    'baseline_dir': 'results/baseline',  # 基线模型结果目录
    'training_batch_dir': 'results/training_{}_batch_results',  # 训练批次结果目录模板
    'evaluation_batch_dir': 'results/evaluation_{}_batch_results',  # 评估批次结果目录模板
    'prediction_batch_dir': 'results/prediction_{}_batch_results',  # 预测批次结果目录模板
    'baseline_batch_dir': 'results/baseline_{}_{}_batch_results',  # 基线批次结果目录模板（元素_模型）
    'analysis_dir': 'results/analysis',  # 模型分析结果目录
    'cache_dir': 'cache'  # 缓存目录
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

# 创建必要的目录
REQUIRED_DIRS = [
    'fits',
    'processed_data',
    'processed_data/cache',
    'processed_data/progress',
    'models',
    'results',
    'results/predictions',
    'results/evaluation',
    'results/hyperopt',
    'results/baseline',
    'plots',
    'plots/training',
    'plots/evaluation',
    'plots/predictions',
    'logs'
]

for directory in REQUIRED_DIRS:
    os.makedirs(directory, exist_ok=True)
    print(f"已创建目录: {directory}")

# 数据预处理配置
preprocessing_config = {
    'csv_files': ['C_FE.csv', 'MG_FE.csv', 'CA_FE.csv'],  # 要处理的CSV文件列表
    'fits_dir': 'fits',                                  # FITS文件目录
    'output_dir': 'processed_data',                      # 处理后数据输出目录
    'wavelength_range': None,                            # 波长范围(None表示使用最大公有波长范围)
    'n_points': None,                                    # 采样点数(None表示根据波长范围和步长自动计算)
    'log_step': 0.0001,                                  # 对数空间重采样步长(dex)
    'compute_common_range': True,                        # 是否计算最大公有波长范围
    'max_workers': None,                                 # 最大工作进程数(None表示自动确定)
    'batch_size': 20,                                    # 批处理大小
    'memory_limit': 0.7,                                 # 内存使用限制(占总内存比例)
    'low_memory_mode': False,                            # 低内存模式
    'denoise': True,                                     # 是否进行去噪处理
    'normalize': True,                                   # 是否进行归一化处理
    'denoise_params': {                                  # 去噪参数
        'first_pass': {'window': 5, 'polyorder': 2},     # 第一次去噪参数(Savitzky-Golay)
        'second_pass': {'window': 7, 'polyorder': 3}     # 第二次去噪参数
    },
    'normalization_method': 'median',                    # 归一化方法
    'split_dataset': True                                # 是否划分数据集
}

# 缓存配置
cache_config = {
    'checkpoint_interval': 100,  # 每处理100个样本保存一次检查点
    'validation_metrics': {
        'min_data_size': 100,    # 最小数据量
        'max_memory_usage': 0.8,  # 最大内存使用比例
        'required_fields': ['data', 'metadata', 'validation_metrics']
    },
    'cache_dirs': {
        'preprocessing': 'processed_data/cache/preprocessing',
        'training': 'processed_data/cache/train',
        'evaluation': 'processed_data/cache/evaluation',
        'prediction': 'processed_data/cache/predict',
        'hyperopt': 'processed_data/cache/hyperopt',
        'baseline': 'processed_data/cache/baseline'
    },
    'file_format': 'npz',  # 缓存文件格式
    'compression': True,   # 是否压缩缓存文件
    'max_cache_size': 1000  # 最大缓存文件数
}

# 进度管理配置
progress_config = {
    'update_interval': 1,  # 进度更新间隔（秒）
    'save_interval': 10,   # 进度保存间隔（秒）
    'progress_dir': 'processed_data/progress',  # 进度保存目录
    'display': {
        'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
        'show_details': True,  # 是否显示详细信息
        'show_time': True,     # 是否显示时间信息
        'show_memory': True    # 是否显示内存使用情况
    }
}

# 添加模型分析配置
analysis_config = {
    'enabled': True,                              # 是否启用模型性能分析
    'batch_size': 32,                             # 分析批处理大小
    'feature_importance': {
        'enabled': True,                          # 是否分析特征重要性
        'sample_size': 1000,                      # 用于分析的样本数
        'top_n_features': 30,                     # 显示前N个重要特征
        'min_importance_threshold': 0.05,         # 最小重要性阈值
    },
    'residual_analysis': {
        'enabled': True,                          # 是否进行残差分析
        'save_predictions': True,                 # 是否保存预测结果
        'plot_qq': True,                          # 是否绘制QQ图
        'plot_hist': True,                        # 是否绘制直方图
    },
    'batch_results': {
        'enabled': True,                          # 是否启用批处理结果
        'save_batch_results': True,               # 是否保存批处理结果
        'generate_batch_summary': True,           # 是否生成批处理摘要
        'generate_trend_plots': True,             # 是否生成趋势图
    },
    'elements': ['C_FE', 'MG_FE', 'CA_FE'],      # 要分析的元素列表
    'results_dir': 'results/analysis'             # 分析结果目录
}

# 将所有配置合并到一个字典中
CONFIG = {
    'data_paths': DATA_CONFIG,
    'data_config': data_config,
    'model_config': model_config,
    'training_config': training_config,
    'tuning_config': tuning_config,
    'evaluation_config': evaluation_config,
    'prediction_config': prediction_config,
    'baseline_config': baseline_config,
    'output_config': output_config,
    'mist_isochrones': mist_isochrones,
    'element_config': element_config,
    'preprocessing_config': preprocessing_config,
    'cache_config': cache_config,
    'progress_config': progress_config,
    'analysis_config': analysis_config
}

class Config:
    # 预处理配置
    PREPROCESSING_BATCH_SIZE = 20
    
    # 训练配置
    TRAINING_BATCH_SIZE = 32
    EPOCHS = 100
    
    # 基线模型配置
    BASELINE_BATCH_SIZE = 1000
    BASELINE_BATCHES_PER_ROUND = 2
    
    # 评估配置
    EVALUATION_BATCH_SIZE = 32
    
    # 预测配置
    PREDICTION_BATCH_SIZE = 100
    
    # 批处理结果配置
    SAVE_BATCH_RESULTS = True
    GENERATE_VISUALIZATIONS = True
    
    # 环境配置
    USE_GPU = True  # 自动检测
    MAX_WORKERS = 4  # 并行处理的工作线程数
    MEMORY_LIMIT = 0.7  # 内存使用限制百分比 