import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from scipy import stats
from scipy.signal import find_peaks
import pandas as pd
import time
from config import analysis_config
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# 设置日志
logger = logging.getLogger(__name__)

def get_element_absorption_lines(element):
    """获取元素的主要吸收线波长数据"""
    # 常见元素吸收线波长（埃）
    element_lines = {
        'FE': {
            'Fe I 4045': 4045.81, 'Fe I 4271': 4271.76, 'Fe I 4383': 4383.55, 
            'Fe I 4404': 4404.75, 'Fe I 5269': 5269.54, 'Fe I 5328': 5328.04,
            'Fe I 5371': 5371.49, 'Fe I 5397': 5397.13, 'Fe I 5405': 5405.77,
            'Fe I 5576': 5576.09, 'Fe I 6137': 6137.69, 'Fe I 6191': 6191.56,
            'Fe I 6393': 6393.60, 'Fe I 6411': 6411.65, 'Fe I 6430': 6430.85,
            'Fe II 4233': 4233.17, 'Fe II 4923': 4923.93, 'Fe II 5018': 5018.44,
            'Fe II 5169': 5169.03, 'Fe II 5316': 5316.62
        },
        'C_FE': {
            'C I 4771': 4771.74, 'C I 5052': 5052.17, 'C I 5380': 5380.34,
            'C I 8335': 8335.15, 'CH 4300': 4300.00, 'C2 5165': 5165.00,
            'C2 5635': 5635.00
        },
        'MG_FE': {
            'Mg I 4571': 4571.10, 'Mg I 4703': 4702.99, 'Mg I 5167': 5167.32,
            'Mg I 5173': 5172.68, 'Mg I 5184': 5183.60, 'Mg I 5528': 5528.41,
            'Mg I 8806': 8806.76, 'Mg II 4481': 4481.13
        },
        'CA_FE': {
            'Ca I 4226': 4226.73, 'Ca I 4425': 4425.44, 'Ca I 4455': 4454.78,
            'Ca I 5589': 5588.75, 'Ca I 5857': 5857.45, 'Ca I 6102': 6102.72,
            'Ca I 6122': 6122.22, 'Ca I 6162': 6162.17, 'Ca I 6439': 6439.08,
            'Ca II 3934': 3933.66, 'Ca II 3969': 3968.47, 'Ca II 8498': 8498.02,
            'Ca II 8542': 8542.09, 'Ca II 8662': 8662.14
        },
        'SI_FE': {
            'Si I 5701': 5701.10, 'Si I 5772': 5772.15, 'Si I 5948': 5948.55,
            'Si I 6155': 6155.13, 'Si I 7034': 7034.90, 'Si I 7944': 7944.00,
            'Si II 6347': 6347.10, 'Si II 6371': 6371.36
        },
        'TI_FE': {
            'Ti I 4533': 4533.24, 'Ti I 4981': 4981.73, 'Ti I 4991': 4991.07,
            'Ti I 5007': 5007.21, 'Ti I 5039': 5038.40, 'Ti I 5173': 5173.74,
            'Ti I 5210': 5210.39, 'Ti II 4395': 4395.03, 'Ti II 4501': 4501.27,
            'Ti II 4534': 4533.96, 'Ti II 4563': 4563.76, 'Ti II 4590': 4589.96
        }
    }
    
    # 处理元素名称，确保与字典键匹配
    element = element.upper()
    if element in element_lines:
        return element_lines[element]
    else:
        # 如果没有特定元素的数据，返回空字典
        logger.warning(f"未找到元素{element}的吸收线数据")
        return {}

def identify_spectral_feature(wavelength, element):
    """根据波长识别可能的光谱特征"""
    # 获取元素吸收线
    element_lines = get_element_absorption_lines(element)
    
    # 检查是否在已知吸收线附近(±5埃)
    for line_name, line_wave in element_lines.items():
        if abs(wavelength - line_wave) < 5:
            return f"{line_name} ({line_wave:.2f}Å)"
    
    # 常见分子带和其他特征
    molecular_bands = {
        'CH G-band': (4290, 4315),
        'CN': [(3850, 3880), (4150, 4215)],
        'C2 Swan': [(5165, 5265), (5535, 5635)],
        'MgH': (4780, 4820),
        'TiO': [(4955, 4975), (5445, 5480), (5730, 5770), (6150, 6250)],
        'Hα': (6563, 6563),
        'Hβ': (4861, 4861),
        'Hγ': (4340, 4340),
        'Hδ': (4101, 4101),
        'Ca II H&K': (3933, 3969),
        'Ca II Triplet': (8498, 8662),
        'Na D': (5889, 5896),
        'Mg b triplet': (5167, 5184)
    }
    
    for band_name, band_range in molecular_bands.items():
        if isinstance(band_range, tuple):
            if band_range[0] <= wavelength <= band_range[1]:
                return f"{band_name} ({band_range[0]}-{band_range[1]}Å)"
        elif isinstance(band_range, list):
            for sub_range in band_range:
                if sub_range[0] <= wavelength <= sub_range[1]:
                    return f"{band_name} ({sub_range[0]}-{sub_range[1]}Å)"
    
    # 波长区域划分
    if 3800 <= wavelength < 4500:
        return "蓝端区域 (很多金属线和氢线)"
    elif 4500 <= wavelength < 5500:
        return "绿端区域 (金属线丰富)"
    elif 5500 <= wavelength < 6500:
        return "黄端区域 (中等强度金属线)"
    elif 6500 <= wavelength < 8000:
        return "红端区域 (相对较少金属线)"
    else:
        return "近红外区域 (主要含Ca II三重线和分子带)"

def get_feature_explanation(wavelength, element):
    """提供波长区域特征的天文物理解释"""
    feature = identify_spectral_feature(wavelength, element)
    
    # 基于元素和特征提供解释
    explanations = {
        'FE': {
            'Fe I': "中性铁吸收线，在金属丰度[Fe/H]估计中起主要作用",
            'Fe II': "电离铁吸收线，对恒星表面重力较敏感，有助于区分矮星和巨星",
        },
        'C_FE': {
            'C I': "中性碳吸收线，较弱但对碳丰度很敏感",
            'CH': "CH分子带，是碳丰度的良好示踪物，尤其在低金属丰度恒星中",
            'C2': "C₂分子带，在碳星中特别明显",
        },
        'MG_FE': {
            'Mg I': "中性镁吸收线，α元素丰度的重要指标",
            'Mg II': "电离镁吸收线，对温度较敏感",
            'MgH': "氢化镁分子带，在低温恒星中较强",
        },
        'CA_FE': {
            'Ca I': "中性钙吸收线，是另一个重要的α元素示踪物",
            'Ca II': "电离钙，特别是H&K线和红外三重线是恒星色球活动的良好指标",
        },
        'SI_FE': {
            'Si I': "中性硅吸收线，是α元素家族成员，在恒星形成历史中具有重要意义",
            'Si II': "电离硅，在较热恒星中更明显",
        },
        'TI_FE': {
            'Ti I': "中性钛吸收线，α元素示踪物",
            'Ti II': "电离钛，在金属贫恒星中相对更强",
        }
    }
    
    element_key = element.upper()
    
    for key, explanation in explanations.get(element_key, {}).items():
        if key in feature:
            return explanation
    
    # 其他特征的通用解释
    if "G-band" in feature:
        return "CH G-band，在4300Å附近的强特征，是碳丰度的良好示踪物"
    elif "CN" in feature:
        return "CN分子带，对碳和氮丰度敏感，也依赖于氧丰度"
    elif "TiO" in feature:
        return "TiO分子带，在红端非常明显，是低温恒星（如M型星）的主要特征"
    elif "H" in feature and any(h in feature for h in ["α", "β", "γ", "δ"]):
        return "巴尔末系列氢线，主要受恒星有效温度影响"
    elif "Ca II H&K" in feature:
        return "Ca II H&K线，对恒星色球活动极为敏感，可用于测量恒星活动和年龄"
    elif "Ca II Triplet" in feature:
        return "Ca II三重线，对恒星表面重力敏感，常用于区分矮星和巨星"
    elif "Na D" in feature:
        return "钠D线双峰，强度受金属丰度和恒星表面重力影响"
    elif "Mg b" in feature:
        return "镁b三重线，是重要的表面重力指标"
    elif "蓝端区域" in feature:
        return "蓝端区域含有大量金属吸收线，对金属丰度非常敏感"
    elif "绿端区域" in feature:
        return "绿端包含许多中等强度的金属线，是丰度分析的理想区域"
    elif "红端区域" in feature:
        return "红端区域金属线相对较少，但包含重要的分子特征"
    else:
        return "此波长区域可能包含多种元素的混合贡献，或受连续谱形状影响"

def analyze_feature_importance(model, data_loader, device, element, 
                              batch_id=None, save_results=True, output_dir='results/feature_importance'):
    """
    分析模型对输入特征的重要性
    """
    # 检查模型类型
    model_type = model.__class__.__name__
    logger.info(f"分析模型类型: {model_type}")
    
    # 准备保存分析结果的目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储每个特征的重要性分数
    importances = []
    
    # 原始参考预测
    reference_predictions = []
    
    # 遍历数据加载器中的数据
    model.eval()
    
    with torch.no_grad():
        for batch_spectra, batch_labels in data_loader:
            batch_spectra = batch_spectra.to(device)
            batch_labels = batch_labels.to(device)
            
            # 获取原始预测
            original_predictions = model(batch_spectra).squeeze()
            reference_predictions.append(original_predictions.cpu().numpy())
            
            # 对于每个波长点
            feature_importance = []
            spectra_numpy = batch_spectra.cpu().numpy()
            
            # 特征掩蔽方法：对于每个特征，将其设为0并测量预测变化
            seq_len = spectra_numpy.shape[2]
            
            # 为了提高效率，我们只分析一部分波长点
            # 对于非常长的光谱，每10个点分析一个
            step = max(1, seq_len // 100)
            
            for i in range(0, seq_len, step):
                # 创建掩蔽数据
                masked_spectra = spectra_numpy.copy()
                masked_spectra[:, :, i] = 0  # 掩蔽当前波长点
                
                # 转换回torch张量
                masked_tensor = torch.FloatTensor(masked_spectra).to(device)
                
                # 预测
                masked_predictions = model(masked_tensor).squeeze()
                
                # 计算特征重要性（预测变化的绝对值）
                importance = torch.abs(original_predictions - masked_predictions).mean().item()
                feature_importance.append((i, importance))
            
            importances.extend(feature_importance)
            break  # 只分析第一个批次的数据
    
    # 将结果转换为DataFrame
    importance_df = pd.DataFrame(importances, columns=['wavelength_idx', 'importance'])
    
    # 对于相同波长点的多个样本，取平均值
    importance_df = importance_df.groupby('wavelength_idx').mean().reset_index()
    
    # 排序并获取前20个重要特征
    top_features = importance_df.sort_values('importance', ascending=False).head(20)
    
    # 绘制特征重要性图
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(top_features)), top_features['importance'], color='skyblue')
    plt.xlabel('波长索引', fontsize=12)
    plt.ylabel('特征重要性', fontsize=12)
    plt.title(f'{element}元素的Top20特征重要性', fontsize=14)
    plt.xticks(range(len(top_features)), top_features['wavelength_idx'])
    plt.grid(axis='y', alpha=0.3)
    
    # 保存图表
    plt.tight_layout()
    if batch_id is not None:
        plot_path = os.path.join(output_dir, f'{element}_feature_importance_batch{batch_id}.png')
    else:
        plot_path = os.path.join(output_dir, f'{element}_feature_importance.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    # 为重要特征添加物理解释
    # 这需要将波长索引转换为实际波长值
    # 假设我们有一个波长范围的映射函数
    
    # 创建一个更详细的解释报告
    report = []
    for idx, row in top_features.iterrows():
        wavelength_idx = int(row['wavelength_idx'])
        importance = row['importance']
        
        # 这里需要将波长索引转换为实际波长值
        # 这是一个示例，实际应用中需要根据数据集的具体波长范围进行调整
        # 假设3500-10000埃的范围，均匀分布在input_size点上
        input_size = data_loader.dataset[0][0].shape[-1]  # 假设是批次中第一个样本的特征数
        wavelength = 3500 + (10000 - 3500) * wavelength_idx / input_size
        
        # 获取特征解释
        feature = identify_spectral_feature(wavelength, element)
        explanation = get_feature_explanation(wavelength, element)
        
        report.append({
            'wavelength_idx': wavelength_idx,
            'estimated_wavelength': wavelength,
            'importance': importance,
            'feature': feature,
            'explanation': explanation
        })
    
    # 将报告转换为DataFrame并保存
    report_df = pd.DataFrame(report)
    if batch_id is not None:
        report_path = os.path.join(output_dir, f'{element}_feature_importance_report_batch{batch_id}.csv')
    else:
        report_path = os.path.join(output_dir, f'{element}_feature_importance_report.csv')
    report_df.to_csv(report_path, index=False)
    
    # 如果需要保存所有特征的重要性
    if save_results:
        if batch_id is not None:
            importance_path = os.path.join(output_dir, f'{element}_all_features_importance_batch{batch_id}.csv')
        else:
            importance_path = os.path.join(output_dir, f'{element}_all_features_importance.csv')
        importance_df.to_csv(importance_path, index=False)
    
    logger.info(f"特征重要性分析完成，结果保存至 {output_dir}")
    
    return top_features

def analyze_residuals(model, data_loader, device, element, 
                   batch_id=None, save_results=True, output_dir='results/residual_analysis'):
    """
    分析模型预测的残差
    """
    model.eval()
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # 处理输入数据中的NaN值
            data_np = data.cpu().numpy()
            has_nan = np.isnan(data_np).any()
            if has_nan:
                logger.warning(f"Input data contains NaN values, attempting to replace with mean")
                # 计算每个特征的均值，忽略NaN值
                feature_means = np.nanmean(data_np, axis=0)
                # 用均值替换NaN值
                nan_mask = np.isnan(data_np)
                for i in range(data_np.shape[0]):
                    sample_nan_mask = nan_mask[i]
                    if sample_nan_mask.any():
                        data_np[i][sample_nan_mask] = feature_means[sample_nan_mask]
                
                # 将处理后的数据转回tensor
                data = torch.tensor(data_np, dtype=torch.float32).to(device)
            
            # 模型预测
            try:
                output = model(data)
                
                # 收集真实值和预测值
                all_true.extend(target.cpu().numpy().flatten())
                all_pred.extend(output.cpu().numpy().flatten())
            except Exception as e:
                logger.error(f"Error during residual analysis prediction: {str(e)}")
                continue
    
    # 转换为numpy数组
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    
    # 过滤NaN值
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not valid_mask.all():
        logger.warning(f"Found {(~valid_mask).sum()} NaN values in true or predicted values, filtering them out")
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
    
    # 如果没有有效数据，返回空结果
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.error("No valid data for residual analysis")
        return {
            'rmse': float('nan'),
            'mae': float('nan'),
            'r2': float('nan'),
            'residual_mean': float('nan'),
            'residual_std': float('nan'),
            'residual_max': float('nan'),
            'residual_min': float('nan')
        }
    
    # 计算残差
    residuals = y_pred - y_true
    
    # 计算评估指标
    try:
        mse = np.mean((y_pred - y_true) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred - y_true))
        
        # 使用手动计算的R²，避免使用sklearn可能出现的问题
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total if ss_total > 0 else 0)
        
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        residual_max = np.max(residuals)
        residual_min = np.min(residuals)
    except Exception as e:
        logger.error(f"Error calculating residual metrics: {str(e)}")
        return {
            'rmse': float('nan'),
            'mae': float('nan'),
            'r2': float('nan'),
            'residual_mean': float('nan'),
            'residual_std': float('nan'),
            'residual_max': float('nan'),
            'residual_min': float('nan')
        }
    
    # 保存结果
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        
        # 批次ID信息
        batch_str = f"_batch_{batch_id}" if batch_id is not None else ""
        
        # 残差分析图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('True Values')
        plt.ylabel('Residuals')
        plt.title(f'Residual Analysis for {element}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{element}{batch_str}_residuals.png'))
        plt.close()
        
        # 残差直方图
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.axvline(x=residual_mean, color='g', linestyle='-')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.title(f'Residual Distribution for {element} (Mean: {residual_mean:.4f}, Std: {residual_std:.4f})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{element}{batch_str}_residual_hist.png'))
        plt.close()
        
        # 真实值vs预测值散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        
        # 添加完美预测线
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'True vs Predicted for {element} (RMSE: {rmse:.4f}, R²: {r2:.4f})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{element}{batch_str}_prediction.png'))
        plt.close()
        
        # 保存指标结果
        with open(os.path.join(output_dir, f'{element}{batch_str}_metrics.txt'), 'w') as f:
            f.write(f"Residual Analysis Results for {element}\n")
            f.write("=" * 50 + "\n")
            f.write(f"RMSE: {rmse:.6f}\n")
            f.write(f"MAE: {mae:.6f}\n")
            f.write(f"R²: {r2:.6f}\n")
            f.write(f"Residual Mean: {residual_mean:.6f}\n")
            f.write(f"Residual Std: {residual_std:.6f}\n")
            f.write(f"Residual Max: {residual_max:.6f}\n")
            f.write(f"Residual Min: {residual_min:.6f}\n")
    
    # 返回指标
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'residual_mean': residual_mean,
        'residual_std': residual_std,
        'residual_max': residual_max,
        'residual_min': residual_min
    }

def analyze_model_performance(model, element, train_loader, val_loader, test_loader, device, 
                              save_feature_importance=True, 
                              save_batch_results=True,
                              batch_id=None,
                              batch_size=None,
                              output_dir='results',
                              ):
    """
    对模型性能进行全面分析
    """
    os.makedirs("results/feature_importance", exist_ok=True)
    os.makedirs("results/residual_analysis", exist_ok=True)
    os.makedirs("results/gcn_analysis", exist_ok=True)
    
    # 检查模型类型
    model_type = model.__class__.__name__
    logger.info(f"开始分析 {model_type} 模型性能...")
    
    # 特征重要性分析
    logger.info(f"开始特征重要性分析，针对 {element} 元素...")
    importance_results = analyze_feature_importance(model, val_loader, device, element,
                                                  batch_id=batch_id,
                                                  save_results=save_batch_results)
    
    # 残差分析
    logger.info(f"开始残差分析，针对 {element} 元素...")
    residual_analysis = analyze_residuals(model, test_loader, device, element,
                                          batch_id=batch_id,
                                          save_results=save_batch_results)
    
    # GCN模型分析（如果适用）
    gcn_analysis = None
    if hasattr(model, 'gcn_layer') and hasattr(model, 'attention'):
        logger.info(f"检测到GCN模型，开始GCN注意力机制分析...")
        gcn_analysis = analyze_gcn_attention(model, val_loader, device, element,
                                           batch_id=batch_id,
                                           save_results=save_batch_results)
    
    logger.info(f"{element} 模型分析完成，结果保存在 {output_dir} 目录中")
    
    # 合并结果
    combined_metrics = {}
    
    # 添加残差分析的指标
    if residual_analysis:
        combined_metrics.update({
            'rmse': residual_analysis.get('rmse', float('nan')),
            'mae': residual_analysis.get('mae', float('nan')),
            'r2': residual_analysis.get('r2', float('nan')),
            'residual_mean': residual_analysis.get('residual_mean', float('nan')),
            'residual_std': residual_analysis.get('residual_std', float('nan'))
        })
    
    # 添加特征重要性分析的指标
    if importance_results is not None:
        if isinstance(importance_results, pd.DataFrame) and not importance_results.empty:
            combined_metrics.update({
                'top_feature_importance': importance_results.iloc[0]['importance'],
                'top_feature_index': importance_results.iloc[0]['wavelength_idx']
            })
        else:
            combined_metrics.update({
                'top_feature_importance': float('nan'),
                'top_feature_index': -1
            })
    
    # 添加GCN分析的指标（如果有）
    if gcn_analysis:
        if 'attention' in gcn_analysis:
            combined_metrics.update({
                'max_attention_idx': gcn_analysis['attention'].get('max_attention_idx', -1),
                'attention_mean': np.mean(gcn_analysis['attention'].get('mean', [0])),
                'attention_std': np.mean(gcn_analysis['attention'].get('std', [0]))
            })
            
        if 'adjacency' in gcn_analysis:
            combined_metrics.update({
                'adjacency_density': gcn_analysis['adjacency'].get('density', float('nan')),
                'adjacency_sparsity': gcn_analysis['adjacency'].get('sparsity', float('nan')),
                'mean_edge_weight': gcn_analysis['adjacency'].get('mean_weight', float('nan'))
            })
    
    # 生成综合报告
    summary_path = os.path.join(output_dir, f"{element}_performance_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"性能分析摘要 - {element}元素 ({model_type}模型)\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("模型类型: " + model_type + "\n\n")
        
        f.write("指标汇总:\n")
        for metric, value in combined_metrics.items():
            if isinstance(value, float):
                f.write(f"{metric}: {value:.6f}\n")
            else:
                f.write(f"{metric}: {value}\n")
        
        f.write("\n分析详情:\n")
        if residual_analysis:
            f.write("残差分析:\n")
            f.write(f"- RMSE: {residual_analysis.get('rmse', 'N/A'):.6f}\n")
            f.write(f"- MAE: {residual_analysis.get('mae', 'N/A'):.6f}\n")
            f.write(f"- R²: {residual_analysis.get('r2', 'N/A'):.6f}\n")
        
        if importance_results is not None:
            f.write("\n特征重要性分析:\n")
            if isinstance(importance_results, pd.DataFrame) and not importance_results.empty:
                f.write(f"- 最重要特征索引: {importance_results.iloc[0]['wavelength_idx']}\n")
                f.write(f"- 最重要特征分数: {importance_results.iloc[0]['importance']:.6f}\n")
        
        if gcn_analysis:
            f.write("\nGCN模型分析:\n")
            if 'attention' in gcn_analysis:
                f.write(f"- 最高注意力索引: {gcn_analysis['attention'].get('max_attention_idx', -1)}\n")
                f.write(f"- 平均注意力权重: {np.mean(gcn_analysis['attention'].get('mean', [0])):.6f}\n")
            
            if 'adjacency' in gcn_analysis:
                f.write(f"- 邻接矩阵稀疏度: {gcn_analysis['adjacency'].get('sparsity', 'N/A'):.6f}\n")
                f.write(f"- 平均边权重: {gcn_analysis['adjacency'].get('mean_weight', 'N/A'):.6f}\n")
    
    results = {}
    results['summary'] = summary_path
    results['metrics'] = combined_metrics
    results['model_type'] = model_type
    
    # 输出结果摘要
    logger.info(f"{element}元素的{model_type}模型性能分析完成")
    if residual_analysis:
        logger.info(f"RMSE: {residual_analysis.get('rmse', 'N/A'):.4f}, R²: {residual_analysis.get('r2', 'N/A'):.4f}")
    
    if gcn_analysis:
        logger.info(f"GCN分析完成，注意力和图结构信息已保存")
    
    logger.info(f"分析结果已保存至 {output_dir} 目录")
    
    return results

# 批次结果目录结构
def setup_batch_directories(element):
    """创建批次结果目录结构"""
    batch_dirs = {
        'feature_importance': os.path.join("results", "feature_importance", f"{element}_batch_results"),
        'residual_analysis': os.path.join("results", "residual_analysis", f"{element}_batch_results"),
        'gcn_analysis': os.path.join("results", "gcn_analysis", f"{element}_batch_results"),
        'batch_tracking': os.path.join("results", "batch_tracking"),
    }
    
    for directory in batch_dirs.values():
        os.makedirs(directory, exist_ok=True)
    
    return batch_dirs

class BatchTracker:
    """批处理跟踪器，用于管理分批分析的结果"""
    def __init__(self, element, analysis_type="feature_importance"):
        """初始化批处理跟踪器
        
        Args:
            element: 元素名称
            analysis_type: 分析类型 ("feature_importance"、"residual_analysis" 或 "gcn_analysis")
        """
        self.element = element
        self.analysis_type = analysis_type
        self.batch_dirs = setup_batch_directories(element)
        
        # 创建或加载批次跟踪文件
        self.tracking_file = os.path.join(
            self.batch_dirs[analysis_type], 
            f"batch_tracking.csv"
        )
        
        if os.path.exists(self.tracking_file):
            self.tracking_df = pd.read_csv(self.tracking_file)
            # 确保批次ID列存在且为整数类型
            if 'batch_id' in self.tracking_df.columns:
                self.tracking_df['batch_id'] = self.tracking_df['batch_id'].astype(int)
            else:
                self.tracking_df['batch_id'] = list(range(1, len(self.tracking_df) + 1))
        else:
            # 初始化空的跟踪DataFrame
            if analysis_type == "feature_importance":
                self.tracking_df = pd.DataFrame(columns=[
                    'batch_id', 'timestamp', 'num_samples', 'top_feature_wavelength', 
                    'top_feature_importance', 'top_feature_name', 'processing_time'
                ])
            elif analysis_type == "residual_analysis":
                self.tracking_df = pd.DataFrame(columns=[
                    'batch_id', 'timestamp', 'num_samples', 'rmse', 'mae', 'r2', 
                    'residual_mean', 'residual_std', 'processing_time'
                ])
            elif analysis_type == "gcn_analysis":
                self.tracking_df = pd.DataFrame(columns=[
                    'batch_id', 'timestamp', 'num_samples', 'max_attention_idx', 
                    'attention_mean', 'attention_std', 'adjacency_sparsity',
                    'mean_edge_weight', 'processing_time'
                ])
            else:
                # 默认跟踪列
                self.tracking_df = pd.DataFrame(columns=[
                    'batch_id', 'timestamp', 'num_samples', 'processing_time'
                ])
    
    def get_next_batch_id(self):
        """获取下一个批次ID"""
        if len(self.tracking_df) == 0:
            return 1
        return max(self.tracking_df['batch_id']) + 1
    
    def add_batch_result(self, batch_result):
        """添加批次结果
        
        Args:
            batch_result: 包含批次结果数据的字典
        """
        # 确保有批次ID
        if 'batch_id' not in batch_result:
            batch_result['batch_id'] = self.get_next_batch_id()
        
        # 添加时间戳
        if 'timestamp' not in batch_result:
            batch_result['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # 添加到跟踪DataFrame
        self.tracking_df = pd.concat([
            self.tracking_df, 
            pd.DataFrame([batch_result])
        ], ignore_index=True)
        
        # 保存更新后的跟踪文件
        self.tracking_df.to_csv(self.tracking_file, index=False)
        
        return batch_result['batch_id']
    
    def get_batch_result(self, batch_id):
        """获取指定批次的结果
        
        Args:
            batch_id: 批次ID
            
        Returns:
            批次结果的字典或None
        """
        batch = self.tracking_df[self.tracking_df['batch_id'] == batch_id]
        if len(batch) == 0:
            return None
        return batch.iloc[0].to_dict()
    
    def get_all_batch_results(self):
        """获取所有批次的结果
        
        Returns:
            所有批次结果的DataFrame
        """
        return self.tracking_df.copy()
    
    def generate_trend_plots(self):
        """生成批次趋势图"""
        if len(self.tracking_df) < 2:
            logger.warning(f"批次数量不足，无法生成趋势图")
            return None
        
        plt.figure(figsize=(12, 8))
        
        if self.analysis_type == "feature_importance":
            # 绘制特征重要性趋势
            plt.subplot(2, 1, 1)
            plt.plot(self.tracking_df['batch_id'], self.tracking_df['top_feature_importance'], 'o-')
            plt.title(f'Feature Importance Trend for {self.element}')
            plt.xlabel('批次ID')
            plt.ylabel('顶级特征重要性分数')
            plt.grid(True, alpha=0.3)
            
            # 绘制处理时间趋势
            plt.subplot(2, 1, 2)
            plt.plot(self.tracking_df['batch_id'], self.tracking_df['processing_time'], 'o-')
            plt.title('Processing Time Trend')
            plt.xlabel('批次ID')
            plt.ylabel('处理时间 (秒)')
            plt.grid(True, alpha=0.3)
            
        elif self.analysis_type == "residual_analysis":
            # 绘制RMSE和MAE趋势
            plt.subplot(2, 2, 1)
            plt.plot(self.tracking_df['batch_id'], self.tracking_df['rmse'], 'o-', label='RMSE')
            plt.plot(self.tracking_df['batch_id'], self.tracking_df['mae'], 's-', label='MAE')
            plt.title(f'Error Metrics Trend for {self.element}')
            plt.xlabel('批次ID')
            plt.ylabel('误差值')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 绘制R²趋势
            plt.subplot(2, 2, 2)
            plt.plot(self.tracking_df['batch_id'], self.tracking_df['r2'], 'o-')
            plt.title('R² Trend')
            plt.xlabel('批次ID')
            plt.ylabel('R²')
            plt.grid(True, alpha=0.3)
            
            # 绘制残差均值趋势
            plt.subplot(2, 2, 3)
            plt.plot(self.tracking_df['batch_id'], self.tracking_df['residual_mean'], 'o-')
            plt.axhline(0, color='r', linestyle='--', alpha=0.3)
            plt.title('Residual Mean Trend')
            plt.xlabel('批次ID')
            plt.ylabel('残差均值')
            plt.grid(True, alpha=0.3)
            
            # 绘制残差标准差趋势
            plt.subplot(2, 2, 4)
            plt.plot(self.tracking_df['batch_id'], self.tracking_df['residual_std'], 'o-')
            plt.title('Residual Standard Deviation Trend')
            plt.xlabel('批次ID')
            plt.ylabel('标准差')
            plt.grid(True, alpha=0.3)
        
        elif self.analysis_type == "gcn_analysis":
            # 绘制注意力平均值趋势
            plt.subplot(2, 2, 1)
            plt.plot(self.tracking_df['batch_id'], self.tracking_df['attention_mean'], 'o-', color='blue')
            plt.title(f'注意力均值趋势 - {self.element}')
            plt.xlabel('批次ID')
            plt.ylabel('注意力均值')
            plt.grid(True, alpha=0.3)
            
            # 绘制注意力标准差趋势
            plt.subplot(2, 2, 2)
            plt.plot(self.tracking_df['batch_id'], self.tracking_df['attention_std'], 'o-', color='green')
            plt.title('注意力标准差趋势')
            plt.xlabel('批次ID')
            plt.ylabel('标准差')
            plt.grid(True, alpha=0.3)
            
            # 绘制邻接矩阵稀疏度趋势
            plt.subplot(2, 2, 3)
            plt.plot(self.tracking_df['batch_id'], self.tracking_df['adjacency_sparsity'], 'o-', color='purple')
            plt.title('邻接矩阵稀疏度趋势')
            plt.xlabel('批次ID')
            plt.ylabel('稀疏度')
            plt.grid(True, alpha=0.3)
            
            # 绘制平均边权重趋势
            plt.subplot(2, 2, 4)
            plt.plot(self.tracking_df['batch_id'], self.tracking_df['mean_edge_weight'], 'o-', color='orange')
            plt.title('平均边权重趋势')
            plt.xlabel('批次ID')
            plt.ylabel('权重值')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        trend_plot_path = os.path.join(
            self.batch_dirs[self.analysis_type], 
            f"{self.element}_batch_trends.png"
        )
        plt.savefig(trend_plot_path, dpi=300)
        plt.close()
        
        return trend_plot_path
    
    def generate_batch_summary(self):
        """生成批次分析总结"""
        if len(self.tracking_df) == 0:
            logger.warning(f"没有批次数据，无法生成总结")
            return None
        
        summary_path = os.path.join(
            self.batch_dirs[self.analysis_type], 
            f"{self.element}_batch_summary.txt"
        )
        
        with open(summary_path, 'w') as f:
            f.write(f"# {self.element}元素 {self.analysis_type} 批次分析总结\n\n")
            f.write(f"总批次数: {len(self.tracking_df)}\n")
            f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if self.analysis_type == "feature_importance":
                # 检查top_feature_importance列是否存在或全为NaN
                if 'top_feature_importance' not in self.tracking_df.columns:
                    f.write("警告: 没有找到特征重要性数据\n\n")
                elif self.tracking_df['top_feature_importance'].isna().all():
                    f.write("警告: 所有特征重要性值均为NaN\n\n")
                else:
                    # 找出最重要的特征所在批次，排除NaN值
                    valid_rows = self.tracking_df['top_feature_importance'].notna()
                    if valid_rows.any():
                        best_batch = self.tracking_df.loc[self.tracking_df.loc[valid_rows, 'top_feature_importance'].idxmax()]
                        f.write(f"最强特征重要性批次: {best_batch['batch_id']} (分数: {best_batch['top_feature_importance']:.4f})\n")
                        
                        # 确保top_feature_wavelength和top_feature_name存在
                        if 'top_feature_wavelength' in best_batch:
                            f.write(f"特征波长: {best_batch['top_feature_wavelength']:.2f}Å\n")
                        
                        if 'top_feature_name' in best_batch:
                            f.write(f"特征名称: {best_batch['top_feature_name']}\n\n")
                
                # 批次处理时间统计，如果存在processing_time列
                if 'processing_time' in self.tracking_df.columns and not self.tracking_df['processing_time'].isna().all():
                    f.write(f"平均处理时间: {self.tracking_df['processing_time'].mean():.2f}秒\n")
                
            else:  # residual_analysis
                # 找出RMSE最小的批次，处理可能的NaN值
                if 'rmse' in self.tracking_df.columns and not self.tracking_df['rmse'].isna().all():
                    valid_rmse = self.tracking_df['rmse'].notna()
                    if valid_rmse.any():
                        best_rmse_batch = self.tracking_df.loc[self.tracking_df.loc[valid_rmse, 'rmse'].idxmin()]
                        f.write(f"最佳RMSE批次: {best_rmse_batch['batch_id']} (RMSE: {best_rmse_batch['rmse']:.4f})\n")
                
                # 找出R²最大的批次，处理可能的NaN值
                if 'r2' in self.tracking_df.columns and not self.tracking_df['r2'].isna().all():
                    valid_r2 = self.tracking_df['r2'].notna()
                    if valid_r2.any():
                        best_r2_batch = self.tracking_df.loc[self.tracking_df.loc[valid_r2, 'r2'].idxmax()]
                        f.write(f"最佳R²批次: {best_r2_batch['batch_id']} (R²: {best_r2_batch['r2']:.4f})\n\n")
                
                # 添加平均指标
                f.write("### 平均性能指标\n")
                for metric in ['rmse', 'mae', 'r2', 'residual_mean', 'residual_std']:
                    if metric in self.tracking_df.columns and not self.tracking_df[metric].isna().all():
                        valid_values = self.tracking_df[metric].dropna()
                        if len(valid_values) > 0:
                            f.write(f"平均{metric.upper()}: {valid_values.mean():.4f}\n")
            
            # 添加批次列表
            f.write("\n### 所有批次\n")
            for _, batch in self.tracking_df.iterrows():
                batch_str = f"批次 {batch['batch_id']}"
                if 'timestamp' in batch:
                    batch_str += f": {batch['timestamp']}"
                f.write(f"{batch_str}\n")
        
        return summary_path

def show_batch_results(element, analysis_type="feature_importance"):
    """显示元素的批次分析结果
    
    Args:
        element: 元素名称
        analysis_type: 分析类型 ("feature_importance" 或 "residual_analysis")
    """
    # 创建批处理跟踪器
    tracker = BatchTracker(element, analysis_type)
    
    # 获取所有批次结果
    batch_results = tracker.get_all_batch_results()
    
    if len(batch_results) == 0:
        logger.info(f"没有找到{element}的{analysis_type}批次结果")
        return
    
    # 显示批次结果
    logger.info(f"{element} {analysis_type} 批次分析结果:")
    logger.info(f"总批次数: {len(batch_results)}")
    
    if analysis_type == "feature_importance":
        # 找出最重要的特征所在批次
        best_batch = batch_results.loc[batch_results['top_feature_importance'].idxmax()]
        logger.info(f"最强特征重要性批次: {best_batch['batch_id']} (分数: {best_batch['top_feature_importance']:.4f})")
        logger.info(f"特征波长: {best_batch['top_feature_wavelength']:.2f}Å, 特征名称: {best_batch['top_feature_name']}")
        
        # 显示批次信息
        logger.info("\n批次详情:")
        for idx, batch in batch_results.iterrows():
            logger.info(f"批次 {batch['batch_id']}: 顶级特征波长={batch['top_feature_wavelength']:.2f}Å, " +
                       f"重要性={batch['top_feature_importance']:.4f}, 样本数={batch['num_samples']}")
    else:  # residual_analysis
        # 找出RMSE最小的批次
        best_rmse_batch = batch_results.loc[batch_results['rmse'].idxmin()]
        logger.info(f"最佳RMSE批次: {best_rmse_batch['batch_id']} (RMSE: {best_rmse_batch['rmse']:.4f})")
        
        # 找出R²最大的批次
        best_r2_batch = batch_results.loc[batch_results['r2'].idxmax()]
        logger.info(f"最佳R²批次: {best_r2_batch['batch_id']} (R²: {best_r2_batch['r2']:.4f})")
        
        # 显示批次信息
        logger.info("\n批次详情:")
        for idx, batch in batch_results.iterrows():
            logger.info(f"批次 {batch['batch_id']}: RMSE={batch['rmse']:.4f}, MAE={batch['mae']:.4f}, " +
                       f"R²={batch['r2']:.4f}, 样本数={batch['num_samples']}")
    
    # 显示结果文件位置
    batch_dirs = setup_batch_directories(element)
    logger.info(f"\n查看详细结果:")
    logger.info(f"1. 批次趋势图: {batch_dirs[analysis_type]}/{element}_batch_trends.png")
    logger.info(f"2. 批次摘要: {batch_dirs[analysis_type]}/{element}_batch_summary.txt")
    logger.info(f"3. 各批次结果: {batch_dirs[analysis_type]}/batch_*.*")

def analyze_gcn_attention(model, data_loader, device, element, 
                        batch_id=None, save_results=True, output_dir='results/gcn_analysis'):
    """
    分析GCN模型中的注意力权重和图结构
    
    Args:
        model: GCN模型 (SpectralResCNN_GCN)
        data_loader: 数据加载器
        device: 设备 (CPU/GPU)
        element: 元素名称
        batch_id: 批次ID (可选)
        save_results: 是否保存结果
        output_dir: 输出目录
    
    Returns:
        分析结果的摘要
    """
    # 检查模型类型
    if not hasattr(model, 'gcn_layer') or not hasattr(model, 'attention'):
        logger.warning(f"提供的模型不是GCN模型或没有注意力机制，无法进行GCN分析")
        return None
    
    logger.info(f"开始分析GCN模型的注意力权重和图结构...")
    
    # 准备保存分析结果的目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置为评估模式
    model.eval()
    
    # 用于存储注意力权重和邻接矩阵
    attention_weights = []
    adjacency_matrices = []
    
    with torch.no_grad():
        # 只分析一个批次的数据
        for batch_spectra, batch_labels in data_loader:
            batch_spectra = batch_spectra.to(device)
            batch_labels = batch_labels.to(device)
            
            # 将数据传递给模型
            # 我们需要在前向传递期间保存中间结果
            # 使用钩子注册器来捕获中间层的输出
            hook_handles = []
            gcn_output = None
            attention_output = None
            adjacency_matrix = None
            
            # 定义钩子函数
            def gcn_layer_hook(module, input, output):
                nonlocal gcn_output, adjacency_matrix
                gcn_output = output
                
                # 尝试获取邻接矩阵
                if hasattr(module, '_cached_adj_mat') and module._cached_adj_mat is not None:
                    adjacency_matrix = module._cached_adj_mat.cpu().numpy()
            
            def attention_hook(module, input, output):
                nonlocal attention_output
                attention_output = output
                
                # 如果注意力模块有注意力权重属性，捕获它
                if hasattr(module, 'attention_weights') and module.attention_weights is not None:
                    attention_weights.append(module.attention_weights.cpu().numpy())
            
            # 注册钩子
            if hasattr(model, 'gcn_layer'):
                hook_handles.append(model.gcn_layer.register_forward_hook(gcn_layer_hook))
            
            if hasattr(model, 'attention'):
                hook_handles.append(model.attention.register_forward_hook(attention_hook))
            
            # 运行前向传递
            _ = model(batch_spectra)
            
            # 移除钩子
            for handle in hook_handles:
                handle.remove()
            
            # 存储邻接矩阵
            if adjacency_matrix is not None:
                adjacency_matrices.append(adjacency_matrix)
            
            # 只分析第一个批次
            break
    
    # 分析结果
    results = {}
    
    # 分析注意力权重
    if attention_weights:
        attention_array = np.concatenate(attention_weights, axis=0)
        attention_mean = np.mean(attention_array, axis=0)
        attention_std = np.std(attention_array, axis=0)
        
        results['attention'] = {
            'mean': attention_mean,
            'std': attention_std,
            'max_attention_idx': np.argmax(attention_mean),
            'min_attention_idx': np.argmin(attention_mean)
        }
        
        # 绘制注意力热图
        plt.figure(figsize=(12, 6))
        
        # 创建一个发光的颜色映射
        colors = [(0, 0, 0.5), (0, 0.5, 1), (1, 1, 1), (1, 0.5, 0), (0.5, 0, 0)]
        custom_cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=256)
        
        sns.heatmap(attention_mean.reshape(1, -1), cmap=custom_cmap, 
                   annot=False, cbar=True, xticklabels=20, yticklabels=False)
        plt.xlabel('波长索引', fontsize=12)
        plt.title(f'{element}元素的GCN注意力分布', fontsize=14)
        plt.tight_layout()
        
        if batch_id is not None:
            attention_path = os.path.join(output_dir, f'{element}_attention_heatmap_batch{batch_id}.png')
        else:
            attention_path = os.path.join(output_dir, f'{element}_attention_heatmap.png')
        plt.savefig(attention_path, dpi=300)
        plt.close()
        
        # 绘制注意力分布线图
        plt.figure(figsize=(12, 6))
        seq_len = len(attention_mean)
        x = np.arange(seq_len)
        plt.plot(x, attention_mean, 'b-', linewidth=2, label='注意力权重')
        plt.fill_between(x, attention_mean - attention_std, attention_mean + attention_std, 
                         color='blue', alpha=0.2, label='标准差')
        plt.xlabel('波长索引', fontsize=12)
        plt.ylabel('注意力权重', fontsize=12)
        plt.title(f'{element}元素的GCN注意力分布', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if batch_id is not None:
            attention_dist_path = os.path.join(output_dir, f'{element}_attention_distribution_batch{batch_id}.png')
        else:
            attention_dist_path = os.path.join(output_dir, f'{element}_attention_distribution.png')
        plt.savefig(attention_dist_path, dpi=300)
        plt.close()
        
        # 标识前5个最高注意力的位置
        top_indices = np.argsort(attention_mean)[-5:][::-1]
        
        # 可视化其对应的波长（假设3500-10000埃的范围）
        input_size = seq_len  # 原始序列长度
        wavelengths = [3500 + (10000 - 3500) * idx / input_size for idx in top_indices]
        
        # 创建表格记录
        top_attention_table = pd.DataFrame({
            'index': top_indices,
            'estimated_wavelength': wavelengths,
            'attention_weight': attention_mean[top_indices]
        })
        
        # 保存表格
        if batch_id is not None:
            table_path = os.path.join(output_dir, f'{element}_top_attention_batch{batch_id}.csv')
        else:
            table_path = os.path.join(output_dir, f'{element}_top_attention.csv')
        top_attention_table.to_csv(table_path, index=False)
        
        # 为这些重要的波长添加物理解释
        top_attention_report = []
        for idx, row in top_attention_table.iterrows():
            wavelength = row['estimated_wavelength']
            attention_weight = row['attention_weight']
            
            # 获取特征解释
            feature = identify_spectral_feature(wavelength, element)
            explanation = get_feature_explanation(wavelength, element)
            
            top_attention_report.append({
                'wavelength_idx': int(row['index']),
                'estimated_wavelength': wavelength,
                'attention_weight': attention_weight,
                'feature': feature,
                'explanation': explanation
            })
        
        # 保存重要特征解释报告
        report_df = pd.DataFrame(top_attention_report)
        if batch_id is not None:
            report_path = os.path.join(output_dir, f'{element}_attention_explanation_batch{batch_id}.csv')
        else:
            report_path = os.path.join(output_dir, f'{element}_attention_explanation.csv')
        report_df.to_csv(report_path, index=False)
    
    # 分析邻接矩阵
    if adjacency_matrices:
        adj_matrix = adjacency_matrices[0]  # 使用第一个批次的邻接矩阵
        
        # 计算邻接矩阵统计信息
        results['adjacency'] = {
            'density': np.mean(adj_matrix > 0),  # 非零元素比例
            'mean_weight': np.mean(adj_matrix),
            'max_weight': np.max(adj_matrix),
            'sparsity': 1.0 - np.count_nonzero(adj_matrix) / adj_matrix.size  # 稀疏度
        }
        
        # 绘制邻接矩阵热图
        plt.figure(figsize=(10, 8))
        
        # 限制矩阵大小以便可视化
        max_size = 100
        if adj_matrix.shape[0] > max_size:
            # 采样矩阵的一部分
            indices = np.linspace(0, adj_matrix.shape[0]-1, max_size, dtype=int)
            adj_matrix_viz = adj_matrix[indices][:, indices]
        else:
            adj_matrix_viz = adj_matrix
        
        sns.heatmap(adj_matrix_viz, cmap='viridis', 
                   annot=False, cbar=True, xticklabels=20, yticklabels=20)
        plt.xlabel('节点索引', fontsize=12)
        plt.ylabel('节点索引', fontsize=12)
        plt.title(f'{element}元素的GCN邻接矩阵 (稀疏度: {results["adjacency"]["sparsity"]:.2f})', fontsize=14)
        plt.tight_layout()
        
        if batch_id is not None:
            adjacency_path = os.path.join(output_dir, f'{element}_adjacency_matrix_batch{batch_id}.png')
        else:
            adjacency_path = os.path.join(output_dir, f'{element}_adjacency_matrix.png')
        plt.savefig(adjacency_path, dpi=300)
        plt.close()
        
        # 绘制邻接矩阵中权重分布直方图
        plt.figure(figsize=(10, 6))
        plt.hist(adj_matrix.flatten(), bins=50, alpha=0.7, color='purple')
        plt.xlabel('连接权重', fontsize=12)
        plt.ylabel('频率', fontsize=12)
        plt.title(f'{element}元素的GCN连接权重分布', fontsize=14)
        plt.axvline(results['adjacency']['mean_weight'], color='r', linestyle='--', 
                    label=f'平均权重: {results["adjacency"]["mean_weight"]:.4f}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if batch_id is not None:
            weights_path = os.path.join(output_dir, f'{element}_adjacency_weights_batch{batch_id}.png')
        else:
            weights_path = os.path.join(output_dir, f'{element}_adjacency_weights.png')
        plt.savefig(weights_path, dpi=300)
        plt.close()
    
    # 保存分析结果摘要
    if save_results:
        if batch_id is not None:
            summary_path = os.path.join(output_dir, f'{element}_gcn_analysis_summary_batch{batch_id}.txt')
        else:
            summary_path = os.path.join(output_dir, f'{element}_gcn_analysis_summary.txt')
            
        with open(summary_path, 'w') as f:
            f.write(f"GCN模型分析结果摘要 - {element}元素\n")
            f.write("=" * 50 + "\n\n")
            
            if 'attention' in results:
                f.write("注意力分析:\n")
                f.write(f"- 最高注意力索引: {results['attention']['max_attention_idx']}\n")
                f.write(f"- 最低注意力索引: {results['attention']['min_attention_idx']}\n")
                f.write(f"- 注意力均值: {np.mean(results['attention']['mean']):.6f}\n")
                f.write(f"- 注意力标准差: {np.mean(results['attention']['std']):.6f}\n\n")
                
                # 添加前5个重要波长的信息
                if 'top_attention_table' in locals():
                    f.write("前5个重要波长:\n")
                    for idx, row in top_attention_table.iterrows():
                        f.write(f"  {idx+1}. 索引 {int(row['index'])}: 预估波长 {row['estimated_wavelength']:.2f}Å, ")
                        f.write(f"注意力权重 {row['attention_weight']:.6f}\n")
                    f.write("\n")
            
            if 'adjacency' in results:
                f.write("邻接矩阵分析:\n")
                f.write(f"- 连接密度: {results['adjacency']['density']:.6f}\n")
                f.write(f"- 平均连接权重: {results['adjacency']['mean_weight']:.6f}\n")
                f.write(f"- 最大连接权重: {results['adjacency']['max_weight']:.6f}\n")
                f.write(f"- 邻接矩阵稀疏度: {results['adjacency']['sparsity']:.6f}\n")
    
    logger.info(f"GCN模型分析完成，结果保存至 {output_dir}")
    
    return results 