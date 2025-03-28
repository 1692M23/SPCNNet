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
    if not hasattr(model, 'fc_layers'):
        logger.warning(f"Model does not have fc_layers attribute, using alternative method for feature importance")
    
    # 准备保存分析结果的目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储每个特征的重要性分数
    importances = []
    
    # 原始参考预测
    reference_predictions = []
    
    # 遍历数据加载器中的数据
    model.eval()
    
    try:
        # 收集一些样本用于特征重要性分析
        samples = []
        labels = []
        
        with torch.no_grad():
            for data, target in data_loader:
                # 只收集少量样本以加快分析速度
                if len(samples) < 50:  # 使用50个样本进行分析
                    # 检查并处理数据中的NaN
                    data_np = data.cpu().numpy()
                    if np.isnan(data_np).any():
                        # 用均值填充NaN
                        feature_means = np.nanmean(data_np, axis=0)
                        nan_mask = np.isnan(data_np)
                        for i in range(data_np.shape[0]):
                            sample_nan_mask = nan_mask[i]
                            if sample_nan_mask.any():
                                data_np[i][sample_nan_mask] = feature_means[sample_nan_mask]
                        data = torch.tensor(data_np, dtype=torch.float32)
                    
                    samples.append(data)
                    labels.append(target)
                else:
                    break
        
        if not samples:
            logger.error("No samples collected for feature importance analysis")
            return None
        
        # 合并样本
        X = torch.cat(samples, dim=0).to(device)
        y_true = torch.cat(labels, dim=0).to(device)
        
        # 获取输入特征数量
        n_features = X.shape[1]
        
        # 获取参考预测
        with torch.no_grad():
            reference_output = model(X).cpu().numpy()
        
        # 对每个特征执行排列重要性
        feature_importances = []
        
        for feature_idx in range(n_features):
            try:
                # 创建特征排列
                X_permuted = X.clone()
                permutation = torch.randperm(X.shape[0])
                X_permuted[:, feature_idx] = X[permutation, feature_idx]
                
                # 获取排列后的预测
                with torch.no_grad():
                    permuted_output = model(X_permuted).cpu().numpy()
                
                # 计算预测变化
                mse_increase = np.mean((permuted_output - reference_output) ** 2)
                feature_importances.append(mse_increase)
            except Exception as e:
                logger.warning(f"Error analyzing feature {feature_idx}: {str(e)}")
                feature_importances.append(0.0)
        
        # 标准化特征重要性
        if max(feature_importances) > 0:
            feature_importances = [f / max(feature_importances) for f in feature_importances]
        
        # 找出最重要的特征
        top_n = 10
        importances_with_idx = sorted([(i, importance) for i, importance in enumerate(feature_importances)], 
                                      key=lambda x: x[1], reverse=True)
        top_features = importances_with_idx[:top_n]
        
        # 找出最重要特征的索引和重要性
        top_importance = top_features[0][1] if top_features else 0
        top_index = top_features[0][0] if top_features else -1
        
        # 创建图表
        if save_results:
            batch_str = f"_batch_{batch_id}" if batch_id is not None else ""
            
            plt.figure(figsize=(12, 8))
            plt.bar(range(len(feature_importances)), feature_importances, alpha=0.7)
            plt.xlabel('Feature Index')
            plt.ylabel('Importance')
            plt.title(f'Feature Importance for {element}')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{element}{batch_str}_feature_importance.png'))
            plt.close()
            
            # 突出显示最重要的特征
            plt.figure(figsize=(12, 8))
            plt.bar(range(len(feature_importances)), feature_importances, alpha=0.3, color='gray')
            plt.bar([x[0] for x in top_features], [x[1] for x in top_features], alpha=0.7, color='red')
            plt.xlabel('Feature Index')
            plt.ylabel('Importance')
            plt.title(f'Top {top_n} Most Important Features')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{element}{batch_str}_top_features.png'))
            plt.close()
            
            # 保存特征重要性报告
            report_path = os.path.join(output_dir, f'{element}{batch_str}_feature_importance.txt')
            with open(report_path, 'w') as f:
                f.write(f"Feature Importance Analysis for {element}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total features analyzed: {len(feature_importances)}\n\n")
                
                f.write("Top 10 Important Features:\n")
                for i, (idx, importance) in enumerate(top_features):
                    f.write(f"{i+1}. Feature {idx}: {importance:.6f}\n")
                
                f.write("\nFeature Importance Statistics:\n")
                f.write(f"Mean importance: {np.mean(feature_importances):.6f}\n")
                f.write(f"Median importance: {np.median(feature_importances):.6f}\n")
                f.write(f"Max importance: {np.max(feature_importances):.6f}\n")
                f.write(f"Min importance: {np.min(feature_importances):.6f}\n")
        
        return {
            'importances': feature_importances,
            'top_features': top_features,
            'top_importance': top_importance,
            'top_index': top_index
        }
    
    except Exception as e:
        logger.error(f"Error in feature importance analysis: {str(e)}")
        return None

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
    
    logger.info(f"Starting feature importance analysis for {element} model...")
    importance_results = analyze_feature_importance(model, val_loader, device, element,
                                                  batch_id=batch_id,
                                                  save_results=save_batch_results)
    
    logger.info(f"Starting residual analysis for {element} model...")
    residual_analysis = analyze_residuals(model, test_loader, device, element,
                                          batch_id=batch_id,
                                          save_results=save_batch_results)
    
    logger.info(f"{element} model analysis completed, results saved in {output_dir} directory")
    
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
    if importance_results:
        combined_metrics.update({
            'top_feature_importance': importance_results.get('top_importance', float('nan')),
            'top_feature_index': importance_results.get('top_index', -1)
        })
    
    # 生成综合报告
    summary_path = os.path.join(output_dir, f"{element}_performance_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Performance Analysis Summary for {element}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Metrics:\n")
        for metric, value in combined_metrics.items():
            if isinstance(value, float):
                f.write(f"{metric}: {value:.6f}\n")
            else:
                f.write(f"{metric}: {value}\n")
        
        f.write("\nAnalysis Files:\n")
        if residual_analysis:
            f.write("Residual Analysis:\n")
            f.write(f"- RMSE: {residual_analysis.get('rmse', 'N/A'):.6f}\n")
            f.write(f"- MAE: {residual_analysis.get('mae', 'N/A'):.6f}\n")
            f.write(f"- R²: {residual_analysis.get('r2', 'N/A'):.6f}\n")
        
        if importance_results:
            f.write("\nFeature Importance Analysis:\n")
            if 'top_features' in importance_results:
                top_features = importance_results['top_features']
                f.write("Top Features:\n")
                for idx, imp in top_features:
                    f.write(f"- Feature {idx}: {imp:.6f}\n")
    
    results = {}
    results['summary'] = summary_path
    results['metrics'] = combined_metrics
    
    # 输出结果摘要
    logger.info(f"Performance analysis for {element} completed")
    if residual_analysis:
        logger.info(f"RMSE for {element}: {residual_analysis.get('rmse', 'N/A'):.4f}")
        logger.info(f"R² for {element}: {residual_analysis.get('r2', 'N/A'):.4f}")
    
    logger.info(f"{element} model analysis completed, results saved in {output_dir} directory")
    
    return results

# 批次结果目录结构
def setup_batch_directories(element):
    """创建批次结果目录结构"""
    batch_dirs = {
        'feature_importance': os.path.join("results", "feature_importance", f"{element}_batch_results"),
        'residual_analysis': os.path.join("results", "residual_analysis", f"{element}_batch_results"),
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
            analysis_type: 分析类型 ("feature_importance" 或 "residual_analysis")
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
            else:  # residual_analysis
                self.tracking_df = pd.DataFrame(columns=[
                    'batch_id', 'timestamp', 'num_samples', 'rmse', 'mae', 'r2', 
                    'residual_mean', 'residual_std', 'processing_time'
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
            
        else:  # residual_analysis
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