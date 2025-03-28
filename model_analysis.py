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

def analyze_feature_importance(model, data_loader, device, element, input_size=4096, top_n=30, 
                             batch_size=None, batch_id=None, save_batch_results=True):
    """
    分析特征重要性，识别对预测结果最重要的光谱波长区域
    兼容preprocessdata7.py的数据格式
    
    参数:
        model: 训练好的模型
        data_loader: 数据加载器或数据集
        device: 计算设备
        element: 元素名称
        input_size: 输入大小
        top_n: 要显示的顶部特征数量
        batch_size: 批量大小，用于批处理
        batch_id: 批次ID，用于批处理
        save_batch_results: 是否保存批处理结果
        
    返回:
        dict: 特征重要性结果
    """
    # 记录开始时间，用于计算处理时间
    start_time = time.time()
    
    # 如果输入是数据加载器，直接使用；否则创建数据加载器
    if not isinstance(data_loader, DataLoader):
        if isinstance(data_loader, tuple) and len(data_loader) >= 2:
            # 假设是(X, y)形式的数据
            X, y = data_loader[0], data_loader[1]
            
            # 确保X形状正确
            if len(X.shape) == 2:
                # [n_samples, n_features] -> [n_samples, 1, n_features]
                X = np.expand_dims(X, 1)
            elif len(X.shape) == 3 and X.shape[1] != 1:
                # 确保通道维度在第二维
                X = np.transpose(X, (0, 2, 1)) if X.shape[2] == 1 else X
                
            # 创建数据集和加载器
            dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
            data_loader = DataLoader(dataset, batch_size=batch_size or 32, shuffle=False)
        else:
            raise ValueError("无法从提供的数据创建数据加载器")
    
    # 设置批处理目录和跟踪器
    if save_batch_results:
        # 确保目录存在
        batch_dir = os.path.join('results', 'feature_importance', f'{element}_batch_results')
        os.makedirs(batch_dir, exist_ok=True)
    
        # 创建批次跟踪器
        batch_tracker = BatchTracker(element, "feature_importance")
    
        # 如果没有指定批次ID，获取下一个可用ID
        if batch_id is None:
            batch_id = batch_tracker.get_next_batch_id()
    
    # 计算特征重要性
    model.eval()
    all_importances = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            # 如果我们只处理特定批次且当前不是目标批次，跳过
            if batch_size is not None and batch_id is not None and batch_idx != batch_id:
                continue
            
            # 获取一个批次的数据
            batch_data = data.detach().cpu().numpy()
            batch_target = target.detach().cpu().numpy()
            
            # 对每个样本计算特征重要性
            batch_importances = []
            
            for i in range(min(len(batch_data), 50)):  # 限制计算样本数以提高效率
                sample = batch_data[i]
                sample_tensor = torch.FloatTensor(sample).unsqueeze(0).to(device)
                
                # 计算基线预测
                baseline_output = model(sample_tensor)
                baseline_value = baseline_output.item()
                
                # 计算每个波长区域的重要性
                importances = np.zeros(sample.shape[-1])
                
                # 使用滑动窗口遮挡不同的区域
                window_size = 50  # 窗口大小
                stride = 25       # 滑动步长
                
                for start in range(0, sample.shape[-1] - window_size + 1, stride):
                    end = start + window_size
                    # 创建遮挡后的样本
                    masked_sample = sample.copy()
                    masked_sample[0, start:end] = 0  # 遮挡该区域
                    
                    masked_tensor = torch.FloatTensor(masked_sample).unsqueeze(0).to(device)
                    masked_output = model(masked_tensor)
                    masked_value = masked_output.item()
                    
                    # 计算区域重要性（预测变化的绝对值）
                    importance = abs(baseline_value - masked_value)
                    
                    # 将重要性分配给该区域的所有波长点
                    importances[start:end] += importance
                
                batch_importances.append(importances)
            
            # 平均该批次的特征重要性
            if batch_importances:
                mean_importances = np.mean(batch_importances, axis=0)
                all_importances.append(mean_importances)
                
                # 如果只处理特定批次，或者需要保存批处理结果，生成该批次的结果
                if (batch_size is not None and batch_id is not None) or save_batch_results:
                    # 获取最重要的波长区域
                    top_indices = np.argsort(mean_importances)[-top_n:][::-1]
                    
                    # 估计波长范围
                    try:
                        # 尝试导入preprocessdata7
                        import importlib
                        pp7 = importlib.import_module('preprocessdata7')
                        if hasattr(pp7, 'LAMOSTPreprocessor'):
                            preprocessor = pp7.LAMOSTPreprocessor()
                            if hasattr(preprocessor, 'get_wavelength_range'):
                                wavelength_min, wavelength_max = preprocessor.get_wavelength_range()
                            else:
                                # 使用估计值
                                wavelength_min, wavelength_max = 3800, 9000
                        else:
                            wavelength_min, wavelength_max = 3800, 9000
                    except:
                        # 默认LAMOST波长范围估计值
                        wavelength_min, wavelength_max = 3800, 9000
                    
                    # 估计每个索引对应的波长
                    wavelengths = np.linspace(wavelength_min, wavelength_max, input_size)
                    
                    # 绘制特征重要性图
                    plt.figure(figsize=(12, 6))
                    plt.subplot(2, 1, 1)
                    plt.plot(mean_importances, alpha=0.7)
                    plt.title(f'元素 {element} 特征重要性分析（批次 {batch_id or batch_idx+1}）')
                    plt.xlabel('波长索引')
                    plt.ylabel('重要性')
                    plt.grid(True, alpha=0.3)
                    
                    plt.subplot(2, 1, 2)
                    plt.plot(mean_importances, alpha=0.3, color='gray')
                    plt.bar(top_indices, mean_importances[top_indices], alpha=0.7, color='red')
                    plt.title(f'顶部 {top_n} 特征重要性')
                    plt.xlabel('波长索引')
                    plt.ylabel('重要性')
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    
                    # 保存图表
                    plot_path = os.path.join(batch_dir, f'batch_{batch_id or batch_idx+1}_feature_importance.png')
                    plt.savefig(plot_path, dpi=300)
                    plt.close()
                    
                    # 保存解释文件
                    explanation_path = os.path.join(batch_dir, f'batch_{batch_id or batch_idx+1}_explanation.txt')
                    with open(explanation_path, 'w') as f:
                        f.write(f"元素 {element} 批次 {batch_id or batch_idx+1} 的特征重要性分析\n")
                        f.write("=" * 50 + "\n\n")
                        f.write("最重要的波长区域及其可能对应的光谱特征：\n\n")
                        
                        for i, idx in enumerate(top_indices):
                            wavelength = wavelengths[idx]
                            feature_explanation = get_feature_explanation(wavelength, element)
                            f.write(f"{i+1}. 波长索引 {idx} (估计波长: {wavelength:.2f} Å):\n")
                            f.write(f"   重要性得分: {mean_importances[idx]:.6f}\n")
                            f.write(f"   可能特征: {feature_explanation}\n\n")
                    
                    # 如果启用批处理结果保存，记录该批次的结果
                    if save_batch_results:
                        # 创建批次结果记录
                        max_importance = np.max(mean_importances)
                        # 检查max_importance是否为NaN
                        if np.isnan(max_importance):
                            logger.warning(f"批次 {batch_id or batch_idx+1} 的特征重要性为NaN，可能需要检查数据或模型")
                            max_importance = 0.0  # 使用默认值代替NaN
                            
                        batch_result = {
                            'batch_id': batch_id or batch_idx+1,
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'num_samples': len(batch_data),
                            'top_feature_wavelength': float(wavelengths[top_indices[0]]) if len(top_indices) > 0 else 0.0,
                            'top_feature_importance': float(max_importance),
                            'top_feature_name': identify_spectral_feature(wavelengths[top_indices[0]], element) if len(top_indices) > 0 else "未知",
                            'top_wavelengths': wavelengths[top_indices].tolist(),
                            'top_importance': mean_importances[top_indices].tolist(),
                            'processing_time': time.time() - start_time
                        }
                        
                        # 添加到批次跟踪器
                        batch_tracker.add_batch_result(batch_result)
                        
                        try:
                            # 生成趋势图
                            if len(batch_tracker.tracking_df) > 1:
                                batch_tracker.generate_trend_plots()
                                
                            # 生成批次摘要
                            batch_tracker.generate_batch_summary()
                        except Exception as e:
                            logger.error(f"生成批次趋势图或摘要时出错: {str(e)}")
                    
                    # 如果只处理特定批次，完成后返回
                    if batch_size is not None and batch_id is not None:
                        return {
                            'batch_id': batch_id,
                            'feature_importance': mean_importances,
                            'top_indices': top_indices.tolist(),
                            'top_wavelengths': wavelengths[top_indices].tolist(),
                            'top_importance': mean_importances[top_indices].tolist(),
                            'max_importance': float(max_importance),
                            'plot_path': plot_path,
                            'explanation_path': explanation_path
                        }
    
    # 如果我们处理了所有批次，计算总体特征重要性
    if all_importances:
        mean_importances = np.mean(all_importances, axis=0)
        top_indices = np.argsort(mean_importances)[-top_n:][::-1]
        
        # 绘制总体特征重要性图
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(mean_importances, alpha=0.7)
        plt.title(f'元素 {element} 特征重要性分析（总体）')
        plt.xlabel('波长索引')
        plt.ylabel('重要性')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(mean_importances, alpha=0.3, color='gray')
        plt.bar(top_indices, mean_importances[top_indices], alpha=0.7, color='red')
        plt.title(f'顶部 {top_n} 特征重要性')
        plt.xlabel('波长索引')
        plt.ylabel('重要性')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_dir = os.path.join('results', 'feature_importance')
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f'{element}_feature_importance.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        # 估计波长范围
        try:
            # 尝试导入preprocessdata7
            import importlib
            pp7 = importlib.import_module('preprocessdata7')
            if hasattr(pp7, 'LAMOSTPreprocessor'):
                preprocessor = pp7.LAMOSTPreprocessor()
                if hasattr(preprocessor, 'get_wavelength_range'):
                    wavelength_min, wavelength_max = preprocessor.get_wavelength_range()
                else:
                    # 使用估计值
                    wavelength_min, wavelength_max = 3800, 9000
            else:
                wavelength_min, wavelength_max = 3800, 9000
        except:
            # 默认LAMOST波长范围估计值
            wavelength_min, wavelength_max = 3800, 9000
        
        # 估计每个索引对应的波长
        wavelengths = np.linspace(wavelength_min, wavelength_max, input_size)
        
        # 保存总体解释文件
        explanation_path = os.path.join(plot_dir, f'{element}_feature_explanation.txt')
        with open(explanation_path, 'w') as f:
            f.write(f"元素 {element} 特征重要性分析\n")
            f.write("=" * 50 + "\n\n")
            f.write("最重要的波长区域及其可能对应的光谱特征：\n\n")
            
            for i, idx in enumerate(top_indices):
                wavelength = wavelengths[idx]
                feature_explanation = get_feature_explanation(wavelength, element)
                f.write(f"{i+1}. 波长索引 {idx} (估计波长: {wavelength:.2f} Å):\n")
                f.write(f"   重要性得分: {mean_importances[idx]:.6f}\n")
                f.write(f"   可能特征: {feature_explanation}\n\n")
        
        # 检查并处理NaN值
        max_importance = np.max(mean_importances)
        if np.isnan(max_importance):
            logger.warning(f"元素 {element} 的总体特征重要性为NaN，返回零值")
            max_importance = 0.0
        
        return {
            'feature_importance': mean_importances,
            'top_indices': top_indices.tolist(),
            'top_wavelengths': wavelengths[top_indices].tolist(),
            'top_importance': mean_importances[top_indices].tolist(),
            'max_importance': float(max_importance),
            'plot_path': plot_path,
            'explanation_path': explanation_path
        }
    else:
        logger.warning(f"未收集到任何有效的特征重要性数据")
        return None

def analyze_residuals(model, test_loader, device, element, batch_size=None, batch_id=None, save_batch_results=True):
    """分析模型预测残差，评估模型在不同元素丰度区间的表现
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        element: 元素名称
        batch_size: 批处理大小，默认为None（使用全部数据）
        batch_id: 批次ID，默认为None（自动分配）
        save_batch_results: 是否保存批处理结果
        
    Returns:
        包含结果文件路径的字典
    """
    # 获取配置
    if batch_size is None and hasattr(analysis_config, 'batch_size'):
        batch_size = analysis_config.get('batch_size', 32)
    
    # 创建批次跟踪器
    if save_batch_results:
        batch_tracker = BatchTracker(element, "residual_analysis")
        if batch_id is None:
            batch_id = batch_tracker.get_next_batch_id()
    
    # 创建结果目录
    results_dir = os.path.join("results", "residual_analysis")
    batch_results_dir = os.path.join(results_dir, f"{element}_batch_results")
    
    if save_batch_results:
        os.makedirs(batch_results_dir, exist_ok=True)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # 开始计时
    start_time = time.time()
    
    # 收集真实值和预测值
    model.eval()
    all_targets = []
    all_predictions = []
    
    logger.info(f"开始分析{element}模型残差 (批次 {batch_id})...")
    
    # 计算处理的样本数量
    num_samples_processed = 0
    
    with torch.no_grad():
        for i, (spectra, targets) in enumerate(test_loader):
            # 如果设置了批处理大小，只处理指定数量的批次
            if batch_size is not None and i >= batch_size:
                break
                
            spectra = spectra.to(device)
            targets = targets.numpy()
            
            predictions = model(spectra).cpu().numpy()
            
            all_targets.append(targets)
            all_predictions.append(predictions)
            
            num_samples_processed += len(spectra)
    
    # 计算处理时间
    processing_time = time.time() - start_time
    
    if not all_targets or not all_predictions:
        logger.error(f"未收集到任何数据，无法进行残差分析")
        return None
    
    # 合并数据
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    
    # 计算残差
    residuals = y_true - y_pred
    
    # 计算评估指标
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    
    # 创建残差散点图
    plt.figure(figsize=(14, 10))
    
    # 添加批次信息到标题
    batch_info = f" (批次 {batch_id})" if batch_id is not None else ""
    
    # 真实值vs预测值散点图
    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(y_true, y_pred, alpha=0.5)
    
    # 添加对角线
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax1.set_title(f'{element}元素真实值vs预测值{batch_info}')
    ax1.set_xlabel('真实值')
    ax1.set_ylabel('预测值')
    
    # 给图添加R²和RMSE说明
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}', 
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # 残差散点图
    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(y_true, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    
    ax2.set_title(f'{element}元素残差分布{batch_info}')
    ax2.set_xlabel('真实值')
    ax2.set_ylabel('残差 (真实值 - 预测值)')
    
    # 残差分布直方图
    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=0, color='r', linestyle='--')
    
    ax3.set_title(f'{element}元素残差直方图{batch_info}')
    ax3.set_xlabel('残差')
    ax3.set_ylabel('频数')
    
    # 添加正态分布拟合曲线
    from scipy.stats import norm
    x = np.linspace(min(residuals), max(residuals), 100)
    y = norm.pdf(x, residual_mean, residual_std) * len(residuals) * (max(residuals) - min(residuals)) / 30
    ax3.plot(x, y, 'r-', linewidth=2)
    
    # 给图添加残差统计信息
    ax3.text(0.05, 0.95, f'均值 = {residual_mean:.4f}\n标准差 = {residual_std:.4f}', 
             transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # QQ图（用于检查残差的正态性）
    ax4 = plt.subplot(2, 2, 4)
    from scipy.stats import probplot
    probplot(residuals, plot=ax4)
    
    ax4.set_title(f'{element}元素残差Q-Q图{batch_info}')
    
    plt.tight_layout()
    
    # 保存图像
    if save_batch_results and batch_id is not None:
        # 保存到批次目录
        output_path = os.path.join(batch_results_dir, f"batch_{batch_id}_residual_analysis.png")
    else:
        # 保存到主目录
        output_path = os.path.join(results_dir, f"{element}_residual_analysis.png")
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    # 保存预测结果
    if save_batch_results and batch_id is not None:
        predictions_file = os.path.join(batch_results_dir, f"batch_{batch_id}_predictions.csv")
    else:
        predictions_file = os.path.join(results_dir, f"{element}_predictions.csv")
    
    predictions_df = pd.DataFrame({
        'true_value': y_true.flatten(),
        'predicted_value': y_pred.flatten(),
        'residual': residuals.flatten()
    })
    
    predictions_df.to_csv(predictions_file, index=False)
    
    # 生成残差分析报告
    if save_batch_results and batch_id is not None:
        report_path = os.path.join(batch_results_dir, f"batch_{batch_id}_residual_report.txt")
    else:
        report_path = os.path.join(results_dir, f"{element}_residual_report.txt")
    
    with open(report_path, 'w') as f:
        # 添加批次信息
        f.write(f"## {element}元素丰度预测残差分析报告")
        if batch_id is not None:
            f.write(f" (批次 {batch_id})\n\n")
        else:
            f.write("\n\n")
            
        f.write("### 性能指标\n")
        f.write(f"RMSE (均方根误差): {rmse:.6f}\n")
        f.write(f"MAE (平均绝对误差): {mae:.6f}\n")
        f.write(f"R² (决定系数): {r2:.6f}\n\n")
        
        f.write("### 残差统计\n")
        f.write(f"样本数: {len(residuals)}\n")
        f.write(f"残差均值: {residual_mean:.6f}\n")
        f.write(f"残差标准差: {residual_std:.6f}\n")
        f.write(f"残差最小值: {np.min(residuals):.6f}\n")
        f.write(f"残差最大值: {np.max(residuals):.6f}\n")
        f.write(f"残差中位数: {np.median(residuals):.6f}\n\n")
        
        # 检查残差是否接近正态分布
        from scipy.stats import shapiro
        try:
            # 对大样本，随机抽取1000个样本进行检验
            test_sample = residuals if len(residuals) < 1000 else np.random.choice(residuals, 1000, replace=False)
            stat, p = shapiro(test_sample)
            f.write("### 残差正态性检验 (Shapiro-Wilk测试)\n")
            f.write(f"统计量: {stat:.6f}\n")
            f.write(f"p值: {p:.6f}\n")
            f.write(f"结论: {'残差可能服从正态分布' if p > 0.05 else '残差可能不服从正态分布'}\n\n")
        except Exception as e:
            f.write(f"无法进行正态性检验: {str(e)}\n\n")
        
        # 添加批次处理信息
        f.write("### 批次处理信息\n")
        f.write(f"处理时间: {processing_time:.2f}秒\n")
        f.write(f"处理样本数: {num_samples_processed}\n")
        f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 如果启用了批处理结果跟踪，则更新跟踪数据
    if save_batch_results:
        # 准备批次结果数据
        batch_result = {
            'batch_id': batch_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_samples': num_samples_processed,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'processing_time': processing_time
        }
        
        # 更新批次跟踪
        batch_tracker.add_batch_result(batch_result)
        
        # 更新趋势图
        if len(batch_tracker.tracking_df) > 1:
            batch_tracker.generate_trend_plots()
            
        # 更新批次摘要
        batch_tracker.generate_batch_summary()
    
    results = {
        'plot': output_path,
        'report': report_path,
        'predictions': predictions_file,
        'metrics': {
            'rmse': rmse, 
            'mae': mae, 
            'r2': r2,
            'residual_mean': residual_mean,
            'residual_std': residual_std
        }
    }
    
    logger.info(f"{element}模型残差分析完成 (批次 {batch_id})，结果保存在{output_path}")
    return results

def analyze_model_performance(model, element, train_loader, val_loader, test_loader, device, 
                           input_size=4096, batch_size=None, save_batch_results=True):
    """对训练好的模型进行全面性能分析
    
    Args:
        model: 模型
        element: 元素名称
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        device: 设备
        input_size: 输入大小
        batch_size: 批处理大小
        save_batch_results: 是否保存批处理结果
        
    Returns:
        包含分析结果的字典
    """
    # 获取配置
    if batch_size is None and hasattr(analysis_config, 'batch_size'):
        batch_size = analysis_config.get('batch_size', 32)
    
    # 创建基本结果目录
    os.makedirs(os.path.join("results", "feature_importance"), exist_ok=True)
    os.makedirs(os.path.join("results", "residual_analysis"), exist_ok=True)
    
    # 如果启用了批处理结果，创建批处理结果目录
    if save_batch_results:
        batch_dirs = setup_batch_directories(element)
    
    results = {}
    batch_id = None
    
    if save_batch_results:
        # 创建批次跟踪器并获取下一个批次ID
        fi_tracker = BatchTracker(element, "feature_importance")
        ra_tracker = BatchTracker(element, "residual_analysis")
        batch_id = fi_tracker.get_next_batch_id()
        
        # 确保两个跟踪器使用相同的批次ID
        if ra_tracker.get_next_batch_id() != batch_id:
            logger.warning(f"特征重要性和残差分析的批次ID不同，使用特征重要性的批次ID: {batch_id}")
    
    # 分析特征重要性
    logger.info(f"开始分析{element}模型的特征重要性...")
    feature_importance_path = analyze_feature_importance(model, val_loader, device, element, 
                                                       input_size=input_size, 
                                                       batch_size=batch_size,
                                                       batch_id=batch_id,
                                                       save_batch_results=save_batch_results)
    results['feature_importance_path'] = feature_importance_path
    
    # 分析残差
    logger.info(f"开始分析{element}模型的残差...")
    residual_analysis = analyze_residuals(model, test_loader, device, element,
                                        batch_size=batch_size,
                                        batch_id=batch_id,
                                        save_batch_results=save_batch_results)
    results['residual_analysis'] = residual_analysis
    
    # 如果启用了批处理结果跟踪，添加批次信息到结果
    if save_batch_results:
        results['batch_id'] = batch_id
        results['feature_importance_tracker'] = fi_tracker
        results['residual_analysis_tracker'] = ra_tracker
        
        # 显示批次结果
        logger.info(f"批次 {batch_id} 分析完成")
        logger.info(f"生成的特征重要性图: {feature_importance_path}")
        if residual_analysis:
            logger.info(f"生成的残差分析图: {residual_analysis['plot']}")
            logger.info(f"元素 {element} 的RMSE: {residual_analysis['metrics']['rmse']:.4f}")
            logger.info(f"元素 {element} 的R²: {residual_analysis['metrics']['r2']:.4f}")
    
    logger.info(f"{element}模型分析完成，结果保存在results目录")
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
            plt.title(f'{self.element}特征重要性趋势')
            plt.xlabel('批次ID')
            plt.ylabel('顶级特征重要性分数')
            plt.grid(True, alpha=0.3)
            
            # 绘制处理时间趋势
            plt.subplot(2, 1, 2)
            plt.plot(self.tracking_df['batch_id'], self.tracking_df['processing_time'], 'o-')
            plt.title('处理时间趋势')
            plt.xlabel('批次ID')
            plt.ylabel('处理时间 (秒)')
            plt.grid(True, alpha=0.3)
            
        else:  # residual_analysis
            # 绘制RMSE和MAE趋势
            plt.subplot(2, 2, 1)
            plt.plot(self.tracking_df['batch_id'], self.tracking_df['rmse'], 'o-', label='RMSE')
            plt.plot(self.tracking_df['batch_id'], self.tracking_df['mae'], 's-', label='MAE')
            plt.title(f'{self.element}误差指标趋势')
            plt.xlabel('批次ID')
            plt.ylabel('误差值')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 绘制R²趋势
            plt.subplot(2, 2, 2)
            plt.plot(self.tracking_df['batch_id'], self.tracking_df['r2'], 'o-')
            plt.title('R²趋势')
            plt.xlabel('批次ID')
            plt.ylabel('R²')
            plt.grid(True, alpha=0.3)
            
            # 绘制残差均值趋势
            plt.subplot(2, 2, 3)
            plt.plot(self.tracking_df['batch_id'], self.tracking_df['residual_mean'], 'o-')
            plt.axhline(0, color='r', linestyle='--', alpha=0.3)
            plt.title('残差均值趋势')
            plt.xlabel('批次ID')
            plt.ylabel('残差均值')
            plt.grid(True, alpha=0.3)
            
            # 绘制残差标准差趋势
            plt.subplot(2, 2, 4)
            plt.plot(self.tracking_df['batch_id'], self.tracking_df['residual_std'], 'o-')
            plt.title('残差标准差趋势')
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