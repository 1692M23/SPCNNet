import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from scipy import stats
from scipy.signal import find_peaks
import pandas as pd
import time

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

def analyze_feature_importance(model, data_loader, device, element, input_size=4096, top_n=30):
    """分析光谱波长区域对预测的重要性并可视化解释"""
    # 创建结果目录
    results_dir = os.path.join("results", "feature_importance")
    os.makedirs(results_dir, exist_ok=True)
    
    # 收集特征梯度
    model.eval()
    wavelengths = np.linspace(3800, 9000, input_size)  # 根据实际波长范围调整
    feature_importance = np.zeros(input_size)
    
    logger.info(f"开始计算{element}的特征重要性...")
    
    try:
        for spectra, targets in data_loader:
            spectra = spectra.to(device).requires_grad_(True)
            outputs = model(spectra)
            
            # 计算每个特征的梯度
            for i in range(min(len(targets), 10)):  # 使用部分样本进行分析
                model.zero_grad()
                if i < len(outputs):
                    outputs[i].backward(retain_graph=(i < min(len(targets), 10)-1))
                    if spectra.grad is not None:
                        grad_data = spectra.grad[i, 0].cpu().numpy()
                        feature_importance += np.abs(grad_data)
    except Exception as e:
        logger.error(f"计算特征重要性时出错: {str(e)}")
        
        # 尝试替代方法 - 扰动分析
        logger.info("尝试使用扰动分析方法...")
        feature_importance = np.zeros(input_size)
        
        with torch.no_grad():
            for spectra, targets in data_loader:
                spectra = spectra.to(device)
                baseline = model(spectra).cpu().numpy()
                
                # 每次扰动一组特征
                for i in range(0, input_size, 20):
                    end_idx = min(i + 20, input_size)
                    perturbed = spectra.clone()
                    perturbed[:, 0, i:end_idx] *= 0.8  # 将该区域减少20%
                    perturbed_output = model(perturbed).cpu().numpy()
                    
                    # 计算输出变化
                    delta = np.abs(perturbed_output - baseline).mean(axis=0)
                    feature_importance[i:end_idx] = delta
                    
                break  # 只使用一个批次
    
    # 平滑重要性得分并归一化
    from scipy.ndimage import gaussian_filter1d
    smoothed_importance = gaussian_filter1d(feature_importance, sigma=5)
    normalized_importance = smoothed_importance / np.max(smoothed_importance)
    
    # 找出顶部重要区域
    try:
        peaks = find_peaks(normalized_importance, height=0.3, distance=20)[0]
        top_peaks = sorted([(p, normalized_importance[p]) for p in peaks], 
                          key=lambda x: x[1], reverse=True)[:top_n]
    except Exception as e:
        logger.error(f"查找峰值时出错: {str(e)}")
        # 如果找不到峰值，就直接选择最高的点
        indices = np.argsort(normalized_importance)[-top_n:]
        top_peaks = [(idx, normalized_importance[idx]) for idx in indices]
    
    # 创建可视化和解释
    plt.figure(figsize=(12, 10))
    
    # 上图：光谱与重要性
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(wavelengths, normalized_importance, 'r-', alpha=0.7, label='特征重要性')
    
    # 添加已知元素吸收线标记
    element_lines = get_element_absorption_lines(element)
    for line_name, line_wavelength in element_lines.items():
        closest_idx = np.argmin(np.abs(wavelengths - line_wavelength))
        if normalized_importance[closest_idx] > 0.2:
            ax1.axvline(line_wavelength, color='blue', linestyle='--', alpha=0.5)
            ax1.text(line_wavelength, 0.8, line_name, rotation=90, fontsize=8)
    
    ax1.set_title(f'{element}元素丰度预测的光谱特征重要性')
    ax1.set_xlabel('波长 (Å)')
    ax1.set_ylabel('重要性分数')
    ax1.legend()
    
    # 下图：Top-N重要区域详细信息表格
    ax2 = plt.subplot(2, 1, 2)
    ax2.axis('off')
    table_data = []
    header = ['波长区域 (Å)', '重要性分数', '可能对应的光谱特征']
    
    for peak_idx, importance in top_peaks:
        wave = wavelengths[peak_idx]
        feature_name = identify_spectral_feature(wave, element)
        table_data.append([f"{wave:.2f}", f"{importance:.4f}", feature_name])
    
    table = ax2.table(cellText=table_data, colLabels=header, 
                     loc='center', cellLoc='center', colWidths=[0.3, 0.2, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{element}_feature_importance.png"), dpi=300)
    plt.close()
    
    # 生成解释性报告
    with open(os.path.join(results_dir, f"{element}_explanation.txt"), 'w') as f:
        f.write(f"## {element}元素丰度预测的关键光谱特征解释\n\n")
        f.write("模型重点关注的波长区域及其天文物理意义：\n\n")
        
        for i, (peak_idx, importance) in enumerate(top_peaks[:10]):
            wave = wavelengths[peak_idx]
            feature_name = identify_spectral_feature(wave, element)
            explanation = get_feature_explanation(wave, element)
            
            f.write(f"### {i+1}. 波长 {wave:.2f}Å (重要性: {importance:.4f})\n")
            f.write(f"特征: {feature_name}\n")
            f.write(f"解释: {explanation}\n\n")
        
        f.write("注：特征重要性反映了模型对特定波长区域的依赖程度，但不一定表示因果关系。")
    
    logger.info(f"{element}特征重要性分析完成，结果保存在{results_dir}目录")
    return os.path.join(results_dir, f"{element}_feature_importance.png")

def analyze_residuals(model, test_loader, device, element):
    """生成详细的残差分析与可视化"""
    # 创建结果目录
    results_dir = os.path.join("results", "residual_analysis")
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info(f"开始分析{element}的残差...")
    model.eval()
    all_preds = []
    all_targets = []
    
    try:
        with torch.no_grad():
            for spectra, targets in test_loader:
                spectra = spectra.to(device)
                outputs = model(spectra)
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
    except Exception as e:
        logger.error(f"预测过程中出错: {str(e)}")
        return None
    
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    residuals = all_preds - all_targets
    
    # 计算统计信息
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    r2 = 1 - (np.sum(residuals**2) / np.sum((all_targets - all_targets.mean())**2))
    
    # 创建详细的残差分析图
    plt.figure(figsize=(15, 12))
    
    # 1. 预测vs真实值散点图
    ax1 = plt.subplot(2, 2, 1)
    scatter = ax1.scatter(all_targets, all_preds, c=np.abs(residuals), 
                         cmap='viridis', alpha=0.7, s=20)
    min_val, max_val = min(all_targets.min(), all_preds.min()), max(all_targets.max(), all_preds.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax1.set_xlabel('真实值')
    ax1.set_ylabel('预测值')
    ax1.set_title(f'{element}预测vs真实值')
    plt.colorbar(scatter, ax=ax1, label='绝对误差')
    
    # 2. 残差vs真实值图
    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(all_targets, residuals, alpha=0.7)
    ax2.axhline(0, color='r', linestyle='--')
    
    # 添加趋势线
    z = np.polyfit(all_targets, residuals, 1)
    p = np.poly1d(z)
    ax2.plot(all_targets, p(all_targets), "r--", alpha=0.8)
    ax2.text(0.05, 0.95, f"趋势线: y={z[0]:.4f}x{z[1]:+.4f}", 
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top')
    
    ax2.set_xlabel('真实值')
    ax2.set_ylabel('残差')
    ax2.set_title('残差分析')
    
    # 3. 残差直方图
    ax3 = plt.subplot(2, 2, 3)
    hist_bins = min(50, len(residuals)//20 + 5)
    ax3.hist(residuals, bins=hist_bins, alpha=0.7, density=True)
    
    # 添加正态分布曲线比较
    from scipy import stats
    mu, std = stats.norm.fit(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    p = stats.norm.pdf(x, mu, std)
    ax3.plot(x, p, 'r--', linewidth=2)
    
    ax3.set_xlabel('残差值')
    ax3.set_ylabel('频率')
    ax3.set_title(f'残差分布 (μ={mu:.4f}, σ={std:.4f})')
    
    # 4. 添加QQ图检验正态性
    ax4 = plt.subplot(2, 2, 4)
    stats.probplot(residuals, plot=ax4)
    ax4.set_title('残差Q-Q图')
    
    # 添加总结统计信息
    plt.figtext(0.5, 0.01, 
               f"模型性能指标: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}\n"
               f"残差均值: {np.mean(residuals):.4f}, 残差标准差: {np.std(residuals):.4f}",
               ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_dir, f"{element}_residual_analysis.png"), dpi=300)
    plt.close()
    
    # 生成解释性报告
    with open(os.path.join(results_dir, f"{element}_residual_explanation.txt"), 'w') as f:
        f.write(f"## {element}元素丰度预测的残差分析报告\n\n")
        
        # 基本统计信息
        f.write("### 基本统计信息\n")
        f.write(f"- RMSE: {rmse:.4f}\n")
        f.write(f"- MAE: {mae:.4f}\n")
        f.write(f"- R²: {r2:.4f}\n")
        f.write(f"- 残差均值: {np.mean(residuals):.4f}\n")
        f.write(f"- 残差标准差: {np.std(residuals):.4f}\n\n")
        
        # 残差分布评估
        f.write("### 残差分布评估\n")
        shapiro_test = stats.shapiro(residuals[:1000] if len(residuals) > 1000 else residuals)  # 避免样本过大
        f.write(f"- Shapiro-Wilk正态性检验: p值={shapiro_test[1]:.6f} ")
        f.write("(残差分布接近正态分布)\n" if shapiro_test[1] > 0.05 else "(残差分布偏离正态分布)\n")
        
        # 系统性偏差评估
        trend_significance = abs(z[0]) / (std / np.sqrt(len(residuals)))
        f.write("\n### 系统性偏差评估\n")
        f.write(f"- 残差与真实值的线性关系: 斜率={z[0]:.6f}, 截距={z[1]:.6f}\n")
        
        if trend_significance > 2:
            f.write("- **存在明显的系统性偏差**: 模型在特定值范围内存在预测偏差\n")
            
            # 分析具体的偏差趋势
            if z[0] > 0:
                f.write("- 趋势表明模型对较高的元素丰度值预测偏高，对较低的值预测偏低\n")
            else:
                f.write("- 趋势表明模型对较高的元素丰度值预测偏低，对较低的值预测偏高\n")
        else:
            f.write("- 无明显系统性偏差: 模型在不同值范围内表现相对一致\n")
        
        # 模型预测行为的解释
        f.write("\n### 模型预测行为解释\n")
        if np.mean(residuals) > 0.05:
            f.write("- 模型总体预测值偏高\n")
        elif np.mean(residuals) < -0.05:
            f.write("- 模型总体预测值偏低\n")
        else:
            f.write("- 模型总体预测平衡，无明显偏高或偏低\n")
            
        # 对天文学应用的建议
        f.write("\n### 对天文学应用的建议\n")
        if r2 > 0.9:
            f.write("- 模型预测非常可靠，适合用于精确的天文参数测量\n")
        elif r2 > 0.7:
            f.write("- 模型预测较为可靠，适合用于天文参数估计，但需注意潜在误差\n")
        else:
            f.write("- 模型预测存在较大不确定性，建议仅用于初步筛选或与其他方法结合使用\n")
    
    logger.info(f"{element}残差分析完成，结果保存在{results_dir}目录")
    return os.path.join(results_dir, f"{element}_residual_analysis.png")

def analyze_model_performance(model, element, train_loader, val_loader, test_loader, device, input_size=4096):
    """对训练好的模型进行全面性能分析"""
    # 创建结果目录
    os.makedirs(os.path.join("results", "feature_importance"), exist_ok=True)
    os.makedirs(os.path.join("results", "residual_analysis"), exist_ok=True)
    
    results = {}
    
    # 分析特征重要性
    logger.info(f"开始分析{element}模型的特征重要性...")
    feature_importance_path = analyze_feature_importance(model, val_loader, device, element, input_size)
    results['feature_importance_path'] = feature_importance_path
    
    # 分析残差
    logger.info(f"开始分析{element}模型的残差...")
    residual_analysis_path = analyze_residuals(model, test_loader, device, element)
    results['residual_analysis_path'] = residual_analysis_path
    
    logger.info(f"{element}模型分析完成，结果保存在results目录")
    return results 