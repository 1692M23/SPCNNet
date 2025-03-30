import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import logging

# 配置日志
logger = logging.getLogger(__name__)

def plot_predictions_vs_true(targets, outputs, element, save_path=None, figsize=(10, 6)):
    """
    绘制预测值与真实值的对比散点图
    
    参数:
        targets (ndarray): 真实值数组
        outputs (ndarray): 预测值数组
        element (str): 元素名称
        save_path (str): 保存路径，默认为None
        figsize (tuple): 图表尺寸
    """
    # 确保输入是一维数组
    if len(targets.shape) > 1 and targets.shape[1] == 1:
        targets = targets.flatten()
    if len(outputs.shape) > 1 and outputs.shape[1] == 1:
        outputs = outputs.flatten()
    
    # 计算评估指标
    mae = np.mean(np.abs(outputs - targets))
    mse = np.mean((outputs - targets) ** 2)
    rmse = np.sqrt(mse)
    
    # 计算R²
    if np.var(targets) == 0:  # 防止除零错误
        r2 = 0
    else:
        r2 = 1 - (np.sum((targets - outputs) ** 2) / np.sum((targets - np.mean(targets)) ** 2))
    
    # 使用scipy计算皮尔逊相关系数
    if len(targets) > 1:
        r_value, p_value = stats.pearsonr(targets, outputs)
    else:
        r_value, p_value = 0, 1
    
    # 创建图表
    plt.figure(figsize=figsize)
    
    # 绘制散点图
    plt.scatter(targets, outputs, alpha=0.6, label=f'样本点 (n={len(targets)})')
    
    # 添加理想线 (y=x)
    min_val = min(np.min(targets), np.min(outputs))
    max_val = max(np.max(targets), np.max(outputs))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想线 (y=x)')
    
    # 添加拟合线
    if len(targets) > 1:
        slope, intercept = np.polyfit(targets, outputs, 1)
        fit_line = slope * np.array([min_val, max_val]) + intercept
        plt.plot([min_val, max_val], fit_line, 'g-', label=f'拟合线 (y={slope:.2f}x+{intercept:.2f})')
    
    # 设置标题和标签
    plt.title(f'{element} 元素丰度预测 vs 真实值')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.grid(True, alpha=0.3)
    
    # 添加评估指标文本
    metrics_text = (
        f"MAE: {mae:.4f}\n"
        f"RMSE: {rmse:.4f}\n"
        f"R²: {r2:.4f}\n"
        f"r: {r_value:.4f}\n"
        f"p-value: {p_value:.4f}\n"
        f"样本数: {len(targets)}"
    )
    plt.annotate(
        metrics_text,
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        plt.close()
        return None

def evaluate_model_simple(outputs, targets):
    """
    简单评估模型性能，计算各种评估指标
    
    参数:
        outputs (ndarray): 模型预测值
        targets (ndarray): 真实目标值
        
    返回:
        dict: 包含各种评估指标的字典
    """
    # 确保输入是一维数组
    if len(targets.shape) > 1 and targets.shape[1] == 1:
        targets = targets.flatten()
    if len(outputs.shape) > 1 and outputs.shape[1] == 1:
        outputs = outputs.flatten()
        
    # 计算各种评估指标
    mae = np.mean(np.abs(outputs - targets))
    mse = np.mean((outputs - targets) ** 2)
    rmse = np.sqrt(mse)
    
    # 计算R²
    if np.var(targets) == 0:
        r2 = 0
    else:
        r2 = 1 - (np.sum((targets - outputs) ** 2) / np.sum((targets - np.mean(targets)) ** 2))
    
    # 计算Pearson相关系数
    if len(targets) > 1:
        r_value, p_value = stats.pearsonr(targets, outputs)
    else:
        r_value, p_value = 0, 1
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'r_value': r_value,
        'p_value': p_value,
        'num_samples': len(targets)
    }

def plot_error_distribution(outputs, targets, element, save_path=None, figsize=(10, 6)):
    """
    绘制预测误差分布直方图
    
    参数:
        outputs (ndarray): 模型预测值
        targets (ndarray): 真实目标值
        element (str): 元素名称
        save_path (str): 保存路径，默认为None
        figsize (tuple): 图表尺寸
    """
    # 确保输入是一维数组
    if len(targets.shape) > 1 and targets.shape[1] == 1:
        targets = targets.flatten()
    if len(outputs.shape) > 1 and outputs.shape[1] == 1:
        outputs = outputs.flatten()
    
    # 计算误差
    errors = outputs - targets
    
    # 计算评估指标
    mae = np.mean(np.abs(errors))
    std = np.std(errors)
    
    # 创建图表
    plt.figure(figsize=figsize)
    
    # 绘制误差直方图
    plt.hist(errors, bins=30, alpha=0.7, density=True)
    
    # 添加垂直线表示均值
    plt.axvline(x=0, color='r', linestyle='--', label='零误差线')
    plt.axvline(x=np.mean(errors), color='g', linestyle='-', label=f'均值 ({np.mean(errors):.4f})')
    
    # 设置标题和标签
    plt.title(f'{element} 预测误差分布')
    plt.xlabel('预测误差 (预测值 - 真实值)')
    plt.ylabel('密度')
    plt.grid(True, alpha=0.3)
    
    # 添加评估指标文本
    metrics_text = (
        f"MAE: {mae:.4f}\n"
        f"标准差: {std:.4f}\n"
        f"均值: {np.mean(errors):.4f}\n"
        f"中位数: {np.median(errors):.4f}\n"
        f"最大误差: {np.max(np.abs(errors)):.4f}\n"
        f"样本数: {len(errors)}"
    )
    plt.annotate(
        metrics_text,
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.legend()
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        plt.close()
        return None 