#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据验证工具：提供数据验证、监控和可视化功能
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import json

logger = logging.getLogger('data_validator')

class DataValidator:
    """数据验证工具类"""
    
    def __init__(self, output_dir='processed_data/validation'):
        """
        初始化数据验证工具
        
        参数:
            output_dir (str): 验证结果输出目录
        """
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, 'plots')
        
        # 创建必要的目录
        for directory in [output_dir, self.plots_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # 验证指标阈值
        self.thresholds = {
            'snr_min': 3.0,  # 最小信噪比
            'wavelength_coverage': 0.9,  # 最小波长覆盖率
            'flux_outlier_std': 3.0,  # 异常值标准差倍数
            'max_nan_ratio': 0.1,  # 最大NaN比例
            'min_points': 1000,  # 最小数据点数
        }
    
    def validate_fits_data(self, data_dict, obsid=None, save_plots=True):
        """
        验证FITS数据的质量
        
        参数:
            data_dict (dict): FITS数据字典
            obsid (str): 观测ID，用于保存验证结果
            save_plots (bool): 是否保存验证图表
            
        返回:
            tuple: (是否通过验证, 验证报告)
        """
        validation_report = {
            'passed': False,
            'metrics': {},
            'issues': [],
            'plots': {}
        }
        
        try:
            # 1. 检查必需的键
            required_keys = {'wavelength', 'flux', 'error', 'mask'}
            if not all(key in data_dict for key in required_keys):
                missing_keys = required_keys - set(data_dict.keys())
                validation_report['issues'].append(f"缺少必需的键: {missing_keys}")
                return False, validation_report
            
            # 2. 获取数据
            wavelength = data_dict['wavelength']
            flux = data_dict['flux']
            error = data_dict['error']
            mask = data_dict['mask']
            
            # 3. 基本数据检查
            if not all(isinstance(arr, np.ndarray) for arr in [wavelength, flux, error, mask]):
                validation_report['issues'].append("数据类型错误：不是NumPy数组")
                return False, validation_report
            
            if not all(len(arr) > self.thresholds['min_points'] for arr in [wavelength, flux, error, mask]):
                validation_report['issues'].append(f"数据点数不足: 需要至少{self.thresholds['min_points']}个点")
                return False, validation_report
            
            # 4. 计算基本指标
            # 信噪比
            snr = np.nanmedian(np.abs(flux) / error)
            validation_report['metrics']['snr'] = snr
            
            # 波长覆盖率
            valid_points = np.sum(~np.isnan(flux))
            coverage = valid_points / len(flux)
            validation_report['metrics']['coverage'] = coverage
            
            # NaN比例
            nan_ratio = np.sum(np.isnan(flux)) / len(flux)
            validation_report['metrics']['nan_ratio'] = nan_ratio
            
            # 异常值检测
            flux_std = np.nanstd(flux)
            flux_mean = np.nanmean(flux)
            outliers = np.abs(flux - flux_mean) > self.thresholds['flux_outlier_std'] * flux_std
            outlier_ratio = np.sum(outliers) / len(flux)
            validation_report['metrics']['outlier_ratio'] = outlier_ratio
            
            # 5. 检查指标是否满足阈值
            if snr < self.thresholds['snr_min']:
                validation_report['issues'].append(f"信噪比过低: {snr:.2f} < {self.thresholds['snr_min']}")
            
            if coverage < self.thresholds['wavelength_coverage']:
                validation_report['issues'].append(f"波长覆盖率不足: {coverage:.2f} < {self.thresholds['wavelength_coverage']}")
            
            if nan_ratio > self.thresholds['max_nan_ratio']:
                validation_report['issues'].append(f"NaN比例过高: {nan_ratio:.2f} > {self.thresholds['max_nan_ratio']}")
            
            # 6. 生成验证图表
            if save_plots and obsid is not None:
                plots = self._generate_validation_plots(wavelength, flux, error, mask, obsid)
                validation_report['plots'] = plots
            
            # 7. 确定最终验证结果
            validation_report['passed'] = len(validation_report['issues']) == 0
            
            return validation_report['passed'], validation_report
            
        except Exception as e:
            validation_report['issues'].append(f"验证过程出错: {str(e)}")
            return False, validation_report
    
    def _generate_validation_plots(self, wavelength, flux, error, mask, obsid):
        """
        生成验证图表
        
        参数:
            wavelength (np.ndarray): 波长数据
            flux (np.ndarray): 流量数据
            error (np.ndarray): 误差数据
            mask (np.ndarray): 掩码数据
            obsid (str): 观测ID
            
        返回:
            dict: 图表文件路径字典
        """
        plots = {}
        
        try:
            # 1. 光谱总览图
            plt.figure(figsize=(12, 8))
            plt.subplot(211)
            plt.plot(wavelength, flux, 'b-', alpha=0.6, label='光谱')
            plt.fill_between(wavelength, flux-error, flux+error, color='gray', alpha=0.2, label='误差')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.title(f'光谱总览 (OBSID: {obsid})')
            
            plt.subplot(212)
            plt.plot(wavelength, error, 'r-', alpha=0.6, label='误差')
            plt.plot(wavelength, mask, 'g-', alpha=0.6, label='掩码')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            overview_plot = os.path.join(self.plots_dir, f'{obsid}_overview.png')
            plt.savefig(overview_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plots['overview'] = overview_plot
            
            # 2. 信噪比分布图
            plt.figure(figsize=(8, 6))
            snr = np.abs(flux / error)
            plt.hist(snr[~np.isnan(snr)], bins=50, alpha=0.6)
            plt.grid(True, alpha=0.3)
            plt.title(f'信噪比分布 (OBSID: {obsid})')
            plt.xlabel('信噪比')
            plt.ylabel('频数')
            
            snr_plot = os.path.join(self.plots_dir, f'{obsid}_snr.png')
            plt.savefig(snr_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plots['snr'] = snr_plot
            
            # 3. 异常值检测图
            plt.figure(figsize=(12, 6))
            flux_std = np.nanstd(flux)
            flux_mean = np.nanmean(flux)
            outliers = np.abs(flux - flux_mean) > self.thresholds['flux_outlier_std'] * flux_std
            
            plt.plot(wavelength[~outliers], flux[~outliers], 'b.', alpha=0.6, label='正常值')
            plt.plot(wavelength[outliers], flux[outliers], 'r.', alpha=0.6, label='异常值')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.title(f'异常值检测 (OBSID: {obsid})')
            
            outlier_plot = os.path.join(self.plots_dir, f'{obsid}_outliers.png')
            plt.savefig(outlier_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plots['outliers'] = outlier_plot
            
            return plots
            
        except Exception as e:
            logger.error(f"生成验证图表时出错: {e}")
            return plots
    
    def validate_batch(self, data_list, save_summary=True):
        """
        批量验证数据
        
        参数:
            data_list (list): 数据字典列表，每个字典包含obsid和data_dict
            save_summary (bool): 是否保存验证摘要
            
        返回:
            dict: 验证结果摘要
        """
        summary = {
            'total': len(data_list),
            'passed': 0,
            'failed': 0,
            'issues': {},
            'metrics': {
                'snr': [],
                'coverage': [],
                'nan_ratio': [],
                'outlier_ratio': []
            }
        }
        
        for item in data_list:
            obsid = item.get('obsid', 'unknown')
            data_dict = item.get('data_dict')
            
            if data_dict is None:
                summary['failed'] += 1
                continue
            
            passed, report = self.validate_fits_data(data_dict, obsid)
            
            if passed:
                summary['passed'] += 1
            else:
                summary['failed'] += 1
                
                # 记录问题
                for issue in report['issues']:
                    if issue not in summary['issues']:
                        summary['issues'][issue] = 0
                    summary['issues'][issue] += 1
            
            # 记录指标
            metrics = report.get('metrics', {})
            for key in summary['metrics']:
                if key in metrics:
                    summary['metrics'][key].append(metrics[key])
        
        # 计算指标统计
        for key in summary['metrics']:
            values = summary['metrics'][key]
            if values:
                summary['metrics'][key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # 保存验证摘要
        if save_summary:
            summary_file = os.path.join(self.output_dir, 'validation_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4)
        
        return summary
    
    def plot_validation_summary(self, summary):
        """
        绘制验证结果摘要图表
        
        参数:
            summary (dict): 验证结果摘要
        """
        # 1. 通过/失败比例饼图
        plt.figure(figsize=(8, 8))
        plt.pie([summary['passed'], summary['failed']], 
                labels=['通过', '失败'],
                colors=['lightgreen', 'salmon'],
                autopct='%1.1f%%')
        plt.title('验证结果分布')
        
        pie_plot = os.path.join(self.plots_dir, 'validation_pie.png')
        plt.savefig(pie_plot, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 问题分布条形图
        if summary['issues']:
            plt.figure(figsize=(12, 6))
            issues = list(summary['issues'].keys())
            counts = list(summary['issues'].values())
            
            plt.barh(issues, counts)
            plt.title('问题分布')
            plt.xlabel('出现次数')
            
            issues_plot = os.path.join(self.plots_dir, 'issues_distribution.png')
            plt.savefig(issues_plot, dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 指标分布箱线图
        plt.figure(figsize=(10, 6))
        metrics_data = []
        labels = []
        for key, values in summary['metrics'].items():
            if isinstance(values, dict) and 'mean' in values:
                metrics_data.append([values['mean'], values['std']])
                labels.append(key)
        
        if metrics_data:
            plt.boxplot(metrics_data, labels=labels)
            plt.title('指标分布')
            plt.grid(True, alpha=0.3)
            
            metrics_plot = os.path.join(self.plots_dir, 'metrics_distribution.png')
            plt.savefig(metrics_plot, dpi=300, bbox_inches='tight')
            plt.close() 