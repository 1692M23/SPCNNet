#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
设置脚本：自动创建目录并安装依赖
"""

import os
import subprocess
import sys
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('setup')

def create_directories():
    """创建必要的目录"""
    directories = [
        'processed_data/cache',    # 缓存目录
        'processed_data/progress', # 进度保存目录
        'models',                 # 模型保存目录
        'logs',                   # 日志目录
        'results',                # 结果目录（包含预测结果）
        'plots'                   # 图表目录
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"已创建目录: {directory}")

def install_requirements():
    """安装依赖包"""
    requirements = [
        'numpy>=1.19.2',
        'pandas>=1.2.0',
        'matplotlib>=3.3.0',
        'torch>=1.7.0',
        'scikit-learn>=0.24.0',
        'astropy>=4.2',
        'scipy>=1.6.0',
        'seaborn>=0.11.0',
        'psutil>=5.8.0',
        'tqdm>=4.50.0',
        'joblib>=1.0.0'
    ]
    
    logger.info("安装依赖包...")
    try:
        for package in requirements:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)
        logger.info("依赖包安装成功")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"安装依赖包失败: {e}")
        return False

def setup_environment():
    """设置环境"""
    try:
        # 克隆仓库
        logger.info("克隆仓库...")
        repo_url = "https://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@github.com/1692M23/SPCNNet.git"
        try:
            subprocess.run(['git', 'clone', repo_url], check=True)
            logger.info("仓库克隆成功")
        except subprocess.CalledProcessError as e:
            logger.error(f"克隆仓库失败: {e}")
            return False
        
        # 创建目录
        create_directories()

        # 安装依赖
        if not install_requirements():
            return False

        logger.info("环境设置完成")
        return True

    except Exception as e:
        logger.error(f"环境设置失败: {e}")
        return False

if __name__ == '__main__':
    if setup_environment():
        logger.info("环境设置成功完成")
    else:
        logger.error("环境设置失败")
        sys.exit(1) 