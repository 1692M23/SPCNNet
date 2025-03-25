#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
设置脚本：自动创建目录、克隆/更新GitHub仓库、安装依赖
"""

import os
import subprocess
import sys
import logging
from setuptools import setup, find_packages

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

def update_github_repo(repo_url, branch='main'):
    """更新GitHub仓库"""
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    
    if os.path.exists(repo_name):
        logger.info(f"更新仓库: {repo_name}")
        try:
            subprocess.run(['git', '-C', repo_name, 'pull'], check=True)
            logger.info("仓库更新成功")
        except subprocess.CalledProcessError as e:
            logger.error(f"更新仓库失败: {e}")
            return False
    else:
        logger.info(f"克隆仓库: {repo_url}")
        try:
            subprocess.run(['git', 'clone', '-b', branch, repo_url], check=True)
            logger.info("仓库克隆成功")
        except subprocess.CalledProcessError as e:
            logger.error(f"克隆仓库失败: {e}")
            return False
    
    return True

def install_requirements():
    """安装依赖包"""
    requirements_file = 'requirements.txt'
    if os.path.exists(requirements_file):
        logger.info("安装依赖包...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', requirements_file], check=True)
            logger.info("依赖包安装成功")
        except subprocess.CalledProcessError as e:
            logger.error(f"安装依赖包失败: {e}")
            return False
    else:
        logger.warning(f"未找到 {requirements_file} 文件")
    
    return True

def setup_environment(repo_url, branch='main'):
    """设置环境"""
    try:
        # 创建目录
        create_directories()
        
        # 更新GitHub仓库
        if not update_github_repo(repo_url, branch):
            return False
        
        # 安装依赖
        if not install_requirements():
            return False
        
        logger.info("环境设置完成")
        return True
        
    except Exception as e:
        logger.error(f"环境设置失败: {e}")
        return False

def main():
    """主函数"""
    # 你的GitHub仓库URL
    repo_url = "https://github.com/1692M23/SPCNNet.git"
    
    # 设置环境
    if setup_environment(repo_url):
        logger.info("环境设置成功完成")
    else:
        logger.error("环境设置失败")
        sys.exit(1)

if __name__ == '__main__':
    main()

setup(
    name="lamost_abundance",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'torch',
        'scikit-learn',
        'astropy',
        'scipy',
        'seaborn',
        'psutil',
        'tqdm',
        'joblib'
    ],
    package_data={
        'lamost_abundance': [
            'fits/*',
            'processed_data/*',
            'models/*',
            'results/*',
            'plots/*'
        ]
    },
    include_package_data=True,
    python_requires='>=3.7'
) 