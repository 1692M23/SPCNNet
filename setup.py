#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
设置脚本：自动创建目录并安装依赖
"""

import os
import subprocess
import sys
import logging
import pkg_resources
from packaging import version

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('setup')

def create_directories():
    """创建必要的目录"""
    directories = [
        os.path.join('processed_data', 'cache', 'predict'),  # 预测缓存目录
        os.path.join('processed_data', 'cache', 'train'),    # 训练缓存目录
        os.path.join('processed_data', 'progress'),          # 进度保存目录
        'models',                                            # 模型保存目录
        os.path.join('logs', 'train'),                      # 训练日志目录
        os.path.join('logs', 'predict'),                    # 预测日志目录
        os.path.join('results', 'predictions'),             # 预测结果目录
        os.path.join('results', 'evaluation'),              # 评估结果目录
        os.path.join('plots', 'preprocessing'),             # 预处理过程图表
        os.path.join('plots', 'training'),                  # 训练过程图表
        os.path.join('plots', 'predictions'),               # 预测结果图表
        os.path.join('plots', 'evaluation'),                # 评估结果图表
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"已创建目录: {directory}")

def get_installed_version(package_name):
    """获取已安装的包版本"""
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None

def parse_requirement(requirement):
    """解析依赖要求，返回包名和版本要求"""
    parts = requirement.split('>=')
    if len(parts) == 2:
        return parts[0], parts[1]
    return requirement, None

def check_version(current_version, required_version):
    """检查版本是否满足要求"""
    if not current_version or not required_version:
        return False
    return version.parse(current_version) >= version.parse(required_version)

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
        'joblib>=1.0.0',
        'hyperopt>=0.2.5'
    ]
    
    logger.info("检查依赖包...")
    try:
        for requirement in requirements:
            package_name, required_version = parse_requirement(requirement)
            current_version = get_installed_version(package_name)
            
            if current_version and required_version:
                if check_version(current_version, required_version):
                    logger.info(f"已安装 {package_name} {current_version} (满足要求 >={required_version})")
                    continue
                else:
                    logger.info(f"更新 {package_name} {current_version} -> >={required_version}")
            else:
                logger.info(f"安装 {package_name} >={required_version if required_version else ''}")
            
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', requirement], 
                             check=True, 
                             capture_output=True,
                             text=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"安装 {requirement} 失败: {e.stderr}")
                return False
                
        logger.info("所有依赖包安装成功")
        return True
    except Exception as e:
        logger.error(f"安装依赖包过程出错: {e}")
        return False

def setup_environment():
    """设置环境"""
    try:
        # 克隆或更新仓库
        logger.info("检查仓库状态...")
        repo_url = "https://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@github.com/1692M23/SPCNNet.git"
        
        if os.path.exists('SPCNNet'):
            logger.info("仓库已存在，尝试更新...")
            try:
                # 切换到仓库目录
                os.chdir('SPCNNet')
                # 重置所有更改
                subprocess.run(['git', 'reset', '--hard'], check=True)
                # 拉取最新代码
                subprocess.run(['git', 'pull'], check=True)
                # 切回上级目录
                os.chdir('..')
                logger.info("仓库更新成功")
            except subprocess.CalledProcessError as e:
                logger.error(f"更新仓库失败: {e}")
                return False
        else:
            logger.info("克隆新仓库...")
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