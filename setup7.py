#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LAMOST光谱预处理环境设置脚本
创建必要目录并检查依赖
"""

import os
import sys
import subprocess
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('setup')

def create_directories():
    """创建必要的目录结构"""
    directories = [
        'fits',                                  # FITS文件目录
        'processed_data',                        # 处理结果主目录
        os.path.join('processed_data', 'cache'), # 缓存目录
        os.path.join('processed_data', 'progress') # 进度保存目录
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"已创建目录: {directory}")

def check_dependencies():
    """检查并安装依赖包"""
    dependencies = [
        'numpy', 
        'pandas', 
        'astropy', 
        'scipy', 
        'matplotlib', 
        'scikit-learn', 
        'tqdm', 
        'psutil'
    ]
    
    missing = []
    for package in dependencies:
        try:
            __import__(package)
            logger.info(f"检查通过: {package}")
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.warning(f"缺少以下依赖包: {', '.join(missing)}")
        install = input("是否自动安装缺失的依赖？(y/n): ").lower()
        if install == 'y':
            for package in missing:
                logger.info(f"正在安装 {package}...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    logger.info(f"{package} 安装成功")
                except:
                    logger.error(f"{package} 安装失败，请手动安装")
        else:
            logger.info("请手动安装缺失的依赖")
            sys.exit(1)

def check_csv_files():
    """检查CSV文件是否存在"""
    csv_files = ['C_FE.csv', 'MG_FE.csv', 'CA_FE.csv']
    missing = [f for f in csv_files if not os.path.exists(f)]
    
    if missing:
        logger.warning(f"缺少以下CSV文件: {', '.join(missing)}")
        logger.info("请确保以下文件存在于当前目录:")
        for csv in csv_files:
            logger.info(f"- {csv}")
        return False
    else:
        logger.info("CSV文件检查通过")
        return True

def check_fits_directory():
    """检查FITS目录"""
    if not os.path.exists('fits'):
        logger.error("FITS目录不存在，将创建空目录")
        os.makedirs('fits', exist_ok=True)
        return False
    
    fits_files = [f for f in os.listdir('fits') 
                 if f.endswith(('.fits', '.fits.gz', '.fit', '.fit.gz'))]
    
    if not fits_files:
        logger.warning("FITS目录中没有发现FITS文件")
        logger.info("请将FITS文件放入fits目录中")
        return False
    else:
        logger.info(f"FITS目录中发现{len(fits_files)}个文件")
        return True

def main():
    """主函数"""
    logger.info("开始LAMOST光谱预处理环境设置")
    
    # 创建目录
    create_directories()
    
    # 检查依赖
    check_dependencies()
    
    # 检查CSV和FITS文件
    csv_status = check_csv_files()
    fits_status = check_fits_directory()
    
    if csv_status and fits_status:
        logger.info("环境设置完成，可以运行preprocessdata7.py进行预处理")
    else:
        logger.warning("环境设置不完整，请解决上述问题后再运行预处理程序")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())