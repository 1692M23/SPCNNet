#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
在Google Colab中设置TPU环境的辅助脚本
"""

import os
import sys
import subprocess
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('setup_tpu')

def install_tpu_dependencies():
    """安装TPU所需的依赖"""
    logger.info("开始安装TPU依赖...")
    
    # 安装PyTorch和PyTorch XLA
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch==2.0.0", "torch_xla==2.0.0", "torch_xla[tpu]",
            "-f", "https://storage.googleapis.com/libtpu-releases/index.html"
        ])
        logger.info("PyTorch和PyTorch XLA安装成功")
    except subprocess.CalledProcessError:
        logger.error("安装PyTorch和PyTorch XLA失败")
        return False
    
    # 设置TPU相关环境变量
    os.environ['XLA_USE_BF16'] = '1'  # 启用bfloat16支持
    os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'  # 增加内存分配
    
    logger.info("TPU环境变量设置完成")
    return True

def verify_tpu_availability():
    """验证TPU是否可用"""
    logger.info("验证TPU可用性...")
    
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        logger.info(f"XLA设备类型: {device.type}")
        
        # 测试简单张量操作
        import torch
        test_tensor = torch.randn(2, 3)
        test_tensor = test_tensor.to(device)
        
        # 强制执行计算
        xm.mark_step()
        
        logger.info("TPU测试成功，可以正常使用")
        return True
    except ImportError:
        logger.error("导入torch_xla失败，TPU环境未正确设置")
        return False
    except Exception as e:
        logger.error(f"TPU测试失败: {str(e)}")
        return False

def setup_parallel_loading():
    """设置并行数据加载"""
    logger.info("设置TPU并行数据加载支持...")
    
    try:
        import torch_xla.distributed.parallel_loader as pl
        logger.info("并行加载器导入成功")
        
        # 示例代码
        logger.info("""
使用并行加载器示例:
    
import torch_xla.distributed.parallel_loader as pl
import torch_xla.core.xla_model as xm

# 常规数据加载器
train_loader = torch.utils.data.DataLoader(...)

# TPU并行加载器
device = xm.xla_device()
train_loader_parallel = pl.MpDeviceLoader(train_loader, device)

# 使用并行加载器训练
for data, target in train_loader_parallel:
    # 训练代码...
    pass
""")
        return True
    except ImportError:
        logger.error("导入parallel_loader失败")
        return False
    except Exception as e:
        logger.error(f"设置并行加载支持失败: {str(e)}")
        return False

def setup_tpu_environment():
    """完整设置TPU环境"""
    logger.info("开始设置TPU环境...")
    
    # 检查是否在Colab中运行
    try:
        import google.colab
        logger.info("检测到Google Colab环境")
    except ImportError:
        logger.warning("当前不是在Google Colab中运行，某些功能可能不可用")
    
    # 依次执行设置步骤
    if install_tpu_dependencies():
        logger.info("TPU依赖安装成功")
    else:
        logger.error("TPU依赖安装失败，请手动安装")
        return False
    
    if verify_tpu_availability():
        logger.info("TPU可用性验证成功")
    else:
        logger.error("TPU可用性验证失败，请检查运行时类型是否设置为TPU")
        return False
    
    if setup_parallel_loading():
        logger.info("并行数据加载设置成功")
    else:
        logger.warning("并行数据加载设置失败，但这不影响基本功能")
    
    logger.info("TPU环境设置完成，可以开始使用")
    return True

if __name__ == "__main__":
    success = setup_tpu_environment()
    
    if success:
        print("\n" + "="*50)
        print("TPU环境设置成功！")
        print("现在可以导入相关模块并开始训练")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("TPU环境设置遇到问题!")
        print("请检查错误信息并尝试手动设置")
        print("="*50) 