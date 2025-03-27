import os
import pickle
import logging
import torch
from sklearn.model_selection import ParameterGrid
from utils import ProgressManager
import config
from main import train_and_evaluate_model

# 设置logger
logger = logging.getLogger(__name__)

def hyperparameter_tuning(element, train_loader, val_loader, grid=None, device=None):
    """
    超参数调优
    
    参数:
        element (str): 元素名称
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        grid (dict): 网格搜索参数
        device (str): 计算设备
        
    返回:
        dict: 最佳超参数
    """
    if device is None:
        device = config.training_config['device']
    
    # 尝试使用批量超参数优化
    try:
        from batch_hyperopt import run_element_hyperopt
        
        logger.info(f"使用批量超参数优化方法为 {element} 获取2组最优超参数")
        
        # 首先尝试读取已有结果
        results_dir = os.path.join(config.output_config['results_dir'], 'hyperopt')
        os.makedirs(results_dir, exist_ok=True)
        
        final_results_file = os.path.join(results_dir, f'{element}_best_params.pkl')
        
        # 如果存在最终结果文件并且不再需要更新，直接加载并返回结果
        if os.path.exists(final_results_file):
            try:
                with open(final_results_file, 'rb') as f:
                    results = pickle.load(f)
                    
                # 检查是否已完成足够批次的处理
                if len(results.get('processed_batches', [])) >= 5:  # 假设5批次足够
                    logger.info(f"从缓存加载 {element} 的最佳超参数: set1={results['best_params_set1']}, set2={results['best_params_set2']}")
                    
                    # 返回第一组参数（后续可以根据需要选择使用哪组参数）
                    return results['best_params_set1']
            except Exception as e:
                logger.warning(f"读取缓存文件失败: {e}，将重新进行超参数优化")
        
        # 如果没有缓存结果或需要继续优化，运行批量优化
        batch_size = 1000  # 每批数据大小
        batches_per_round = 2  # 每轮处理的批次数，这里设置较小以便于测试
        
        # 运行批量超参数优化
        results = run_element_hyperopt(
            element=element,
            batch_size=batch_size,
            batches_per_round=batches_per_round
        )
        
        if results:
            logger.info(f"{element} 批量超参数优化完成:")
            logger.info(f"最佳参数组1: {results['best_params_set1']}")
            logger.info(f"最佳参数组2: {results['best_params_set2']}")
            
            # 默认返回第一组参数
            return results['best_params_set1']
            
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning(f"无法使用批量超参数优化模块: {e}")
    
    # 检查是否可以使用高级优化方法（hyperopt）
    try:
        from hyperopt_tuning import run_hyperopt_tuning, load_best_params
        
        # 检查是否已有保存的超参数
        saved_params = load_best_params(element)
        if saved_params:
            logger.info(f"从缓存加载 {element} 的最佳超参数: {saved_params}")
            return saved_params
        
        # 使用两阶段超参数优化
        logger.info(f"使用两阶段hyperopt优化 {element} 的超参数")
        best_params = run_hyperopt_tuning(
            element=element,
            train_loader=train_loader,
            val_loader=val_loader,
            max_evals_stage1=20,  # 减少评估次数以节省时间
            max_evals_stage2=10,
            device=device
        )
        
        return best_params
        
    except (ImportError, ModuleNotFoundError):
        logger.warning("无法导入hyperopt_tuning模块，回退到网格搜索方法")
    
    # 回退到简单的网格搜索
    # 如果没有提供网格参数，使用默认值
    if grid is None:
        grid = {
            'lr': [0.001, 0.0005, 0.0001],
            'weight_decay': [1e-4, 1e-5, 1e-6],
            'num_blocks': [2, 3, 4],
            'num_filters': [32, 64]
        }
    
    logger.info(f"开始 {element} 的超参数调优，参数网格:")
    for param, values in grid.items():
        logger.info(f"  {param}: {values}")
    
    # 生成所有参数组合
    param_combinations = list(ParameterGrid(grid))
    logger.info(f"共 {len(param_combinations)} 种参数组合")
    
    # 记录最佳结果
    best_val_loss = float('inf')
    best_params = None
    
    # 使用进度管理器
    with ProgressManager(len(param_combinations), desc=f"{element} 超参数调优") as progress:
        # 遍历所有参数组合
        for i, params in enumerate(param_combinations):
            logger.info(f"参数组合 {i+1}/{len(param_combinations)}: {params}")
            
            # 为当前超参数添加固定参数
            current_params = {
                **params,
                'num_epochs': min(config.training_config['num_epochs'], 50),  # 调优时使用较少的时代数
                'patience': config.training_config['early_stopping_patience']
            }
            
            # 训练模型
            try:
                _, val_loss, _ = train_and_evaluate_model(
                    f"{element}_tune_{i}",
                    train_loader,
                    val_loader,
                    hyperparams=current_params,
                    device=device
                )
                
                # 更新最佳参数
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = current_params
                    logger.info(f"找到新的最佳参数，验证损失: {best_val_loss:.6f}")
                    
            except Exception as e:
                logger.error(f"参数组合 {params} 训练失败: {str(e)}")
            
            # 更新进度
            progress.update(1)
    
    logger.info(f"超参数调优完成")
    logger.info(f"最佳参数: {best_params}")
    logger.info(f"最佳验证损失: {best_val_loss:.6f}")
    
    return best_params 