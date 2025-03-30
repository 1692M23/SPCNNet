# 创建ensemble_models.py文件

import os
import torch
import numpy as np
import logging
from model import SpectralResCNN, SpectralResCNN_GCN, train, evaluate_model
from utils import set_seed
import matplotlib.pyplot as plt

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ensemble")

def train_ensemble(element, train_loader, val_loader, test_loader, num_models=5, 
                  base_seed=42, device=None, use_gcn=False, use_gru=False):
    """
    训练模型集成
    
    参数:
        element (str): 元素名称
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器 
        test_loader: 测试数据加载器
        num_models (int): 集成的模型数量
        base_seed (int): 基础随机种子
        device: 计算设备
        use_gcn (bool): 是否使用GCN
        use_gru (bool): 是否使用GRU
        
    返回:
        list: 训练好的模型列表
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建保存模型的目录
    ensemble_dir = os.path.join('models', 'ensemble', element)
    os.makedirs(ensemble_dir, exist_ok=True)
    
    # 训练配置
    config = {
        'training': {
            'lr': 0.0005,
            'weight_decay': 1e-4,
            'batch_size': 64,
            'num_epochs': 120,
            'early_stopping_patience': 20,
            'device': device
        },
        'model_config': {
            'model_dir': ensemble_dir,
            'use_gcn': use_gcn,
            'use_gru': use_gru
        }
    }
    
    # 训练多个模型
    models = []
    for i in range(num_models):
        model_seed = base_seed + i * 10
        logger.info(f"训练模型 {i+1}/{num_models}，种子：{model_seed}")
        
        # 设置随机种子
        set_seed(model_seed)
        
        # 创建模型
        input_size = next(iter(train_loader))[0].shape[2]  # 假设数据形状为[batch, channel, length]
        
        if use_gcn:
            model = SpectralResCNN_GCN(input_size=input_size, device=device)
        else:
            model = SpectralResCNN(input_size=input_size).to(device)
        
        # 训练模型
        train_losses, val_losses = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            element=f"{element}_ensemble_{i+1}"
        )
        
        # 保存训练好的模型
        models.append(model)
        
        # 评估单个模型性能
        test_metrics = evaluate_model(model, test_loader, device)
        logger.info(f"模型 {i+1} 测试集性能: RMSE={test_metrics['rmse']:.4f}, R²={test_metrics['r2']:.4f}")
    
    # 对集成模型进行评估
    ensemble_predictions, ensemble_uncertainties = predict_with_ensemble(models, test_loader, device)
    
    # 计算真实值
    all_targets = []
    for _, targets in test_loader:
        all_targets.append(targets.cpu().numpy())
    all_targets = np.concatenate(all_targets)
    
    # 计算R²
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(all_targets, ensemble_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, ensemble_predictions))
    
    logger.info(f"集成模型测试集性能: RMSE={rmse:.4f}, R²={r2:.4f}")
    logger.info(f"预测不确定性标准差: {np.mean(ensemble_uncertainties):.4f}")
    
    # 保存预测结果
    results = {
        'true_values': all_targets,
        'ensemble_predictions': ensemble_predictions,
        'uncertainties': ensemble_uncertainties
    }
    np.savez(f'results/ensemble_{element}_predictions.npz', **results)

    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.scatter(all_targets, ensemble_predictions, alpha=0.5)
    plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(f'{element} 集成模型预测 vs 真实值')
    plt.savefig(f'plots/{element}_ensemble_predictions.png')
    
    return models

def predict_with_ensemble(models, data_loader, device=None):
    """
    使用模型集成进行预测
    
    参数:
        models (list): 模型列表
        data_loader: 数据加载器
        device: 计算设备
        
    返回:
        tuple: (平均预测值, 预测不确定性)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 收集每个模型的预测结果
    all_model_predictions = []
    
    for model in models:
        model.eval()
        model_predictions = []
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(device)
                outputs = model(data)
                model_predictions.append(outputs.cpu().numpy())
        
        # 合并批次预测
        model_predictions = np.concatenate(model_predictions, axis=0)
        all_model_predictions.append(model_predictions)
    
    # 转换为numpy数组 [num_models, num_samples, 1]
    all_model_predictions = np.array(all_model_predictions)
    
    # 计算平均预测值 [num_samples, 1]
    ensemble_predictions = np.mean(all_model_predictions, axis=0)
    
    # 计算预测不确定性 (标准差) [num_samples, 1]
    ensemble_uncertainties = np.std(all_model_predictions, axis=0)
    
    return ensemble_predictions.squeeze(), ensemble_uncertainties.squeeze()

def analyze_ensemble_uncertainty(ensemble_predictions, ensemble_uncertainties, all_targets, element):
    """
    分析集成模型的不确定性与预测误差的关系
    
    参数:
        ensemble_predictions: 集成模型预测值
        ensemble_uncertainties: 预测不确定性
        all_targets: 真实值
        element: 元素名称
    """
    # 计算绝对误差
    abs_errors = np.abs(ensemble_predictions - all_targets)
    
    # 创建不确定性与误差的散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(ensemble_uncertainties, abs_errors, alpha=0.5)
    
    # 添加趋势线
    z = np.polyfit(ensemble_uncertainties, abs_errors, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(ensemble_uncertainties), p(np.sort(ensemble_uncertainties)), "r--")
    
    # 计算相关系数
    corr = np.corrcoef(ensemble_uncertainties, abs_errors)[0, 1]
    
    plt.xlabel('预测不确定性')
    plt.ylabel('绝对误差')
    plt.title(f'{element} 预测不确定性与绝对误差关系 (相关系数: {corr:.4f})')
    plt.grid(True)
    
    # 保存图表
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{element}_uncertainty_vs_error.png')
    plt.close()
    
    # 计算不确定性百分位数，并分析各个区间的误差
    percentiles = [0, 25, 50, 75, 100]
    uncertainty_thresholds = np.percentile(ensemble_uncertainties, percentiles)
    
    # 创建区间分析表格
    interval_metrics = []
    
    for i in range(len(percentiles) - 1):
        mask = (ensemble_uncertainties >= uncertainty_thresholds[i]) & (ensemble_uncertainties < uncertainty_thresholds[i+1])
        interval_errors = abs_errors[mask]
        
        if len(interval_errors) > 0:
            interval_metrics.append({
                'interval': f"{percentiles[i]}-{percentiles[i+1]}%",
                'samples': len(interval_errors),
                'mean_error': np.mean(interval_errors),
                'median_error': np.median(interval_errors),
                'max_error': np.max(interval_errors),
                'mean_uncertainty': np.mean(ensemble_uncertainties[mask])
            })
    
    # 保存区间分析
    with open(f'results/ensemble_{element}_uncertainty_analysis.txt', 'w') as f:
        f.write(f"集成模型不确定性分析 - {element}\n")
        f.write(f"总样本数: {len(abs_errors)}\n")
        f.write(f"平均不确定性: {np.mean(ensemble_uncertainties):.6f}\n")
        f.write(f"不确定性与绝对误差相关系数: {corr:.6f}\n\n")
        f.write("不确定性百分位区间分析:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'区间':<10} {'样本数':<10} {'平均误差':<12} {'中位数误差':<12} {'最大误差':<12} {'平均不确定性':<15}\n")
        f.write("-" * 80 + "\n")
        
        for metric in interval_metrics:
            f.write(f"{metric['interval']:<10} {metric['samples']:<10} {metric['mean_error']:.6f}    {metric['median_error']:.6f}    {metric['max_error']:.6f}    {metric['mean_uncertainty']:.6f}\n")
    
    logger.info(f"已完成 {element} 的不确定性分析")
    return corr

def analyze_model_diversity(models, test_loader, device=None):
    """
    分析集成模型中各个模型的多样性
    
    参数:
        models: 模型列表
        test_loader: 测试数据加载器
        device: 计算设备
    
    返回:
        模型之间的相关系数矩阵
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 收集每个模型的预测结果
    model_predictions = []
    
    # 收集真实标签
    all_targets = []
    
    for model in models:
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                if len(all_targets) < len(models):
                    all_targets.append(targets.cpu().numpy())
                
                data = data.to(device)
                outputs = model(data)
                predictions.append(outputs.cpu().numpy())
        
        # 合并批次预测
        predictions = np.concatenate(predictions, axis=0)
        model_predictions.append(predictions.squeeze())
    
    # 合并批次标签
    all_targets = np.concatenate(all_targets[0], axis=0)
    
    # 计算模型之间的相关系数
    num_models = len(models)
    correlation_matrix = np.zeros((num_models, num_models))
    
    for i in range(num_models):
        for j in range(num_models):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                correlation_matrix[i, j] = np.corrcoef(model_predictions[i], model_predictions[j])[0, 1]
    
    # 计算每个模型与真实值的相关系数
    true_correlations = []
    for i in range(num_models):
        true_correlations.append(np.corrcoef(model_predictions[i], all_targets)[0, 1])
    
    # 可视化相关系数矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=0, vmax=1)
    plt.colorbar(label='相关系数')
    plt.title('模型间预测相关系数矩阵')
    
    model_names = [f'Model {i+1}' for i in range(num_models)]
    plt.xticks(np.arange(num_models), model_names, rotation=45)
    plt.yticks(np.arange(num_models), model_names)
    
    # 在每个单元格上标注相关系数值
    for i in range(num_models):
        for j in range(num_models):
            plt.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                   ha="center", va="center", color="black" if correlation_matrix[i, j] < 0.7 else "white")
    
    plt.tight_layout()
    plt.savefig('plots/model_correlation_matrix.png')
    plt.close()
    
    # 可视化每个模型与真实值的相关系数
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, true_correlations)
    plt.xlabel('模型')
    plt.ylabel('与真实值的相关系数')
    plt.title('各模型与真实值的相关系数')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('plots/model_true_correlation.png')
    plt.close()
    
    # 返回相关系数矩阵
    return correlation_matrix, true_correlations

def weighted_ensemble_prediction(models, data_loader, weights=None, device=None):
    """
    使用加权集成进行预测
    
    参数:
        models: 模型列表
        data_loader: 数据加载器
        weights: 模型权重列表，如果为None则使用相等权重
        device: 计算设备
        
    返回:
        tuple: (加权预测值, 预测不确定性)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 如果未提供权重，则使用相等权重
    if weights is None:
        weights = np.ones(len(models)) / len(models)
    else:
        # 归一化权重
        weights = np.array(weights) / np.sum(weights)
    
    # 收集每个模型的预测结果
    all_model_predictions = []
    
    for model in models:
        model.eval()
        model_predictions = []
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(device)
                outputs = model(data)
                model_predictions.append(outputs.cpu().numpy())
        
        # 合并批次预测
        model_predictions = np.concatenate(model_predictions, axis=0)
        all_model_predictions.append(model_predictions)
    
    # 转换为numpy数组 [num_models, num_samples, 1]
    all_model_predictions = np.array(all_model_predictions)
    
    # 计算加权预测值 [num_samples, 1]
    weighted_predictions = np.sum(all_model_predictions * weights[:, np.newaxis, np.newaxis], axis=0)
    
    # 计算预测不确定性 (标准差) [num_samples, 1]
    uncertainties = np.std(all_model_predictions, axis=0)
    
    return weighted_predictions.squeeze(), uncertainties.squeeze()

def optimize_ensemble_weights(models, val_loader, device=None):
    """
    优化集成模型的权重
    
    参数:
        models: 模型列表
        val_loader: 验证数据加载器
        device: 计算设备
        
    返回:
        最优权重
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 收集每个模型在验证集上的预测
    model_predictions = []
    all_targets = []
    
    for model in models:
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                if len(all_targets) == 0:
                    all_targets.append(targets.cpu().numpy())
                
                data = data.to(device)
                outputs = model(data)
                predictions.append(outputs.cpu().numpy())
        
        # 合并批次预测
        predictions = np.concatenate(predictions, axis=0)
        model_predictions.append(predictions.squeeze())
    
    # 合并批次标签
    all_targets = np.concatenate(all_targets[0], axis=0)
    
    # 基于每个模型的性能计算权重
    weights = []
    
    for predictions in model_predictions:
        # 计算MSE
        mse = np.mean((predictions - all_targets) ** 2)
        # 权重与MSE成反比
        weights.append(1.0 / mse)
    
    # 归一化权重
    weights = np.array(weights) / np.sum(weights)
    
    return weights

if __name__ == "__main__":
    import argparse
    
    # 命令行参数
    parser = argparse.ArgumentParser(description='模型集成训练与评估')
    parser.add_argument('--element', type=str, required=True, help='元素名称')
    parser.add_argument('--num_models', type=int, default=5, help='集成模型数量')
    parser.add_argument('--base_seed', type=int, default=42, help='基础随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    parser.add_argument('--no_gcn', action='store_true', help='不使用GCN')
    parser.add_argument('--no_gru', action='store_true', help='不使用GRU')
    # 添加超参数参数
    parser.add_argument('--lr', type=float, default=0.0005, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=120, help='训练轮数')
    parser.add_argument('--early_stopping', type=int, default=20, help='早停轮数')
    parser.add_argument('--analyze_diversity', action='store_true', help='分析模型多样性')
    parser.add_argument('--optimize_weights', action='store_true', help='优化集成权重')
    args = parser.parse_args()
    
    # 从主模块加载数据
    from main import load_data, create_data_loaders
    
    # 加载数据
    train_path = os.path.join('processed_data', 'train_dataset.npz')
    val_path = os.path.join('processed_data', 'val_dataset.npz')
    test_path = os.path.join('processed_data', 'test_dataset.npz')
    
    # 加载训练、验证和测试数据
    X_train, y_train, elements_train = load_data(train_path, args.element)
    X_val, y_val, elements_val = load_data(val_path, args.element)
    X_test, y_test, elements_test = load_data(test_path, args.element)
    
    # 创建数据加载器
    train_loader = create_data_loaders(X_train, y_train, batch_size=args.batch_size)
    val_loader = create_data_loaders(X_val, y_val, batch_size=args.batch_size, shuffle=False)
    test_loader = create_data_loaders(X_test, y_test, batch_size=args.batch_size, shuffle=False)
    
    # 更新训练配置
    config = {
        'training': {
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'early_stopping_patience': args.early_stopping,
            'device': torch.device(args.device)
        },
        'model_config': {
            'model_dir': os.path.join('models', 'ensemble', args.element),
            'use_gcn': not args.no_gcn,
            'use_gru': not args.no_gru
        }
    }
    
    # 训练模型集成
    models = train_ensemble(
        args.element,
        train_loader,
        val_loader,
        test_loader,
        num_models=args.num_models,
        base_seed=args.base_seed,
        device=torch.device(args.device),
        use_gcn=not args.no_gcn,
        use_gru=not args.no_gru
    )
    
    # 对集成模型进行评估
    ensemble_predictions, ensemble_uncertainties = predict_with_ensemble(models, test_loader, torch.device(args.device))
    
    # 计算真实值
    all_targets = []
    for _, targets in test_loader:
        all_targets.append(targets.cpu().numpy())
    all_targets = np.concatenate(all_targets)
    
    # 分析不确定性
    analyze_ensemble_uncertainty(ensemble_predictions, ensemble_uncertainties, all_targets, args.element)
    
    # 分析模型多样性
    if args.analyze_diversity:
        correlation_matrix, true_correlations = analyze_model_diversity(models, test_loader, torch.device(args.device))
        logger.info(f"模型间相关系数矩阵:\n{correlation_matrix}")
        logger.info(f"各模型与真实值的相关系数: {true_correlations}")
    
    # 优化集成权重
    if args.optimize_weights:
        logger.info("优化集成权重...")
        optimal_weights = optimize_ensemble_weights(models, val_loader, torch.device(args.device))
        logger.info(f"最优权重: {optimal_weights}")
        
        # 使用最优权重进行预测
        weighted_predictions, weighted_uncertainties = weighted_ensemble_prediction(
            models, test_loader, weights=optimal_weights, device=torch.device(args.device))
        
        # 计算加权集成性能
        from sklearn.metrics import r2_score, mean_squared_error
        weighted_r2 = r2_score(all_targets, weighted_predictions)
        weighted_rmse = np.sqrt(mean_squared_error(all_targets, weighted_predictions))
        
        logger.info(f"优化权重后的集成模型性能: RMSE={weighted_rmse:.4f}, R²={weighted_r2:.4f}")
        
        # 可视化加权结果
        plt.figure(figsize=(10, 6))
        plt.scatter(all_targets, weighted_predictions, alpha=0.5)
        plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(f'{args.element} 加权集成预测 vs 真实值 (RMSE: {weighted_rmse:.4f})')
        plt.savefig(f'plots/{args.element}_weighted_ensemble_predictions.png')
        plt.close()
    
    print(f"成功训练 {len(models)} 个模型的集成")