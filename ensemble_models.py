# 创建ensemble_models.py文件

import os
import torch
import numpy as np
import logging
from model import SpectralResCNN, SpectralResCNN_GCN, train, evaluate_model
from utils import set_seed

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
    args = parser.parse_args()
    
    # 从主模块加载数据
    from main import load_data, create_data_loaders
    
    # 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.element)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, 
        batch_size=64
    )
    
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
    
    print(f"成功训练 {len(models)} 个模型的集成")