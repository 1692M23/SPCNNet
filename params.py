from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

# 添加数据集类定义
class SpectralDataset(Dataset):
    """光谱数据集类"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(1)  # 添加通道维度
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 加载数据函数
def load_data(file_path):
    """加载预处理好的光谱数据，兼容preprocessdata7.py格式"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到数据文件: {file_path}")
    
    data = np.load(file_path, allow_pickle=True)
    
    # 支持preprocessdata7.py输出的不同键名
    try:
        # 尝试不同的关键字来获取数据
        if 'X' in data:
            X = data['X']
        elif 'spectra' in data:
            X = data['spectra']
        elif 'flux' in data:
            X = data['flux']
        else:
            # 尝试使用第一个键，如果没有特定的键
            first_key = list(data.keys())[0]
            print(f"无法找到标准特征键名，使用第一个可用键: {first_key}")
            X = data[first_key]
        
        # 获取标签
        if 'y' in data:
            y = data['y']
        elif 'labels' in data:
            y = data['labels']
        elif 'abundance' in data:
            y = data['abundance']
        else:
            # 尝试使用第二个键
            keys = list(data.keys())
            if len(keys) > 1:
                second_key = keys[1]
                print(f"无法找到标准标签键名，使用第二个可用键: {second_key}")
                y = data[second_key]
            else:
                raise KeyError("无法找到光谱或标签数据")
        
        print(f"成功加载数据: X形状={X.shape}, y形状={y.shape}")
        return X, y
    except KeyError as e:
        # 如果发生错误，尝试加载替代键
        print(f"加载数据时出错: {e}")
        print("可用键:", list(data.keys()))
        raise e
    except Exception as e:
        print(f"加载数据失败: {e}")
        raise e

# 定义SPCNNet模型（保持CNN+残差模块不变）
class SPCNNet(nn.Module):
    def __init__(self, input_size, num_filters, filter_size, hidden_size):
        super(SPCNNet, self).__init__()
        self.conv1 = nn.Conv1d(1, num_filters, filter_size, padding='same')
        self.conv2 = nn.Conv1d(num_filters, num_filters, filter_size, padding='same')
        self.pool = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(num_filters * (input_size // 4), hidden_size)
        self.fc2 = nn.Linear(hidden_size, 3)  # 输出3个恒星参数：Teff, log g, [M/H]
        self.relu = nn.ReLU()
        self.residual = nn.Conv1d(1, num_filters, 1)  # 残差连接

    def forward(self, x):
        residual = self.residual(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x) + residual)  # 残差模块
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练函数（添加L2正则化）
def train_model(model, train_loader, val_loader, epochs, lr, weight_decay=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience, early_stop = 10, 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
        
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= patience:
                break
    
    return {'val_loss': best_val_loss}

# 加载训练和验证数据集
try:
    # 尝试从processed_data目录加载数据
    X_train, y_train = load_data('processed_data/train_dataset.npz')
    X_val, y_val = load_data('processed_data/val_dataset.npz')
    
    # 创建数据集
    train_dataset = SpectralDataset(X_train, y_train)
    val_dataset = SpectralDataset(X_val, y_val)
    print(f"成功加载数据集: 训练集{len(train_dataset)}样本, 验证集{len(val_dataset)}样本")
except Exception as e:
    print(f"加载数据集失败: {e}")
    print("请确保处理好的数据集文件存在于processed_data目录中")
    # 创建一些示例数据作为占位符
    print("创建示例数据集...")
    X_train = np.random.rand(100, 3000)
    y_train = np.random.rand(100, 3)
    X_val = np.random.rand(20, 3000)
    y_val = np.random.rand(20, 3)
    train_dataset = SpectralDataset(X_train, y_train)
    val_dataset = SpectralDataset(X_val, y_val)

# 目标函数：评估模型性能
def objective(params):
    """
    目标函数：根据超参数创建并训练模型，返回验证集上的损失。
    输入：params - 超参数字典
    输出：字典，包含损失值和状态
    """
    model = SPCNNet(
        input_size=params['input_size'],
        num_filters=params['num_filters'],
        filter_size=params['filter_size'],
        hidden_size=params['hidden_size']
    )
    # 使用全局变量中的数据集创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
    
    history = train_model(model, train_loader, val_loader, epochs=50, lr=params['lr'])
    return {'loss': history['val_loss'], 'status': STATUS_OK}

# 第一阶段：随机搜索初步确定架构
space_stage1 = {
    'input_size': hp.choice('input_size', [3000]),  # 固定输入尺寸，基于LAMOST光谱
    'num_filters': hp.choice('num_filters', [16, 32, 64]),  # 卷积层滤波器数量
    'filter_size': hp.choice('filter_size', [3, 5, 7]),  # 滤波器长度
    'hidden_size': hp.choice('hidden_size', [128, 256, 512]),  # 全连接层节点数
    'batch_size': hp.choice('batch_size', [16, 32, 64]),  # 批量大小
    'lr': hp.loguniform('lr', -5, -3)  # 学习率对数均匀分布
}

trials_stage1 = Trials()
best_stage1 = fmin(
    fn=objective,
    space=space_stage1,
    algo=tpe.suggest,  # 使用TPE算法
    max_evals=30,  # 初步评估30次
    trials=trials_stage1
)

# 第二阶段：基于第一阶段结果，精细调整关键参数
space_stage2 = {
    'input_size': hp.choice('input_size', [3000]),
    'num_filters': hp.quniform('num_filters', best_stage1['num_filters']-8, best_stage1['num_filters']+8, 1),
    'filter_size': hp.choice('filter_size', [best_stage1['filter_size']]),
    'hidden_size': hp.quniform('hidden_size', best_stage1['hidden_size']-64, best_stage1['hidden_size']+64, 1),
    'batch_size': hp.choice('batch_size', [best_stage1['batch_size']]),
    'lr': hp.loguniform('lr', -5, -3)
}

trials_stage2 = Trials()
best = fmin(
    fn=objective,
    space=space_stage2,
    algo=tpe.suggest,
    max_evals=20,  # 精细调整20次
    trials=trials_stage2
)

print("最佳超参数:", best)