import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from config import CONFIG

class XGBoostModel:
    """
    XGBoost模型
    特点：
    1. 优秀的特征处理能力
    2. 内置正则化，防止过拟合
    3. 支持并行计算
    4. 处理高维特征的能力强
    """
    def __init__(self, config=CONFIG):
        self.params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.01,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'seed': config['data']['random_state']
        }
        self.model = None
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        训练XGBoost模型
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # 训练
        self.model = xgb.train(
            self.params,
            dtrain,
            evals=[(dtrain, 'train'), (dval, 'val')],
            verbose_eval=100
        )
    
    def predict(self, X):
        """
        预测
        """
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def save(self, element):
        """
        保存模型
        """
        model_path = os.path.join(CONFIG['output']['model_dir'], f'xgboost_{element}.json')
        self.model.save_model(model_path)
    
    def load(self, element):
        """
        加载模型
        """
        self.model = xgb.Booster()
        model_path = os.path.join(CONFIG['output']['model_dir'], f'xgboost_{element}.json')
        self.model.load_model(model_path)

class LightGBMModel:
    """
    LightGBM模型
    特点：
    1. 更快的训练速度
    2. 更低的内存消耗
    3. 更好的准确性
    4. 支持直接处理类别特征
    """
    def __init__(self, config=CONFIG):
        self.params = {
            'objective': 'regression',
            'metric': 'mse',
            'learning_rate': 0.01,
            'num_leaves': 31,
            'max_depth': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'seed': config['data']['random_state']
        }
        self.model = None
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        训练LightGBM模型
        """
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 训练
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, val_data],
            verbose_eval=100
        )
    
    def predict(self, X):
        """
        预测
        """
        return self.model.predict(X)
    
    def save(self, element):
        """
        保存模型
        """
        model_path = os.path.join(CONFIG['output']['model_dir'], f'lightgbm_{element}.txt')
        self.model.save_model(model_path)
    
    def load(self, element):
        """
        加载模型
        """
        self.model = lgb.Booster()
        model_path = os.path.join(CONFIG['output']['model_dir'], f'lightgbm_{element}.txt')
        self.model.load_model(model_path)

def evaluate_baseline_model(y_true, y_pred, model_name, element):
    """
    评估基线模型性能
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    dex = np.std(y_pred - y_true)
    
    results = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'dex': dex
    }
    
    # 保存结果
    results_dir = CONFIG['output']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f'{model_name}_{element}_results.txt')
    
    with open(results_path, 'w') as f:
        f.write(f"Evaluation Results for {model_name} on {element}\n")
        f.write("=" * 50 + "\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"R²: {r2:.4f}\n")
        f.write(f"Prediction Precision (DEX): {dex:.4f}\n")
    
    return results 