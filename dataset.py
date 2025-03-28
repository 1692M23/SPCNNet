import torch
from torch.utils.data import Dataset
import numpy as np

class LAMOSTDataset(Dataset):
    """LAMOST数据集类"""
    def __init__(self, data, transform=None):
        """
        初始化数据集
        
        参数:
            data: 包含光谱数据的列表
            transform: 可选的数据转换函数
        """
        self.data = []
        self.transform = transform
        
        # 数据验证和过滤
        for item in data:
            spectrum = item.get('data', item.get('spectrum'))  # 支持新旧格式
            label = item.get('metadata', {}).get('label', item.get('label'))
            
            if spectrum is None or label is None:
                print(f"警告: 跳过无效数据项 - 光谱或标签为None")
                continue
                
            if np.isnan(spectrum).any() or np.isinf(spectrum).any():
                print(f"警告: 跳过包含NaN或无穷值的数据项")
                continue
                
            if np.all(spectrum == 0):
                print(f"警告: 跳过全零光谱数据")
                continue
                
            if np.std(spectrum) < 1e-10:
                print(f"警告: 跳过标准差过小的光谱数据")
                continue
            
            self.data.append(item)
        
        print(f"数据集初始化完成: 有效数据项数量={len(self.data)}")
        
        # 计算数据集统计信息
        all_spectra = np.array([item.get('data', item.get('spectrum')) for item in self.data])
        all_labels = np.array([item.get('metadata', {}).get('label', item.get('label')) for item in self.data])
        
        self.spectrum_mean = np.mean(all_spectra, axis=0)
        self.spectrum_std = np.std(all_spectra, axis=0)
        self.label_mean = np.mean(all_labels)
        self.label_std = np.std(all_labels)
        
        # 防止除零
        self.spectrum_std = np.where(self.spectrum_std < 1e-10, 1.0, self.spectrum_std)
        self.label_std = max(self.label_std, 1e-10)
        
        print("\n=== 数据集统计信息 ===")
        print(f"光谱均值范围: {np.min(self.spectrum_mean):.4f} - {np.max(self.spectrum_mean):.4f}")
        print(f"光谱标准差范围: {np.min(self.spectrum_std):.4f} - {np.max(self.spectrum_std):.4f}")
        print(f"标签均值: {self.label_mean:.4f}")
        print(f"标签标准差: {self.label_std:.4f}")
        print("=== 统计信息结束 ===\n")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 获取光谱和标签，支持新旧格式
        spectrum = item.get('data', item.get('spectrum'))
        label = item.get('metadata', {}).get('label', item.get('label'))
        
        # 标准化数据
        spectrum = (spectrum - self.spectrum_mean) / self.spectrum_std
        label = (label - self.label_mean) / self.label_std
        
        # 转换为张量
        spectrum = torch.FloatTensor(spectrum)
        label = torch.FloatTensor([label])
        
        # 应用额外的转换（如果有）
        if self.transform is not None:
            spectrum = self.transform(spectrum)
        
        return spectrum, label
        
    def get_normalization_params(self):
        """返回数据集的归一化参数，用于预测时的标准化"""
        return {
            'spectrum_mean': self.spectrum_mean,
            'spectrum_std': self.spectrum_std,
            'label_mean': self.label_mean,
            'label_std': self.label_std
        }
        
    def inverse_transform_label(self, normalized_label):
        """将标准化的标签转换回原始值"""
        return normalized_label * self.label_std + self.label_mean 