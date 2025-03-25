import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def train_test_split_data(data, test_size=0.2, random_state=42):
    """将数据集分割为训练集和测试集"""
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data

class LAMOSTDataset(Dataset):
    """LAMOST数据集类"""
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        wavelength = torch.FloatTensor(item['wavelength'])
        flux = torch.FloatTensor(item['flux'])
        label = torch.FloatTensor([item['label']])
        return wavelength, flux, label

def create_data_loaders(train_data, val_data, test_data, batch_size=32, num_workers=4):
    """创建数据加载器"""
    train_dataset = LAMOSTDataset(train_data)
    val_dataset = LAMOSTDataset(val_data)
    test_dataset = LAMOSTDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader 