#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具模块：提供缓存和进度管理功能
"""

import os
import json
import pickle
import logging
from datetime import datetime
from tqdm import tqdm

logger = logging.getLogger('utils')

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir='cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_info_file = os.path.join(cache_dir, 'cache_info.json')
        self.cache_info = self._load_cache_info()
    
    def _load_cache_info(self):
        """加载缓存信息"""
        if os.path.exists(self.cache_info_file):
            try:
                with open(self.cache_info_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache_info(self):
        """保存缓存信息"""
        with open(self.cache_info_file, 'w') as f:
            json.dump(self.cache_info, f, indent=4)
    
    def get_cache(self, key):
        """获取缓存数据"""
        if key in self.cache_info:
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except:
                    return None
        return None
    
    def set_cache(self, key, data):
        """设置缓存数据"""
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            self.cache_info[key] = {
                'timestamp': datetime.now().isoformat(),
                'size': os.path.getsize(cache_file)
            }
            self._save_cache_info()
            return True
        except:
            return False
    
    def clear_cache(self, key=None):
        """清除缓存"""
        if key is None:
            # 清除所有缓存
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, file))
            self.cache_info = {}
            self._save_cache_info()
        else:
            # 清除特定缓存
            if key in self.cache_info:
                cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                del self.cache_info[key]
                self._save_cache_info()
    
    def get_cache_size(self):
        """获取缓存总大小（字节）"""
        total_size = 0
        for info in self.cache_info.values():
            total_size += info['size']
        return total_size
    
    def list_caches(self):
        """列出所有缓存"""
        return self.cache_info

def ask_clear_cache(cache_manager):
    """询问是否清除缓存"""
    if cache_manager.cache_info:
        print("\n发现现有缓存:")
        for key, info in cache_manager.cache_info.items():
            print(f"- {key}: {datetime.fromisoformat(info['timestamp'])}")
        print(f"\n总缓存大小: {cache_manager.get_cache_size() / 1024 / 1024:.2f} MB")
        
        while True:
            response = input("\n是否清除缓存？(y/n): ").lower()
            if response in ['y', 'n']:
                if response == 'y':
                    cache_manager.clear_cache()
                    print("已清除所有缓存")
                return response == 'y'
            print("请输入 y 或 n")

class ProgressManager:
    """进度管理器"""
    
    def __init__(self, total, desc=None):
        self.total = total
        self.desc = desc or "处理进度"
        self.pbar = None
    
    def __enter__(self):
        self.pbar = tqdm(total=self.total, desc=self.desc)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar:
            self.pbar.close()
    
    def update(self, n=1):
        """更新进度"""
        if self.pbar:
            self.pbar.update(n)
    
    def set_description(self, desc):
        """设置描述"""
        if self.pbar:
            self.pbar.set_description(desc) 