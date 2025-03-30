import numpy as np
import os
from tqdm import tqdm
import argparse

def merge_npz_files(input_files, output_file, is_train_data=True):
    """
    将多个NPZ文件合并成一个NPZ文件，专为光谱数据设计
    
    参数:
        input_files: npz文件路径列表
        output_file: 输出文件路径
        is_train_data: 是否为训练数据（决定如何分割验证集）
    """
    # 初始化存储所有数据的列表
    all_X = []
    all_y = []
    all_elements = []
    
    # 遍历每个输入文件
    print(f"正在合并 {len(input_files)} 个NPZ文件...")
    for file_path in tqdm(input_files, desc="合并文件"):
        # 加载npz文件
        data = np.load(file_path)
        
        # 将数据添加到合并列表中
        all_X.append(data['X'])
        all_y.append(data['y'])
        
        # 元素标识可能是一维数组或字符串数组
        if 'elements' in data:
            all_elements.append(data['elements'])
        
        # 关闭文件
        data.close()
    
    # 合并数组
    X_merged = np.vstack(all_X) if all_X else np.array([])
    y_merged = np.concatenate(all_y) if all_y else np.array([])
    
    # 合并元素标识（如果存在）
    if all_elements:
        try:
            elements_merged = np.concatenate(all_elements)
        except:
            # 如果无法直接连接，保留第一个文件的元素标识
            elements_merged = all_elements[0]
            print("警告: 无法合并元素标识，使用第一个文件的值")
    else:
        elements_merged = np.array([])
    
    # 如果是训练数据，可以选择拆分为训练集和验证集
    if is_train_data:
        # 计算拆分点（例如80%训练，20%验证）
        split_point = int(len(X_merged) * 0.8)
        
        # 创建训练集
        train_file = output_file.replace('.npz', '_train.npz')
        np.savez(train_file, 
                X=X_merged[:split_point], 
                y=y_merged[:split_point],
                elements=elements_merged[:split_point] if len(elements_merged) > 0 else elements_merged)
        
        # 创建验证集
        val_file = output_file.replace('.npz', '_val.npz')
        np.savez(val_file, 
                X=X_merged[split_point:], 
                y=y_merged[split_point:],
                elements=elements_merged[split_point:] if len(elements_merged) > 0 else elements_merged)
        
        print(f"已创建训练集 ({split_point} 样本): {train_file}")
        print(f"已创建验证集 ({len(X_merged) - split_point} 样本): {val_file}")
    else:
        # 保存所有合并后的数据
        np.savez(output_file, 
                X=X_merged, 
                y=y_merged,
                elements=elements_merged)
        
        print(f"已合并 {len(X_merged)} 个样本到: {output_file}")
    
    # 打印合并后数据的形状信息
    print(f"X形状: {X_merged.shape}")
    print(f"y形状: {y_merged.shape}")
    if len(elements_merged) > 0:
        print(f"elements形状: {elements_merged.shape}")
    
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='合并多个NPZ文件')
    parser.add_argument('--input_files', nargs='+', required=True, help='输入NPZ文件路径列表')
    parser.add_argument('--output_file', required=True, help='输出文件路径')
    parser.add_argument('--is_train', action='store_true', help='是否为训练数据（如果是，将拆分为训练集和验证集）')
    parser.add_argument('--test_split', action='store_true', help='是否拆分测试集（将不再拆分验证集）')
    
    args = parser.parse_args()
    
    if args.test_split:
        # 直接保存为test文件
        merged_file = merge_npz_files(args.input_files, args.output_file, is_train_data=False)
    else:
        # 正常处理
        merged_file = merge_npz_files(args.input_files, args.output_file, is_train_data=args.is_train)
    
    print(f"处理完成! 最终文件: {merged_file}")