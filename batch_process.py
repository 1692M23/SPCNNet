import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import argparse

def process_csv_batch(csv_file, element, batch_size=10000, output_dir="processed_data/batches"):
    """
    处理CSV文件并按批次生成NPZ文件
    
    参数:
        csv_file: 输入CSV文件路径
        element: 要处理的元素名称
        batch_size: 每批处理的数据条数
        output_dir: 输出目录
    
    返回:
        生成的NPZ文件路径列表
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取CSV文件
    print(f"正在读取CSV文件: {csv_file}")
    df = pd.read_csv(csv_file)
    total_rows = len(df)
    print(f"共加载 {total_rows} 条数据")
    
    # 计算批次数量
    num_batches = (total_rows + batch_size - 1) // batch_size
    print(f"将分成 {num_batches} 个批次处理")
    
    npz_files = []
    
    # 按批次处理数据
    for i in tqdm(range(num_batches), desc="处理批次"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_rows)
        
        # 获取当前批次数据
        batch_df = df.iloc[start_idx:end_idx]
        
        # 处理数据 - 这里应根据实际预处理逻辑修改
        # 假设我们从现有代码中调用预处理函数
        from main import preprocess_data
        X_batch, y_batch, elements_batch = preprocess_data(batch_df, element)
        
        # 保存为NPZ文件
        batch_file = os.path.join(output_dir, f"{element}_batch_{i+1}.npz")
        np.savez(batch_file, 
                X=X_batch, 
                y=y_batch, 
                elements=elements_batch)
        
        npz_files.append(batch_file)
        print(f"已保存批次 {i+1}/{num_batches} 到 {batch_file}")
    
    return npz_files

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

def process_and_merge(csv_files, element, batch_size=10000, output_dir="processed_data", is_train=True):
    """
    处理多个CSV文件并将结果合并
    
    参数:
        csv_files: 输入CSV文件路径列表
        element: 要处理的元素名称
        batch_size: 每批处理的数据条数
        output_dir: 输出目录
        is_train: 是否为训练数据
    
    返回:
        最终合并后的NPZ文件路径
    """
    batch_dir = os.path.join(output_dir, "batches", element)
    os.makedirs(batch_dir, exist_ok=True)
    
    all_npz_files = []
    
    # 处理每个CSV文件
    for csv_file in csv_files:
        print(f"\n开始处理文件: {csv_file}")
        npz_files = process_csv_batch(csv_file, element, batch_size, batch_dir)
        all_npz_files.extend(npz_files)
    
    # 合并所有NPZ文件
    if is_train:
        final_output = os.path.join(output_dir, "train_dataset.npz")
    else:
        final_output = os.path.join(output_dir, "test_dataset.npz")
    
    merge_npz_files(all_npz_files, final_output, is_train_data=is_train)
    
    return final_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分批处理光谱数据并合并NPZ文件')
    parser.add_argument('--csv_files', nargs='+', required=True, help='输入CSV文件路径列表')
    parser.add_argument('--element', required=True, help='要处理的元素名称')
    parser.add_argument('--batch_size', type=int, default=10000, help='每批处理的数据条数')
    parser.add_argument('--output_dir', default="processed_data", help='输出目录')
    parser.add_argument('--is_train', action='store_true', help='是否为训练数据')
    
    args = parser.parse_args()
    
    final_file = process_and_merge(
        args.csv_files, 
        args.element, 
        args.batch_size, 
        args.output_dir, 
        args.is_train
    )
    
    print(f"处理完成! 最终文件: {final_file}")