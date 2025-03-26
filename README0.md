# 恒星光谱元素丰度预测系统

本项目使用深度学习方法从LAMOST光谱数据中预测恒星的元素丰度值（X_FE，其中X代表C、Mg、Ca等元素）。通过残差卷积神经网络模型对光谱数据进行分析，可以快速准确地估计这些元素的丰度值。

## 项目结构

```
.
├── config.py            # 配置文件
├── preprocessdata.py    # 光谱数据预处理
├── model.py            # 模型定义和训练函数
├── main.py             # 训练和测试主程序
├── evaluation.py       # 模型评估和可视化工具
├── predict.py          # 预测功能
├── batch_hyperopt.py   # 批量超参数优化
├── baseline_models.py  # 基线模型比较
├── model_analysis.py   # 模型分析工具（特征重要性和残差分析）
├── data_validator.py   # 数据验证工具
├── fits/               # 原始光谱FITS文件
├── processed_data/     # 处理后的数据
│   ├── cache/         # 缓存目录
│   │   ├── preprocessing/  # 预处理缓存
│   │   ├── train/     # 训练缓存
│   │   ├── evaluation/ # 评估缓存
│   │   ├── hyperopt/  # 超参数优化缓存
│   │   └── predict/   # 预测缓存
│   ├── progress/      # 进度保存目录
│   ├── validation/    # 数据验证结果
│   │   └── plots/     # 验证图表
│   ├── train_dataset.npz    # 训练集
│   ├── val_dataset.npz      # 验证集
│   ├── test_dataset.npz     # 测试集
│   ├── reference_dataset.npz # 参考数据集（训练+验证+测试）
│   └── prediction_dataset.npz # 预测数据集
├── models/             # 模型保存目录
├── logs/              # 日志目录
│   ├── train/        # 训练日志
│   ├── predict/      # 预测日志
│   └── validation/   # 验证日志
├── results/           # 结果目录
│   ├── preprocessing/             # 预处理批次结果（修改）
│   │   ├── C_FE/                  # 按元素分类
│   │   │   ├── batch_1/           # 每个批次的结果
│   │   │   │   ├── metrics.txt    # 批次统计信息 
│   │   │   │   ├── results.csv    # 详细结果数据
│   │   │   │   └── plots/         # 批次可视化
│   │   │   ├── summary.txt        # 处理总结
│   │   │   └── plots/             # 总体可视化
│   │   └── fits_processing/       # FITS文件处理结果
│   ├── predictions/  # 预测结果
│   ├── evaluation/   # 评估结果
│   ├── hyperopt/     # 超参数优化结果
│   ├── feature_importance/        # 特征重要性分析结果
│   │   ├── C_FE_batch_results/    # 按元素分类的批次结果
│   │   │   ├── batch_1_feature_importance.png    # 批次特征重要性图
│   │   │   ├── batch_1_explanation.txt           # 批次特征解释
│   │   │   ├── batch_tracking.csv                # 批次跟踪数据
│   │   │   ├── C_FE_batch_trends.png             # 趋势图
│   │   │   └── C_FE_batch_summary.txt            # 批次摘要
│   │   └── C_FE_feature_importance.png           # 总体特征重要性图
│   ├── residual_analysis/         # 残差分析结果
│   │   ├── C_FE_batch_results/    # 按元素分类的批次结果
│   │   │   ├── batch_1_residual_analysis.png     # 批次残差分析图
│   │   │   ├── batch_1_residual_report.txt       # 批次残差报告
│   │   │   ├── batch_1_predictions.csv           # 批次预测结果
│   │   │   ├── batch_tracking.csv                # 批次跟踪数据
│   │   │   ├── C_FE_batch_trends.png             # 趋势图
│   │   │   └── C_FE_batch_summary.txt            # 批次摘要
│   │   └── C_FE_residual_analysis.png            # 总体残差分析图
│   └── batch_tracking/            # 批次跟踪结果
├── plots/             # 图表目录
│   ├── preprocessing/ # 预处理过程图表（光谱处理过程可视化、数据分布统计等）
│   ├── training/     # 训练过程图表（损失曲线、学习率变化、指标趋势等）
│   ├── predictions/  # 预测结果图表（预测结果分布、误差分析等）
│   └── evaluation/   # 评估结果图表（性能指标对比、HR图等）
├── X_FE.csv           # 元素丰度参考数据
├── galah_X_FE.csv     # GALAH数据集元素丰度数据
└── LASP_X_FE.csv      # LASP数据集元素丰度数据
```

## 数据预处理

数据预处理由`preprocessdata.py`实现，主要包括以下步骤和功能：

### 主要处理流程
1. 读取原始LAMOST FITS光谱数据
2. 光谱去噪、波长校准和红移校正
3. 波长重采样（统一波长范围和采样点数）
4. 连续谱归一化
5. 二次去噪和最终归一化
6. 按7:1:2的比例划分训练集、验证集和测试集

### 最新改进
1. **双数据集处理机制**：支持参考数据集和预测数据集两种模式
   - 参考数据集：参与预处理和划分，用于训练、验证和测试（X_FE.csv等）
   - 预测数据集：仅参与预处理，用于预测和评估（galah_X_FE.csv等）

2. **增强的数据验证**：
   - 自动检测和过滤无效数据
   - 确保所有光谱长度一致（使用最常见长度）
   - 信噪比评估和异常值处理

3. **鲁棒的归一化算法**：
   - 自适应分段连续谱拟合（蓝端和红端使用不同策略）
   - 基于信噪比动态调整处理参数
   - 多级防护避免归一化失败
   - 离群值检测和处理机制

4. **自动数据可视化**：
   - 生成处理过程的详细可视化
   - 显示不同处理阶段的光谱变化
   - 标注主要吸收线

### 已知问题
1. **短波长（蓝端）区域处理挑战**：
   - 3800-5000Å区域信噪比通常较低
   - 包含多条强吸收线可能影响连续谱拟合
   - 部分光谱在此区域可能出现异常归一化结果

2. **低信噪比光谱限制**：
   - 极低信噪比（<3）的光谱可能无法可靠处理
   - 自动筛选机制会过滤掉质量最差的光谱

3. **长波长处理**：
   - 有些光谱在长波长端（>8500Å）存在残余特征
   - 可能是仪器效应或大气吸收的影响

预处理后的数据包含以下字段：
- `X`: 处理后的光谱数据数组
- `y`: 目标元素丰度值
- `elements`: 元素名称标识
- `filenames`: 原始FITS文件名

## 模型架构

使用SpectralResCNN模型，这是一个基于残差连接的深度卷积神经网络：
- 输入层：处理原始光谱
- 3组残差块：提取不同级别的特征
- 全连接层：进行预测

## 工作流程

### 1. 数据预处理

使用`preprocessdata.py`进行数据预处理：

```bash
# 处理参考数据集（用于训练）
python preprocessdata.py --reference_csv X_FE.csv \
                       --fits_dir fits \
                       --output_dir processed_data

# 处理预测数据集
python preprocessdata.py --prediction_csv galah_X_FE.csv \
                       --fits_dir fits \
                       --output_dir processed_data

# 同时处理参考和预测数据集
python preprocessdata.py --reference_csv X_FE.csv \
                       --prediction_csv galah_X_FE.csv \
                       --fits_dir fits \
                       --output_dir processed_data

# 使用自定义配置进行预处理
python preprocessdata.py --reference_csv X_FE.csv \
                       --fits_dir fits \
                       --output_dir processed_data \
                       --batch_size 32 \
                       --n_workers 4 \
                       --no_cache \
                       --denoise \
                       --normalize

# 清除缓存并重新处理
python preprocessdata.py --reference_csv X_FE.csv \
                       --clear_cache
```

预处理后的数据将保存在`processed_data`目录下：
- `train_dataset.npz`: 训练集
- `val_dataset.npz`: 验证集
- `test_dataset.npz`: 测试集
- `reference_dataset.npz`: 完整参考数据集
- `prediction_dataset.npz`: 预测数据集

处理日志将保存在`logs`目录下，处理进度信息将保存在`processed_data/progress`目录下。

### 2. 训练与超参数调优

使用`main.py`进行训练和超参数调优：

```bash
# 完整流程（训练、调优和测试）
python main.py --mode all --data_path processed_data/reference_dataset.npz

# 仅训练模型
python main.py --mode train --data_path processed_data/reference_dataset.npz

# 仅进行超参数调优
python main.py --mode tune --data_path processed_data/reference_dataset.npz

# 仅测试模型
python main.py --mode test --data_path processed_data/test_dataset.npz

# 训练特定元素的模型
python main.py --mode train --data_path processed_data/reference_dataset.npz --elements C_FE MG_FE

# 使用自定义配置进行训练
python main.py --mode train \
               --data_path processed_data/reference_dataset.npz \
               --batch_size 64 \
               --learning_rate 0.001 \
               --epochs 200 \
               --early_stopping 20
               
# 使用批量超参数优化
python main.py --mode tune \
               --data_path processed_data/reference_dataset.npz \
               --elements C_FE \
               --batch_size_hyperopt 1000 \
               --batches_per_round 2
               
# 查看分批处理结果
python main.py --mode show_results \
               --elements C_FE \
               --result_type training

# 执行模型分析并启用批处理
python main.py --mode analyze \
               --elements C_FE MG_FE \
               --analysis_type both \
               --analysis_batch_size 32 \
               --save_batch_results

# 查看模型分析批处理结果
python main.py --mode show_results \
               --elements C_FE \
               --result_type analysis
```

超参数调优过程：
1. **批量超参数优化**：使用`batch_hyperopt.py`进行批量数据处理
   - 可设置批量大小（`--batch_size_hyperopt`）和每轮批次数（`--batches_per_round`）
   - 支持中断和继续，自动保存中间结果
   - 生成两组最佳参数，用于比较和选择

2. **传统超参数优化**：
   - 使用网格搜索或贝叶斯优化方法
   - 对每组参数在验证集上评估性能
   - 选择最佳参数组合
   - 使用最佳参数重新训练模型

训练过程中的监控指标：
- 训练损失
- 验证损失
- 各项评估指标（MAE、RMSE、R²）
- 学习率变化
- GPU内存使用情况

#### 实时批次处理结果

现在，系统在以下过程中生成实时批次处理结果：

1. **训练过程**：
   - 每个Epoch后生成评估指标和散点图
   - 实时更新训练趋势图
   - 保存批次结果到`results/training_{element}_batch_results`目录

2. **评估过程**：
   - 每个批次后生成评估指标和可视化
   - 自动生成误差分布图和散点图
   - 保存批次结果到`results/evaluation_{element}_batch_results`目录

3. **预测过程**：
   - 每个批次后生成分布图和统计信息
   - 实时更新预测趋势
   - 保存批次结果到`results/prediction_{element}_batch_results`目录

使用`--mode show_results`命令查看任何处理阶段的批次结果：

```bash
# 查看训练批次结果
python main.py --mode show_results --elements C_FE --result_type training

# 查看评估批次结果
python main.py --mode show_results --elements C_FE --result_type evaluation

# 查看预测批次结果
python main.py --mode show_results --elements C_FE --result_type prediction
```

所有训练日志将保存在`logs/train`目录下，训练过程的图表将保存在相应的批次结果目录中。

### 3. 预测过程

使用`predict.py`进行预测：

```bash
# 使用训练好的模型进行预测
python predict.py --data_path processed_data/prediction_dataset.npz \
                 --output_dir results/predictions \
                 --elements C_FE MG_FE CA_FE
                 
# 使用集成模型进行预测
python predict.py --data_path processed_data/prediction_dataset.npz \
                 --ensemble_mode \
                 --output_dir results/predictions
```

### 4. 模型分析

使用`model_analysis.py`功能或通过`main.py`的`analyze`模式进行模型分析：

#### 特征重要性分析

分析光谱波长区域对预测的重要性，找出对元素丰度预测最重要的光谱特征：

```bash
# 执行特征重要性分析
python main.py --mode analyze \
               --elements C_FE \
               --analysis_type feature_importance \
               --analysis_batch_size 32 \
               --save_batch_results
```

特征重要性分析会生成以下结果：
- 光谱特征重要性图（突出显示重要波长区域）
- 特征解释文件（解释关键光谱特征的天文物理意义）
- 批次处理趋势和摘要（多批次模式下）

#### 残差分析

分析模型预测的残差分布，评估模型在不同元素丰度区间的表现：

```bash
# 执行残差分析
python main.py --mode analyze \
               --elements C_FE \
               --analysis_type residual_analysis \
               --analysis_batch_size 32 \
               --save_batch_results
```

残差分析会生成以下结果：
- 真实值vs预测值散点图
- 残差分布图（直方图和散点图）
- 残差QQ图（检查正态性）
- 性能指标报告（RMSE、MAE、R²等）
- 批次处理趋势和摘要（多批次模式下）

#### 完整模型分析

同时执行特征重要性分析和残差分析：

```bash
# 执行完整分析
python main.py --mode analyze \
               --elements C_FE MG_FE CA_FE \
               --analysis_type both \
               --analysis_batch_size 32 \
               --save_batch_results
```

#### 批次分析比较

比较不同批次的分析结果，查找趋势和模式：

```bash
# 显示特征重要性批次分析结果
python main.py --mode show_results \
               --elements C_FE \
               --result_type analysis \
               --analysis_type feature_importance

# 显示残差分析批次结果
python main.py --mode show_results \
               --elements C_FE \
               --result_type analysis \
               --analysis_type residual_analysis
```

批次分析结果包括：
1. **特征重要性趋势**：跟踪不同批次的顶级特征重要性变化
2. **残差指标趋势**：跟踪RMSE、MAE、R²等指标随批次变化
3. **批次摘要报告**：汇总每个批次的关键发现
4. **批次间比较**：比较不同批次的结果差异

### 5. 基线模型比较

使用`baseline_models.py`进行基线模型比较：

```bash
# 使用XGBoost模型进行基线比较
python baseline_models.py --element C_FE --model xgboost

# 使用LightGBM模型进行基线比较
python baseline_models.py --element C_FE --model lightgbm

# 对比两种模型性能
python baseline_models.py --element C_FE --model both

# 使用批处理提高性能
python baseline_models.py --element C_FE --model both \
                         --batch_size 1000 --batches_per_round 2

# 查看分批处理结果
python baseline_models.py --element C_FE --model both --show_batch_results
```

基线模型比较结果将保存在`results/baseline`目录下，分批处理的结果将保存在`results/baseline_{element}_{model}_batch_results`目录下。

## 系统特点

1. **分批处理和实时结果生成**：
   - 所有处理阶段（训练、评估、预测）均支持分批处理
   - 每个批次处理后立即生成可视化和评估结果
   - 自动保存中间结果，支持随时中断和继续处理

2. **批量超参数优化**：
   - 支持大规模数据的超参数优化
   - 自动缓存和恢复中间状态
   - 生成两组最佳参数供比较选择

3. **多种可视化和评估工具**：
   - 训练过程趋势图、散点图
   - 评估性能指标和误差分布图
   - 预测结果分布和统计信息

4. **模型比较和集成**：
   - 基线模型和深度学习模型比较
   - 模型集成提高预测性能
   - 批次性能比较和选择

5. **高效的缓存和进度管理**：
   - 智能缓存机制减少重复计算
   - 进度跟踪和可视化
   - 支持低内存环境的处理策略

## 注意事项

1. **数据质量**：使用预处理系统时，留意处理日志中的警告信息。部分低质量光谱可能会被过滤掉，最终处理的数据量可能小于原始数据量。

2. **光谱可视化**：检查处理后的光谱可视化结果，确保归一化合理。特别注意蓝端和红端区域是否存在异常。

3. **数据路径**：确保在`config.py`中正确设置数据文件路径。在cursor环境中，processed_data文件夹存放在根目录bishecomputer下。

4. **FITS文件结构**：程序默认从FITS文件的第一个HDU读取数据，如果找不到数据，会尝试从第二个HDU读取。

5. **CSV文件匹配**：通过spec列匹配光谱和对应的teff、logg值，确保CSV文件中的spec列与光谱数据正确对应。

6. **MIST等时线**：评估可视化中包含了MIST恒星等时线，用于对比预测结果与恒星演化理论。

7. **运行环境**：代码设计为同时支持本地环境和Colab环境运行。在Colab中，可能需要调整文件路径。

## 依赖库

以下是项目所需的主要依赖库：

```
numpy
pandas
matplotlib
torch
scikit-learn
astropy
scipy
seaborn
psutil
tqdm
joblib
```

可以通过以下命令安装：

```bash
pip install numpy pandas matplotlib torch scikit-learn astropy scipy seaborn psutil tqdm joblib
```

## 系统架构

### 数据预处理
- 读取原始LAMOST FITS光谱数据
- 去噪、波长定标和红移校正
- 重采样和归一化
- 引入双数据集处理机制，分别处理参考数据集和预测数据集
- 参考数据集按7:1:2比例划分为训练集、验证集和测试集

### 缓存机制
系统实现了高效的缓存机制，主要包括以下特点：
1. 检查点保存
   - 每处理1000个样本自动保存一次检查点
   - 支持断点续传，可以从上次中断处继续处理
   - 检查点包含完整的处理状态和验证指标

2. 数据验证
   - 自动验证缓存数据的完整性
   - 检查数据格式和必要字段
   - 监控内存使用情况

3. 缓存目录结构
   - preprocessing/: 预处理缓存
   - training/: 训练过程缓存
   - evaluation/: 评估结果缓存
   - prediction/: 预测结果缓存

4. 性能优化
   - 避免重复计算
   - 支持增量更新
   - 自动清理过期缓存

### 模型架构
- 使用CNN残差网络进行光谱特征提取
- 支持多元素丰度同时预测
- 集成学习策略，提高预测稳定性
- MC-Dropout用于不确定性估计

### 评估系统
- 支持多元素丰度评估
- 提供详细的评估指标
- 可视化预测结果
- 按恒星类型（矮星/巨星）分组评估

## 使用说明

### 环境配置
1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置参数：
- 在 `config.py` 中设置相关参数
- 包括数据路径、模型参数、训练参数等

### 运行流程
1. 数据预处理：
```bash
python preprocessdata.py
```

2. 模型训练：
```bash
python main.py --train
```

3. 预测评估：
```bash
python main.py --predict
```

### 注意事项
1. 首次运行时会自动创建缓存目录
2. 程序中断后可以从检查点继续运行
3. 定期检查日志文件了解处理进度
4. 确保有足够的磁盘空间存储缓存文件

## 性能优化
1. 使用缓存机制避免重复计算
2. 支持断点续传，提高处理效率
3. 自动验证数据完整性
4. 内存使用优化

## 未来改进
1. 优化缓存清理策略
2. 添加分布式处理支持
3. 改进数据验证机制
4. 优化内存管理

## 最近更新日志

### 2023年11月更新

#### 增加了模型分析批处理功能
- 添加了`model_analysis.py`模块，支持特征重要性和残差分析
- 实现了批处理跟踪功能，可以比较和可视化不同批次的分析结果
- 增强了特征解释功能，可以识别和解释重要光谱特征
- 在`main.py`中添加了`analyze`模式，支持批处理分析和结果显示
- 配置文件`config.py`中添加了分析参数控制

#### 改进的批处理功能
- 在多个模块中统一了批处理机制
- 实现了批次结果的自动保存和加载
- 添加了批次间比较和趋势分析功能
- 增强了批次可视化和报告生成

#### 命令行接口更新
- 添加了`--analysis_type`、`--analysis_batch_size`和`--save_batch_results`参数
- 在`show_results`命令中添加了`analysis`结果类型
- 添加了分析批次结果显示功能

未来工作计划:
- 进一步优化批处理性能
- 添加更多特征解释和可视化选项
- 扩展到更多元素的批处理分析支持 