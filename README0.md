# 恒星光谱元素丰度预测系统

本项目使用深度学习方法从LAMOST光谱数据中预测恒星的元素丰度值（C_FE、MG_FE、CA_FE）。通过残差卷积神经网络模型对光谱数据进行分析，可以快速准确地估计这些元素的丰度值。

## 项目结构

```
.
├── config.py            # 配置文件
├── preprocessdata.py    # 光谱数据预处理
├── model.py             # 模型定义和训练函数
├── main.py              # 训练和测试主程序
├── evaluation.py        # 模型评估和可视化工具
├── predict.py           # 单独的预测功能
├── fits/                # 原始光谱FITS文件
├── processed_data/      # 处理后的数据
│   ├── cache/           # 处理过程中的临时缓存文件
│   ├── progress/        # 处理进度信息
│   ├── train_dataset.npz  # 训练集
│   ├── val_dataset.npz    # 验证集
│   ├── test_dataset.npz   # 测试集
│   └── processed_data.npz # 全部处理后的数据
├── prediction_data/     # 预测数据集处理结果
├── C_FE.csv             # C元素丰度参考数据
├── MG_FE.csv            # Mg元素丰度参考数据
├── CA_FE.csv            # Ca元素丰度参考数据
├── models/              # 保存的模型
├── results/             # 评估结果
└── plots/               # 图表输出
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
   - 参考数据集：参与预处理、划分和交叉验证（CA_FE.csv等）
   - 预测数据集：仅用于预测和评估（galah_CA_FE.csv等）

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
# 在Google Colab上运行（会提示上传数据）
!python preprocessdata.py

# 指定参考数据集
!python preprocessdata.py --reference CA_FE.csv

# 同时指定参考和预测数据集
!python preprocessdata.py --reference CA_FE.csv --prediction galah_CA_FE.csv

# 不使用缓存，重新处理所有数据
!python preprocessdata.py --reference CA_FE.csv --no_resume
```

### 2. 训练与超参数调优

使用`main.py`进行训练和超参数调优：

```bash
# 完整流程（训练、调优和测试）
python main.py --mode all

# 仅训练模型
python main.py --mode train

# 仅进行超参数调优
python main.py --mode tune

# 仅测试模型
python main.py --mode test

# 指定元素
python main.py --mode all --elements C_FE MG_FE
```

超参数调优过程：
1. 尝试不同的学习率、批次大小和权重衰减组合
2. 在验证集上评估每组参数的性能
3. 选择验证损失最低的参数组合
4. 使用最佳参数重新训练模型

### 3. 模型评估与可视化

使用`evaluation.py`进行模型评估和可视化：

```bash
# 评估所有元素并生成所有图表
python evaluation.py

# 仅生成预测vs真实值对比图
python evaluation.py --plot_type compare

# 仅生成评估指标对比图
python evaluation.py --plot_type metrics

# 指定元素
python evaluation.py --elements C_FE MG_FE
```

评估指标包括：
- RMSE (Root Mean Square Error): 预测误差的平方根
- MAE (Mean Absolute Error): 平均绝对误差
- MSE (Mean Squared Error): 均方误差
- R² (R-squared): 决定系数
- DEX: 预测精度，即预测值与真实值差的标准差

可视化图表：
1. 预测值vs真实值对比图：横坐标为teff，纵坐标为logg，颜色表示元素丰度
2. 评估指标对比折线图：对比三种元素的性能指标

### 4. 预测新数据

使用`predict.py`预测新的LAMOST光谱数据：

```bash
# 预测光谱元素丰度
python predict.py --input /path/to/lamost_spectra.fits --output predictions.csv

# 绘制预测结果分布图
python predict.py --input /path/to/lamost_spectra.fits --output predictions.csv --plot

# 指定元素
python predict.py --input /path/to/lamost_spectra.fits --elements C_FE MG_FE
```

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