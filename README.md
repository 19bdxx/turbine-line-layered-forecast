# 输电线分层预测系统 (Turbine Line Layered Forecast)

## 项目简介

本项目是一个**风电输电线功率分层预测系统**，针对某风电场的输电线（丙线）进行短期功率预测。系统采用**分层预测**策略：先对线路上的每台风机分别建立预测模型，再将各风机预测功率求和，并扣除输电线损耗，最终得到线路净输出功率预测值。同时与**全局预测**（直接对线路总功率建模）进行对比，评估两种策略的精度差异。

支持在同一管道下对比以下 **6 种模型**：

| 模型 | 类型 | 依赖 |
|------|------|------|
| `lightgbm` | 梯度提升树 | lightgbm |
| `xgboost` | 梯度提升树 | xgboost |
| `random_forest` | 随机森林 | scikit-learn |
| `ridge` | 岭回归（线性） | scikit-learn |
| `mlp` | 多层感知机 | scikit-learn |
| `lstm` | 长短期记忆网络 | torch |

---

## 项目结构

```
turbine-line-layered-forecast/
├── prepare_data.py   # 数据准备：清洗原始数据 + 联合重复段质量检查
├── forecast.py       # 预测管道：全局预测 + 单机预测 + 线损分析（支持 6 种模型）
└── compare_power.py  # 可视化分析：风机集合功率 vs 输电线实测功率
```

> **旧版文件**（`#0构建数据集.py`、`#1检查重复时间.py`、`全局预测.py`、`demo2.py`、
> `预测结果汇总.py`、`预测结果分析.py`、`输电线_风机集合功率对比.py`）已保留以供参考，
> 顶部有弃用说明，功能均已合并到上述三个文件中。

---

## 安装依赖

```bash
pip install pandas numpy scikit-learn lightgbm xgboost openpyxl matplotlib
# LSTM 模型额外需要 PyTorch
pip install torch
```

---

## 运行流程

### 步骤 1：数据准备

```bash
python prepare_data.py
```

- 读取原始 CSV，过滤重复时间戳，选取所需列，保存清洗后数据集
- 对清洗后数据执行各风机多字段联合重复段检测

### 步骤 2：多模型预测（核心）

```bash
python forecast.py
```

对 `MODELS` 列表中的每个模型 × 每个预测步长 `N`，自动完成：
1. **全局预测**：直接对输电线聚合功率建模
2. **单机预测**：对每台风机独立建模
3. **线损分析**：汇总单机预测，查表扣除线损，与全局预测对比

### 步骤 3：功率对比分析（可选）

```bash
python compare_power.py
```

对比风机功率求和与线路实测功率，生成误差分布统计及可视化图表。

---

## 配置说明

三个脚本顶部均有 **"配置"** 区域，可直接修改路径，或通过环境变量传入（避免修改代码）：

| 脚本 | 环境变量 | 说明 |
|------|---------|------|
| `prepare_data.py` | `RAW_DATA_FILE` | 原始数据路径 |
| `prepare_data.py` | `DUPLICATE_TS_FILE` | 重复时间戳文件路径 |
| `prepare_data.py` | `CLEAN_OUTPUT_FILE` | 清洗结果输出路径 |
| `forecast.py` | `DATA_FILE` | 清洗后数据路径 |
| `forecast.py` | `OUTPUT_ROOT` | 预测结果输出根目录 |
| `forecast.py` | `LINE_LOSS_FILE` | 线损查找表路径 |
| `compare_power.py` | `DATA_FILE` | 清洗后数据路径 |

在 `forecast.py` 中，可通过修改 `MODELS` 列表控制启用哪些模型：

```python
MODELS = [
    "lightgbm",
    "xgboost",
    "random_forest",
    "ridge",
    "mlp",
    "lstm",    # 需要 PyTorch
]
```

---

## 数据说明

### 输入数据字段

| 字段 | 说明 |
|------|------|
| `timestamp` | 时间戳（采样间隔 60 秒）|
| `LIMIT_POWER` | 线路限制功率（kW）|
| `ACTIVE_POWER_BING_PROCESS` | 丙线聚合有功功率（kW）|
| `ACTIVE_POWER_#153` ~ `ACTIVE_POWER_#158` | 各风机有功功率（kW）|
| `WINDSPEED_#153` ~ `WINDSPEED_#158` | 各风机风速（m/s）|
| `REACTIVE_POWER_#153` ~ `REACTIVE_POWER_#158` | 各风机无功功率（kvar）|
| `STATUS_#153` ~ `STATUS_#158` | 各风机运行状态 |

### 数据过滤规则

- 剔除 `LIMIT_POWER` 在 0–900 之间的数据（对应停机或限功率状态）
- 将负值功率截断为 0
- 剔除时间间隔不连续（> 60 秒）的数据段

---

## 模型说明

### 特征构造

采用**滑动窗口**方式构建时序特征：

- **回溯步长 M = 60**（即过去 60 个时间步，约 60 分钟）
- 每步特征：当前时刻的有功功率 + 风速，共 **2 × M = 120 维**
- LSTM 模型将特征重塑为 `(样本数, M, 2)` 的三维序列输入

### 预测步长

| 预测步长 N | 对应预测时长 |
|-----------|-------------|
| 60 步 | 约 1 小时 |
| 120 步 | 约 2 小时 |
| 180 步 | 约 3 小时 |

数据集按 **80/20** 划分训练集与测试集，评估指标为 RMSE、MAE、R²。

---

## 输出结果

```
<OUTPUT_ROOT>/
├── global/
│   ├── <model>/
│   │   └── predictions_global_M60_N{N}.csv   # 全局模型预测值
│   └── metrics_global_<model>.csv            # 全局模型指标汇总
├── turbine/
│   └── <model>/
│       ├── turbine_<id>/
│       │   └── predictions_turbine<id>_M60_N{N}.csv  # 单机预测值
│       ├── all_predictions_M60_N{N}.csv      # 所有风机预测值对齐汇总
│       └── metrics_turbine_<model>.csv        # 单机指标汇总
└── comparison/
    ├── comparison_<model>_M60_N{N}.csv        # 全局 vs 分层净功率 指标对比
    └── final_result_<model>_M60_N{N}.csv      # 含线损计算的完整结果
```

---

## 核心思路

```
原始数据
   │
   ▼ prepare_data.py
数据清洗（去重、过滤停机、对齐时间戳）
   │
   ▼ forecast.py（对每种模型重复以下流程）
   ├──► 全局预测（直接预测线路总功率）
   │
   └──► 单机预测（每台风机独立建模）
           │
           ▼
        各风机预测功率求和
           │
           ▼
        扣除输电线损耗（查表法）
           │
           ▼
        线路净功率预测
           │
           ▼
        与全局预测对比，评估分层策略精度
```

