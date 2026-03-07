"""
风电输电线功率预测管道 (Wind Power Transmission Line Forecast Pipeline)

支持的模型 (Supported models):
    lightgbm, xgboost, random_forest, ridge, mlp, lstm

设备支持 (Device Support):
    LSTM 模型自动优先使用 GPU（CUDA），无 GPU 时回退到 CPU。
    其他模型（lightgbm / xgboost / random_forest / ridge / mlp）使用 CPU 多核加速。
    启动时会打印检测到的设备信息。

运行方式：
    python forecast.py

流程：
    对每个模型 × 每个预测步长 N：
      1. 全局预测：直接对输电线聚合功率建模
      2. 单机预测：对每台风机独立建模
      3. 线损分析：汇总单机预测，查表扣除线损，与全局预测对比

注意：运行前请将下方"配置"部分的路径修改为实际路径，
      或通过对应环境变量指定。LSTM 模型需要安装 PyTorch（pip install torch）。
"""

import csv
import gc
import os
import random
import time
import traceback
import warnings
from datetime import datetime
from typing import Callable, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch as _torch

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# 全局配置 (Global Configuration)
# ============================================================
DATA_FILE = os.environ.get(
    "DATA_FILE",
    r"G:\WindPowerForecast\输电线丙分风机预测\原始数据"
    r"\峡阳峡沙点位数据20240315-20241201_峡阳B_V4_去重.csv",
)
OUTPUT_ROOT = os.environ.get(
    "OUTPUT_ROOT",
    r"G:\WindPowerForecast\输电线丙分风机预测\多模型预测结果",
)
LINE_LOSS_FILE = os.environ.get(
    "LINE_LOSS_FILE",
    r"G:\WindPowerForecast\输电线丙分风机预测\输电线损分析\丙_功率区间均值对比.csv",
)

TURBINE_IDS      = list(range(153, 159))   # 参与建模的风机编号列表
M                = 60                       # 回溯步长（时间步数）
N_LIST           = [60, 120, 180]           # 预测步长列表（60步≈1h，120步≈2h，180步≈3h）
TIME_GAP_SECONDS = 60                       # 数据采样间隔（秒）
RANDOM_SEED      = 42

# 要测试的模型，注释掉不需要的模型即可
MODELS = [
    "lightgbm",
    "xgboost",
    "random_forest",
    "ridge",
    "mlp",
    "lstm",
]

# ============================================================
# 工具函数 (Utilities)
# ============================================================

def ensure_dir(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def check_time_continuity(timestamps: pd.Series, max_gap_seconds: int) -> bool:
    diffs = timestamps.diff().dropna().dt.total_seconds()
    return bool((diffs <= max_gap_seconds).all())


# 模块级设备缓存：None 表示尚未检测，首次调用 _get_torch_device() 时初始化
_TORCH_DEVICE: "Optional[_torch.device]" = None


def _get_torch_device() -> "torch.device":
    """
    懒加载并缓存 PyTorch 设备（优先 GPU，无 GPU 回退 CPU）。
    首次调用时打印检测到的设备信息，后续调用直接返回缓存结果。

    Raises:
        ImportError: 若 PyTorch 未安装。
    """
    global _TORCH_DEVICE
    if _TORCH_DEVICE is not None:
        return _TORCH_DEVICE  # type: ignore[return-value]

    try:
        import torch
    except ImportError:
        raise ImportError("LSTM 模型需要 PyTorch，请运行: pip install torch")

    if torch.cuda.is_available():
        _TORCH_DEVICE = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(
            f"🖥️  GPU 已启用: {gpu_name}（共 {gpu_count} 块，显存 {total_mem_gb:.1f} GB）"
        )
    else:
        _TORCH_DEVICE = torch.device("cpu")
        print("🖥️  未检测到可用 GPU，LSTM 将使用 CPU 训练")

    return _TORCH_DEVICE  # type: ignore[return-value]


# ============================================================
# 数据集构建 (Dataset Builders)
# ============================================================

def build_global_dataset(
    df: pd.DataFrame, M: int, N: int, time_gap_seconds: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构建全局预测数据集。
    特征：过去 M 步的（均值风速，线路有功功率）交替排列，共 2M 维。
    目标：M 步结束后第 N 步的线路有功功率。
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    timestamps = df["timestamp"].reset_index(drop=True)

    ws_cols = [c for c in df.columns if c.startswith("WINDSPEED_#")]
    ws_mean = df[ws_cols].to_numpy().mean(axis=1)
    ap = df["ACTIVE_POWER_BING_PROCESS"].to_numpy()

    # 预建时间戳哈希索引，O(1) 查找目标时间行，替代逐窗口全表过滤
    ts_to_idx: dict = {ts: idx for idx, ts in enumerate(timestamps)}
    # 末尾 N 步的窗口起点必然找不到目标，提前缩小上界避免无效循环
    upper = max(0, len(df) - M - N + 1)

    features_list, targets_list, ts_list = [], [], []
    skipped = 0

    for i in range(upper):
        if not check_time_continuity(timestamps[i : i + M], time_gap_seconds):
            skipped += 1
            continue
        target_time = timestamps[i + M - 1] + pd.Timedelta(seconds=N * time_gap_seconds)
        target_idx = ts_to_idx.get(target_time)
        if target_idx is None:
            skipped += 1
            continue
        flat = []
        for step in range(M):
            flat.extend([ws_mean[i + step], ap[i + step]])
        features_list.append(flat)
        targets_list.append(float(ap[target_idx]))
        ts_list.append(target_time)

    print(f"  ⏳ 全局数据集：跳过 {skipped} 个无效窗口，有效样本 {len(features_list)} 个")
    return (
        np.array(features_list, dtype=np.float32),
        np.array(targets_list, dtype=np.float32),
        np.array(ts_list),
    )


def build_turbine_dataset(
    df: pd.DataFrame, turbine_id: int, M: int, N: int, time_gap_seconds: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构建单台风机预测数据集。
    特征：过去 M 步的（有功功率，风速）交替排列，共 2M 维。
    目标：M 步结束后第 N 步的该风机有功功率。
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    timestamps = df["timestamp"].reset_index(drop=True)

    ap_col = f"ACTIVE_POWER_#{turbine_id}"
    ws_col = f"WINDSPEED_#{turbine_id}"
    df[ap_col] = df[ap_col].clip(lower=0)

    ap = df[ap_col].to_numpy()
    ws = df[ws_col].to_numpy()

    # 预建时间戳哈希索引，O(1) 查找目标时间行，替代逐窗口全表过滤
    ts_to_idx: dict = {ts: idx for idx, ts in enumerate(timestamps)}
    # 末尾 N 步的窗口起点必然找不到目标，提前缩小上界避免无效循环
    upper = max(0, len(df) - M - N + 1)

    features_list, targets_list, ts_list = [], [], []
    skipped = 0

    for i in range(upper):
        if not check_time_continuity(timestamps[i : i + M], time_gap_seconds):
            skipped += 1
            continue
        target_time = timestamps[i + M - 1] + pd.Timedelta(seconds=N * time_gap_seconds)
        target_idx = ts_to_idx.get(target_time)
        if target_idx is None:
            skipped += 1
            continue
        flat = []
        for step in range(M):
            flat.extend([ap[i + step], ws[i + step]])
        features_list.append(flat)
        targets_list.append(float(ap[target_idx]))
        ts_list.append(target_time)

    print(f"  ⏳ 风机 {turbine_id}：跳过 {skipped} 个无效窗口，有效样本 {len(features_list)} 个")
    return (
        np.array(features_list, dtype=np.float32),
        np.array(targets_list, dtype=np.float32),
        np.array(ts_list),
    )


# ============================================================
# 模型工厂 (Model Factory)
# ============================================================

def _train_lightgbm(
    X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> Tuple[object, Callable]:
    import lightgbm as lgb

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.02,
        "num_leaves": 127,
        "max_depth": 12,
        "min_data_in_leaf": 5,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": RANDOM_SEED,
    }
    tr_ds = lgb.Dataset(X_tr, label=y_tr)
    val_ds = lgb.Dataset(X_val, label=y_val)
    model = lgb.train(
        params,
        tr_ds,
        num_boost_round=2000,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(200)],
    )

    def predict(X: np.ndarray) -> np.ndarray:
        return model.predict(X, num_iteration=model.best_iteration)

    return model, predict


def _train_xgboost(
    X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> Tuple[object, Callable]:
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=30,
        random_state=RANDOM_SEED,
        verbosity=0,
        eval_metric="rmse",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model, model.predict


def _train_random_forest(
    X_tr: np.ndarray, y_tr: np.ndarray
) -> Tuple[object, Callable]:
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    return model, model.predict


def _train_ridge(
    X_tr: np.ndarray, y_tr: np.ndarray
) -> Tuple[object, Callable]:
    from sklearn.linear_model import Ridge

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_tr)
    model = Ridge(alpha=1.0)
    model.fit(X_s, y_tr)

    def predict(X: np.ndarray) -> np.ndarray:
        return model.predict(scaler.transform(X))

    return (model, scaler), predict


def _train_mlp(
    X_tr: np.ndarray, y_tr: np.ndarray
) -> Tuple[object, Callable]:
    from sklearn.neural_network import MLPRegressor

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_tr)
    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=500,
        early_stopping=True,       # 内部从训练集划分验证集
        validation_fraction=0.1,
        shuffle=False,             # 保持时间序列顺序，取末尾 10% 作验证
        n_iter_no_change=20,
        random_state=RANDOM_SEED,
        verbose=False,
        learning_rate_init=0.001,
    )
    model.fit(X_s, y_tr)

    def predict(X: np.ndarray) -> np.ndarray:
        return model.predict(scaler.transform(X))

    return (model, scaler), predict


def _train_lstm(
    X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, M: int
) -> Tuple[object, Callable]:
    """
    使用 PyTorch 训练双层 LSTM 模型。
    输入特征被重塑为 (samples, M, 2) 的时序格式，训练前做 StandardScaler 归一化。
    自动优先使用 GPU（CUDA），无 GPU 时回退到 CPU。
    需要安装 PyTorch：pip install torch
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        raise ImportError("LSTM 模型需要 PyTorch，请运行: pip install torch")

    # 使用全局缓存的设备（_get_torch_device 已在启动时调用过，这里直接取缓存）
    device = _get_torch_device()
    torch.manual_seed(RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(RANDOM_SEED)  # GPU 随机种子，确保可复现
    input_size = 2  # 每个时间步有 2 个特征

    # 归一化：仅在训练集拟合，再应用到验证集
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    def to_tensor(X: np.ndarray) -> "torch.Tensor":
        return torch.tensor(X.reshape(-1, M, input_size), dtype=torch.float32).to(device)

    class LSTMNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=64,
                num_layers=2,
                batch_first=True,
                dropout=0.2,
            )
            self.fc = nn.Linear(64, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    model = LSTMNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X_tr_t = to_tensor(X_tr)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)
    X_val_t = to_tensor(X_val)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=256, shuffle=True)

    best_val_loss = float("inf")
    patience = 20   # 连续 20 个 epoch 验证损失不降低则停止训练
    no_improve = 0
    best_state = None

    for epoch in range(100):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"    LSTM 早停于 epoch {epoch + 1}，最优验证损失 {best_val_loss:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def predict(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(to_tensor(scaler.transform(X))).cpu().numpy()

    return (model, scaler), predict


def train_model(
    model_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    M: int,
) -> Tuple[object, Callable]:
    """
    训练指定模型，返回 (model_obj, predict_fn)。
    predict_fn 接受 2-D numpy 数组，返回 1-D 预测值数组。
    """
    if model_name == "lightgbm":
        return _train_lightgbm(X_tr, y_tr, X_val, y_val)
    if model_name == "xgboost":
        return _train_xgboost(X_tr, y_tr, X_val, y_val)
    if model_name == "random_forest":
        return _train_random_forest(X_tr, y_tr)
    if model_name == "ridge":
        return _train_ridge(X_tr, y_tr)
    if model_name == "mlp":
        return _train_mlp(X_tr, y_tr)
    if model_name == "lstm":
        return _train_lstm(X_tr, y_tr, X_val, y_val, M)
    raise ValueError(
        f"不支持的模型: {model_name}。"
        f"可选: lightgbm, xgboost, random_forest, ridge, mlp, lstm"
    )


# ============================================================
# 评估与保存 (Evaluation & IO)
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mean_abs_true = float(np.mean(np.abs(y_true)))
    nrmse = rmse / mean_abs_true if mean_abs_true > 0 else float("nan")
    return {
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "NRMSE": nrmse,  # 归一化均方根误差（按真值绝对值均值归一化）
    }


def save_metrics(metrics: dict, filepath: str) -> None:
    ensure_dir(os.path.dirname(filepath) or ".")
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


def save_predictions(
    timestamps: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, filepath: str
) -> None:
    ensure_dir(os.path.dirname(filepath) or ".")
    pd.DataFrame(
        {"timestamp": timestamps, "true_value": y_true, "predicted_value": y_pred}
    ).to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"  📝 预测结果已保存: {filepath}")


# ============================================================
# 全局预测 (Global Forecast)
# ============================================================

def run_global_forecast(
    df: pd.DataFrame, model_name: str, M: int, N: int, output_root: str, exp_id: str
) -> None:
    """对输电线聚合功率直接建模，保存指标与预测结果。"""
    print(f"\n{'=' * 60}")
    print(f"🌐 全局预测 | 模型: {model_name} | M={M} | N={N}")
    print(f"{'=' * 60}")

    X, y, timestamps = build_global_dataset(df, M, N, TIME_GAP_SECONDS)
    if len(X) == 0:
        print("  ⚠️ 没有有效数据，跳过。")
        return

    split = int(len(X) * 0.8)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]
    ts_val = timestamps[split:]

    t0 = time.time()
    _, predict_fn = train_model(model_name, X_tr, y_tr, X_val, y_val, M)
    elapsed = time.time() - t0

    y_pred = np.clip(predict_fn(X_val), 0, None)
    m = compute_metrics(y_val, y_pred)
    print(
        f"  ✅ RMSE={m['RMSE']:.2f} | MAE={m['MAE']:.2f} | R²={m['R2']:.4f}"
        f" | NRMSE={m['NRMSE']:.4f} | 耗时 {elapsed:.1f}s"
    )

    model_dir = os.path.join(output_root, "global", model_name)

    save_metrics(
        {"model": model_name, "mode": "global", "M": M, "N": N, "experiment_id": exp_id, **m},
        os.path.join(output_root, "global", f"metrics_global_{model_name}.csv"),
    )
    save_predictions(
        ts_val,
        y_val,
        y_pred,
        os.path.join(model_dir, f"predictions_global_M{M}_N{N}.csv"),
    )
    gc.collect()


# ============================================================
# 单机预测 (Per-Turbine Forecast)
# ============================================================

def run_turbine_forecast(
    df: pd.DataFrame, model_name: str, M: int, N: int, output_root: str, exp_id: str
) -> Optional[pd.DataFrame]:
    """
    对所有风机逐一训练，保存结果，返回按时间戳对齐的汇总 DataFrame。
    如果所有风机均无有效数据，返回 None。
    """
    print(f"\n{'=' * 60}")
    print(f"🔩 单机预测 | 模型: {model_name} | M={M} | N={N}")
    print(f"{'=' * 60}")

    all_pred_dfs = []

    for tid in TURBINE_IDS:
        print(f"\n  🔄 风机 {tid}")
        X, y, timestamps = build_turbine_dataset(df, tid, M, N, TIME_GAP_SECONDS)

        if len(X) == 0:
            print(f"  ⚠️ 风机 {tid} 无有效数据，跳过。")
            continue

        split = int(len(X) * 0.8)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]
        ts_val = timestamps[split:]

        t0 = time.time()
        _, predict_fn = train_model(model_name, X_tr, y_tr, X_val, y_val, M)
        elapsed = time.time() - t0

        y_pred = np.clip(predict_fn(X_val), 0, None)
        m = compute_metrics(y_val, y_pred)
        print(
            f"  ✅ 风机 {tid} RMSE={m['RMSE']:.2f} | MAE={m['MAE']:.2f}"
            f" | R²={m['R2']:.4f} | NRMSE={m['NRMSE']:.4f} | 耗时 {elapsed:.1f}s"
        )

        turbine_dir = os.path.join(output_root, "turbine", model_name, f"turbine_{tid}")

        save_metrics(
            {
                "model": model_name,
                "mode": "turbine",
                "turbine_id": tid,
                "M": M,
                "N": N,
                "experiment_id": exp_id,
                **m,
            },
            os.path.join(output_root, "turbine", model_name, f"metrics_turbine_{model_name}.csv"),
        )
        save_predictions(
            ts_val,
            y_val,
            y_pred,
            os.path.join(turbine_dir, f"predictions_turbine{tid}_M{M}_N{N}.csv"),
        )

        all_pred_dfs.append(
            pd.DataFrame(
                {
                    "timestamp": ts_val,
                    f"ACTIVE_POWER_#{tid}": y_val,
                    f"PREDICTED_ACTIVE_POWER_#{tid}": y_pred,
                }
            )
        )

        del X, y, timestamps
        gc.collect()

    if not all_pred_dfs:
        return None

    merged = all_pred_dfs[0]
    for df_p in all_pred_dfs[1:]:
        merged = pd.merge(merged, df_p, on="timestamp", how="outer")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    summary_path = os.path.join(
        output_root, "turbine", model_name, f"all_predictions_M{M}_N{N}.csv"
    )
    ensure_dir(os.path.dirname(summary_path))
    merged.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n  📊 汇总预测已保存: {summary_path}")
    return merged


# ============================================================
# 线损分析与对比 (Line Loss & Comparison)
# ============================================================

def load_line_loss(line_loss_file: str) -> pd.DataFrame:
    """读取线损查找表，解析功率区间字符串为数值列。"""
    df_ll = pd.read_csv(line_loss_file)
    df_ll[["min_power", "max_power"]] = (
        df_ll["功率区间"].str.extract(r"\((.*), (.*)\]").astype(float)
    )
    df_ll["line_loss"] = df_ll["筛选后_平均误差"]
    return df_ll


def _get_line_loss(power: float, df_ll: pd.DataFrame) -> float:
    mask = (df_ll["min_power"] < power) & (power <= df_ll["max_power"])
    rows = df_ll[mask]
    if rows.empty:
        warnings.warn(
            f"功率值 {power:.4f} 超出线损查找表区间范围，默认线损返回 0.0",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.0
    return float(rows.iloc[0]["line_loss"])


def apply_line_loss_and_compare(
    df_turbine: pd.DataFrame,
    df_global: pd.DataFrame,
    df_ll: pd.DataFrame,
    model_name: str,
    M: int,
    N: int,
    output_root: str,
) -> None:
    """
    合并单机预测汇总与全局预测，计算线损净功率，
    与全局预测对比并保存评估指标和完整结果。
    """
    print(f"\n📊 线损分析 | 模型: {model_name} | M={M} | N={N}")

    df = pd.merge(df_turbine, df_global, on="timestamp", how="inner").sort_values("timestamp")

    df["ACTIVE_POWER_BING_PROCESS"] = df["ACTIVE_POWER_BING_PROCESS"].clip(lower=0)
    df["PREDICTED_ACTIVE_POWER_BING_PROCESS"] = df[
        "PREDICTED_ACTIVE_POWER_BING_PROCESS"
    ].clip(lower=0)

    pred_cols = [c for c in df.columns if c.startswith("PREDICTED_ACTIVE_POWER_#")]
    df[pred_cols] = df[pred_cols].clip(lower=0)

    df["SUM_PREDICTED"] = df[pred_cols].sum(axis=1) / 1000  # 与线路测量功率列保持量纲一致
    df["LINE_LOSS"] = df["SUM_PREDICTED"].apply(lambda p: _get_line_loss(p, df_ll))
    df["NET_PREDICTED"] = (df["SUM_PREDICTED"] - df["LINE_LOSS"]).clip(lower=0)

    y_true = df["ACTIVE_POWER_BING_PROCESS"].to_numpy()
    m_global = compute_metrics(y_true, df["PREDICTED_ACTIVE_POWER_BING_PROCESS"].to_numpy())
    m_net = compute_metrics(y_true, df["NET_PREDICTED"].to_numpy())

    print(
        f"  全局模型    → RMSE={m_global['RMSE']:.4f} | MAE={m_global['MAE']:.4f}"
        f" | R²={m_global['R2']:.4f} | NRMSE={m_global['NRMSE']:.4f}"
    )
    print(
        f"  分层净功率  → RMSE={m_net['RMSE']:.4f} | MAE={m_net['MAE']:.4f}"
        f" | R²={m_net['R2']:.4f} | NRMSE={m_net['NRMSE']:.4f}"
    )

    compare_dir = os.path.join(output_root, "comparison")
    ensure_dir(compare_dir)

    compare_csv = os.path.join(compare_dir, f"comparison_{model_name}_M{M}_N{N}.csv")
    with open(compare_csv, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["label"] + list(m_global.keys()))
        writer.writeheader()
        writer.writerow({"label": "全局模型预测", **m_global})
        writer.writerow({"label": "分层净功率（扣线损）", **m_net})
    print(f"  ✅ 对比指标已保存: {compare_csv}")

    result_csv = os.path.join(compare_dir, f"final_result_{model_name}_M{M}_N{N}.csv")
    df.to_csv(result_csv, index=False, encoding="utf-8-sig")
    print(f"  ✅ 完整结果已保存: {result_csv}")


# ============================================================
# 主程序 (Main)
# ============================================================

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # 提前检测 PyTorch 设备（GPU 优先），打印一次设备信息
    # 若未安装 PyTorch 则仅跳过 LSTM，其他模型不受影响
    if "lstm" in MODELS:
        try:
            _get_torch_device()
        except ImportError as e:
            print(f"⚠️  {e}（将跳过 lstm 模型）")

    ensure_dir(OUTPUT_ROOT)

    # 加载并过滤数据
    print(f"📂 加载数据: {DATA_FILE}")
    df_raw = pd.read_csv(DATA_FILE)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="coerce")
    df_raw = df_raw.dropna(subset=["timestamp"])
    df_raw = df_raw[(df_raw["LIMIT_POWER"] == 0) | (df_raw["LIMIT_POWER"] >= 900)]
    print(f"✅ 数据加载完成，共 {len(df_raw)} 行")

    # 加载线损表（文件不存在时跳过线损分析）
    df_ll = None
    if os.path.exists(LINE_LOSS_FILE):
        df_ll = load_line_loss(LINE_LOSS_FILE)
        print(f"✅ 线损表加载完成，共 {len(df_ll)} 个区间")
    else:
        print(f"⚠️  线损文件不存在，将跳过线损分析: {LINE_LOSS_FILE}")

    # 本次实验 ID，贯穿所有模型与步长的输出，便于追溯同一批结果
    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"🆔 实验 ID: {exp_id}")

    for model_name in MODELS:
        print(f"\n{'#' * 60}")
        print(f"# 模型: {model_name}")
        print(f"{'#' * 60}")

        for N in N_LIST:
            try:
                # 1. 全局预测（dataset builder 内部通过 sort_values 生成局部副本，无需 .copy()）
                run_global_forecast(df_raw, model_name, M, N, OUTPUT_ROOT, exp_id)

                # 2. 单机预测
                df_turbine_merged = run_turbine_forecast(df_raw, model_name, M, N, OUTPUT_ROOT, exp_id)

                # 3. 线损分析与对比
                if df_ll is not None and df_turbine_merged is not None:
                    global_pred_file = os.path.join(
                        OUTPUT_ROOT, "global", model_name,
                        f"predictions_global_M{M}_N{N}.csv",
                    )
                    if os.path.exists(global_pred_file):
                        df_global_preds = pd.read_csv(global_pred_file, parse_dates=["timestamp"])
                        df_global_preds = df_global_preds.rename(
                            columns={
                                "true_value": "ACTIVE_POWER_BING_PROCESS",
                                "predicted_value": "PREDICTED_ACTIVE_POWER_BING_PROCESS",
                            }
                        )
                        apply_line_loss_and_compare(
                            df_turbine_merged, df_global_preds, df_ll,
                            model_name, M, N, OUTPUT_ROOT,
                        )

            except Exception as e:
                print(f"\n❌ 模型 {model_name} N={N} 出错: {e}")
                traceback.print_exc()

    print(f"\n🎉 全部完成！结果保存在: {OUTPUT_ROOT}")
