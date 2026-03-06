# ⚠️ 此文件已合并到 forecast.py，建议使用新文件（支持多模型）。
# This file has been consolidated into forecast.py (multi-model support).

import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import csv
import random
import gc
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import time

# 固定随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def check_time_continuity(timestamps, max_gap_seconds):
    diffs = timestamps.diff().dropna().dt.total_seconds()
    return all(diffs <= max_gap_seconds)


def build_global_dataset(df, M, N, time_gap_seconds):
    df = df.sort_values('timestamp').reset_index(drop=True)
    timestamps = df['timestamp'].reset_index(drop=True)

    windspeed_columns = [col for col in df.columns if col.startswith('WINDSPEED_#')]
    windspeed_array = df[windspeed_columns].to_numpy()
    windspeed_mean_array = np.mean(windspeed_array, axis=1)

    active_power_array = df['ACTIVE_POWER_BING_PROCESS'].to_numpy()
    limit_power_array = df['LIMIT_POWER'].to_numpy()

    features_list, targets_list, timestamps_list = [], [], []
    skipped_due_to_time = 0

    for i in range(len(df) - M + 1):
        idx_start = i
        idx_end = i + M

        if not check_time_continuity(timestamps[idx_start:idx_end], time_gap_seconds):
            skipped_due_to_time += 1
            continue

        target_time = timestamps[idx_end - 1] + pd.Timedelta(seconds=N * time_gap_seconds)
        future_match = df[df['timestamp'] == target_time]
        if future_match.empty:
            skipped_due_to_time += 1
            continue

        flat_features = []
        for step in range(M):
            flat_features.extend([
                windspeed_mean_array[idx_start + step],
                active_power_array[idx_start + step]
                # limit_power_array[idx_start + step]
            ])

        target_value = future_match.iloc[0]['ACTIVE_POWER_BING_PROCESS']
        features_list.append(flat_features)
        targets_list.append(target_value)
        timestamps_list.append(target_time)

    print(f"⏳ 全局特征跳过了 {skipped_due_to_time} 个无效窗口")
    return np.array(features_list), np.array(targets_list), np.array(timestamps_list)


def save_metrics_to_csv(metrics, directory, filename):
    ensure_dir(directory)
    filepath = os.path.join(directory, filename)
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


def save_predictions_to_csv(timestamps, y_true, y_pred, directory, M, N, experiment_id):
    ensure_dir(directory)
    output_filename = f'predictions_global_M{M}_N{N}.csv'
    filepath = os.path.join(directory, output_filename)
    df_out = pd.DataFrame({
        'timestamp': timestamps,
        'ACTIVE_POWER_BING_PROCESS': y_true,
        'PREDICTED_ACTIVE_POWER_BING_PROCESS': y_pred
    })
    df_out.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"📝 预测结果已保存到: {filepath}")


def run_experiment(X, y, timestamps, feature_names, M, N, time_gap_seconds, experiment_id, output_dir):
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    timestamps_test = timestamps[split_index:]

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, free_raw_data=False)
    test_data = lgb.Dataset(X_test, label=y_test, feature_name=feature_names, free_raw_data=False)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.02,
        'num_leaves': 127,
        'max_depth': 12,
        'min_data_in_leaf': 5,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'bagging_freq': 1,
        'verbose': -1,
        'seed': RANDOM_SEED,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(10)]
    )

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'\n✅ 全局实验 {experiment_id}')
    print(f'RMSE : {rmse:.2f} | MAE : {mae:.2f} | R² : {r2:.4f}')

    metrics_filename = f'metrics_global_M{M}_N{N}.csv'
    metrics = {
        'experiment_id': experiment_id,
        'M': M,
        'N': N,
        'time_gap_seconds': time_gap_seconds,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    save_metrics_to_csv(metrics, output_dir, metrics_filename)
    save_predictions_to_csv(timestamps_test, y_test, y_pred, output_dir, M, N, experiment_id)

    train_data = None
    test_data = None
    gc.collect()

    return model


if __name__ == '__main__':
    # 参数配置
    data_file = r"G:\WindPowerForecast\输电线丙分风机预测\原始数据\峡阳峡沙点位数据20240315-20241201_峡阳B_V4_去重.csv"
    M = 60
    for N in [60, 120,180]:
        time_gap_seconds = 60
        output_dir = r"G:\WindPowerForecast\输电线丙分风机预测\全局特征预测结果_无限电"
        ensure_dir(output_dir)

        # 数据准备
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])

        print(f"✅ 数据集加载完成，包含 {len(df)} 行数据")

        # 找出LIMIT_POWER大于0小于900行，筛除
        df = df[(df['LIMIT_POWER'] == 0) | (df['LIMIT_POWER'] >= 900)]
        print(f"✅ 数据集加载完成，包含 {len(df)} 行数据")

        # 构建数据集
        X, y, timestamps = build_global_dataset(df, M, N, time_gap_seconds)

        if len(X) == 0:
            print("⚠️ 没有有效数据，结束。")
        else:
            feature_names = []
            for step in range(M):
                feature_names.extend([
                    f'WS_MEAN_t-{M - 1 - step}',
                    f'ACTIVE_POWER_BING_PROCESS_t-{M - 1 - step}'
                    # f'LIMIT_POWER_t-{M - 1 - step}'
                ])

            experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            time1 = time.time()
            model = run_experiment(X, y, timestamps, feature_names, M, N, time_gap_seconds, experiment_id, output_dir)
            time2 = time.time()
            print(f"✅ 全局特征训练完成，耗时 {time2 - time1:.2f} 秒\n")

            del X, y, timestamps, feature_names, model
            gc.collect()