# ⚠️ 此文件已合并到 forecast.py（run_turbine_forecast 函数），建议使用新文件。
# This file has been consolidated into forecast.py (run_turbine_forecast function).

import os
import pandas as pd

def merge_metrics(root_dir, output_filename, required_substring=None):
    all_dfs = []
    for turbine_dir in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, turbine_dir)
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                if file.startswith('metrics') and file.endswith('.csv'):
                    if required_substring is None or required_substring in file:
                        file_path = os.path.join(dir_path, file)
                        df = pd.read_csv(file_path)
                        all_dfs.append(df)

    if not all_dfs:
        print("⚠️ 没有找到符合条件的 metrics 文件。")
        return

    merged_df = pd.concat(all_dfs, ignore_index=True)
    output_path = os.path.join(root_dir, output_filename)
    merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ Metrics 已保存到 {output_path}")

def merge_and_align_predictions(root_dir, output_filename, required_substring=None):
    merged_dfs = []
    for turbine_dir in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, turbine_dir)
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                if file.startswith('predictions') and file.endswith('.csv'):
                    if required_substring is None or required_substring in file:
                        file_path = os.path.join(dir_path, file)

                        # 提取风机编号
                        parts = file.split('_')
                        fan_id = next((part.replace('turbine', '').strip() for part in parts if part.startswith('turbine')), None)
                        if fan_id is None:
                            continue

                        df = pd.read_csv(file_path)
                        df = df.rename(columns={
                            'true_value': f'ACTIVE_POWER_#{fan_id}',
                            'predicted_value': f'PREDICTED_ACTIVE_POWER_#{fan_id}'
                        })
                        merged_dfs.append(df)

    if not merged_dfs:
        print("⚠️ 没有找到符合条件的 predictions 文件。")
        return

    result_df = merged_dfs[0]
    for df in merged_dfs[1:]:
        result_df = pd.merge(result_df, df, on='timestamp', how='outer')

    result_df = result_df.sort_values('timestamp').reset_index(drop=True)
    output_path = os.path.join(root_dir, output_filename)
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ Predictions 已保存到 {output_path}")

if __name__ == "__main__":
    root_dir = r"G:\WindPowerForecast\输电线丙分风机预测\风机预测结果_无限电"

    for required_substring in ["M60_N15","M60_N240"]:
        # 合并 Metrics
        merge_metrics(root_dir, output_filename=f"all_metrics_{required_substring}_无限电.csv", required_substring=required_substring)

        # 合并 Predictions
        merge_and_align_predictions(root_dir, output_filename=f"all_predictions_aligned_{required_substring}_无限电.csv", required_substring=required_substring)