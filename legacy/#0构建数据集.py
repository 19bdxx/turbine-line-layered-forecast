# ⚠️ 此文件已合并到 prepare_data.py，建议使用新文件。
# This file has been consolidated into prepare_data.py.

import pandas as pd

# 读取第一个文件
file_path1 = r"G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\按场站划分\峡阳B\峡阳峡沙点位数据20240315-20241201_峡阳B_V4.csv"
df = pd.read_csv(file_path1)
print("原始数据行数：", len(df))

# 读取重复时间戳文件
file_path2 = r"输电线丙_联合重复时间戳并集.csv"
df2 = pd.read_csv(file_path2)
print("需过滤时间戳数量：", len(df2))

# 实际匹配的时间戳数量
matched_count = df['timestamp'].isin(df2['timestamp']).sum()
print("实际匹配删除的行数：", matched_count)

# 过滤掉df2中存在的timestamp
filtered_df = df[~df['timestamp'].isin(df2['timestamp'])]
print("过滤后剩余行数：", len(filtered_df))

# 构建需要的列名列表
base_columns = ['timestamp','LIMIT_POWER','ACTIVE_POWER_BING_PROCESS','ACTIVE_POWER_BING']
status_columns = [f'STATUS_#{i}' for i in range(153, 200)]
power_columns = [f'ACTIVE_POWER_#{i}' for i in range(153, 200)]
REACTIVE_POWER_columns = [f'REACTIVE_POWER_#{i}' for i in range(153, 200)]
speed_columns = [f'WINDSPEED_#{i}' for i in range(153, 200)]
WINDDIRECTION_columns = [f'WINDDIRECTION_#{i}' for i in range(153, 200)]

# 确保选取的列在实际数据中存在
available_columns = [col for col in base_columns + status_columns + power_columns + REACTIVE_POWER_columns + speed_columns + WINDDIRECTION_columns if col in filtered_df.columns]

# 筛选并保存
filtered_df[available_columns].to_csv(
    r"峡阳峡沙点位数据20240315-20241201_峡阳B_V4_去重.csv",
    index=False
)
print("保存完成，文件路径：峡阳峡沙点位数据20240315-20241201_峡阳B_V4_去重.csv")