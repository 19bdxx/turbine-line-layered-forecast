# ⚠️ 此文件已重构为 compare_power.py，建议使用新文件。
# This file has been refactored into compare_power.py.

import pandas as pd
import matplotlib.pyplot as plt

# 中文字体配置
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置参数
clip_fan_power_to_zero = True     # 是否将风机功率负值置零
clip_line_power_to_zero = True    # 是否将线路测量功率负值置零
plot_style = 'scatter'               # 'scatter' 或 'line'

# 读取数据
file_path = r"G:\WindPowerForecast\输电线丙分风机预测\峡阳峡沙点位数据20240315-20241201_峡阳B_V4_去重.csv"
df = pd.read_csv(file_path, encoding='gbk', parse_dates=['timestamp'])

# 风机到线路映射
line_to_fans = {
    '戊': list(range(63, 110)),
    '丁': list(range(110, 153)),
    '丙': list(range(153, 200))
}

# 线路测量列映射
line_measure_cols = {
    '戊': 'ACTIVE_POWER_WU_PROCESS',
    '丁': 'ACTIVE_POWER_DING_PROCESS',
    '丙': 'ACTIVE_POWER_BING_PROCESS'
}

# 负功率归零处理 - 风机
if clip_fan_power_to_zero:
    for fans in line_to_fans.values():
        for fan in fans:
            col_name = f'ACTIVE_POWER_#{fan}'
            if col_name in df.columns:
                df[col_name] = df[col_name].clip(lower=0)
    print("✅ 所有风机负值已置零")
else:
    print("⚠️ 未进行风机功率负值置零")

# 负功率归零处理 - 线路
if clip_line_power_to_zero:
    for line, measure_col in line_measure_cols.items():
        if measure_col in df.columns:
            df[measure_col] = df[measure_col].clip(lower=0)
    print("✅ 所有线路测量功率负值已置零")
else:
    print("⚠️ 未进行线路测量功率负值置零")

# 检查每台风机功率是否有小于0的情况
print("\n🔍 检查风机功率是否存在小于0的情况：")
for fans in line_to_fans.values():
    for fan in fans:
        col_name = f'ACTIVE_POWER_#{fan}'
        if col_name in df.columns:
            negative_count = (df[col_name] < 0).sum()
            if negative_count > 0:
                print(f"⚠️ 风机 #{fan} 有 {negative_count} 个功率小于 0 的记录")
            else:
                print(f"✅ 风机 #{fan} 无功率小于 0 的记录")

# 检查输电线路测量功率是否有小于0的情况
print("\n🔍 检查线路测量功率是否存在小于0的情况：")
for line, measure_col in line_measure_cols.items():
    if measure_col in df.columns:
        negative_count = (df[measure_col] < 0).sum()
        if negative_count > 0:
            print(f"⚠️ 线路 {line} 测量列 {measure_col} 有 {negative_count} 个小于 0 的记录")
        else:
            print(f"✅ 线路 {line} 测量列 {measure_col} 无小于 0 的记录")

# 逐线路分析
for line, fans in line_to_fans.items():
    print(f"🔄 处理线路 {line}")

    # 构建风机列名
    fan_cols = [f'ACTIVE_POWER_#{fan}' for fan in fans if f'ACTIVE_POWER_#{fan}' in df.columns]
    if not fan_cols:
        print(f"⚠️ 线路 {line} 无有效风机数据")
        continue

    # 计算风机功率总和（单位转换为 kW）
    df[f'{line}_fan_sum'] = df[fan_cols].sum(axis=1) / 1000

    # 获取线路测量列
    measure_col = line_measure_cols.get(line)
    if measure_col not in df.columns:
        print(f"⚠️ 线路 {line} 无测量列 {measure_col}")
        continue

    # 计算误差
    df[f'{line}_error'] = df[f'{line}_fan_sum'] - df[measure_col]

    # 误差正负比例统计
    error_positive_count = (df[f'{line}_error'] > 0).sum()
    error_negative_count = (df[f'{line}_error'] < 0).sum()
    error_zero_count = (df[f'{line}_error'] == 0).sum()
    total_count = len(df)

    positive_ratio = error_positive_count / total_count
    negative_ratio = error_negative_count / total_count
    zero_ratio = error_zero_count / total_count

    print(f"📊 线路 {line} 误差正负比例统计：")
    print(f"误差 > 0 的数量: {error_positive_count}，占比: {positive_ratio:.2%}")
    print(f"误差 < 0 的数量: {error_negative_count}，占比: {negative_ratio:.2%}")
    print(f"误差 = 0 的数量: {error_zero_count}，占比: {zero_ratio:.2%}\n")

    # 绘图：风机总和、线路测量、误差
    plt.figure(figsize=(12, 5))

    if plot_style == 'scatter':
        plt.scatter(df['timestamp'], df[f'{line}_fan_sum'], label='风机功率总和 (kW)', color='blue', s=1)
        plt.scatter(df['timestamp'], df[measure_col], label='线路测量功率 (kW)', color='green', s=1)
        plt.scatter(df['timestamp'], df[f'{line}_error'], label='功率误差 (kW)', color='red', s=1)
    else:
        plt.plot(df['timestamp'], df[f'{line}_fan_sum'], label='风机功率总和 (kW)', color='blue')
        plt.plot(df['timestamp'], df[measure_col], label='线路测量功率 (kW)', color='green')
        plt.plot(df['timestamp'], df[f'{line}_error'], label='功率误差 (kW)', color='red')

    plt.title(f'{line} 线路 风机总和 vs 测量功率 vs 误差')
    plt.xlabel('时间')
    plt.ylabel('功率 (kW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 输出误差统计描述
    error_stats = df[f'{line}_error'].describe()
    print(f'线路 {line} 误差统计描述：\n{error_stats}\n')

    # 保存分析结果到 CSV
    output_cols = ['timestamp', f'{line}_fan_sum', measure_col, f'{line}_error']
    output_df = df[output_cols].copy()
    output_filename = f'{line}_功率分析结果.csv'
    output_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"📝 线路 {line} 分析结果已保存到: {output_filename}\n")
