import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import csv

# 文件路径配置
fenji_file_path = r"G:\WindPowerForecast\输电线丙分风机预测\风机预测结果_无限电\all_predictions_aligned_M60_N240_无限电.csv"
global_file_path = r"G:\WindPowerForecast\输电线丙分风机预测\全局特征预测结果_无限电\predictions_global_M60_N240_20250519_233058.csv"
line_loss_file_path = r"G:\WindPowerForecast\输电线丙分风机预测\输电线损分析\丙_功率区间均值对比.csv"
adjusted_output_path = r"G:\WindPowerForecast\输电线丙分风机预测\预测结果对比\final_adjusted_with_net_power_M60_N240_无限电.csv"
metrics_output_path = r"G:\WindPowerForecast\输电线丙分风机预测\预测结果对比\comparison_metrics_after_line_loss_M60_N240_无限电.csv"

# ========== 读取并合并数据 ==========
df_fenji = pd.read_csv(fenji_file_path, parse_dates=['timestamp'])
df_global = pd.read_csv(global_file_path, parse_dates=['timestamp'])
df_merged = pd.merge(df_fenji, df_global, on='timestamp', how='inner').sort_values('timestamp')

# 负值处理
df_merged['ACTIVE_POWER_BING_PROCESS'] = df_merged['ACTIVE_POWER_BING_PROCESS'].clip(lower=0)
df_merged['PREDICTED_ACTIVE_POWER_BING_PROCESS'] = df_merged['PREDICTED_ACTIVE_POWER_BING_PROCESS'].clip(lower=0)

predicted_cols = [col for col in df_merged.columns if col.startswith('PREDICTED_ACTIVE_POWER_#')]
df_merged[predicted_cols] = df_merged[predicted_cols].clip(lower=0)

# 风机预测总和（单位千瓦）
df_merged['SUM_PREDICTED_ACTIVE_POWER'] = df_merged[predicted_cols].sum(axis=1) / 1000

# ========== 读取线损表 CSV ==========
df_line_loss = pd.read_csv(line_loss_file_path)

# 解析区间字符串 "(0.0, 5.0]" 为数字
df_line_loss[['min_power', 'max_power']] = df_line_loss['功率区间'].str.extract(r'\((.*), (.*)\]', expand=True).astype(float)
df_line_loss['line_loss'] = df_line_loss['筛选后_平均误差']

def get_line_loss(power):
    for _, row in df_line_loss.iterrows():
        if row['min_power'] < power <= row['max_power']:
            return row['line_loss']
    return 0  # 默认最大线损

# 计算线损和净功率
df_merged['LINE_LOSS'] = df_merged['SUM_PREDICTED_ACTIVE_POWER'].apply(get_line_loss)
df_merged['NET_PREDICTED_POWER'] = df_merged['SUM_PREDICTED_ACTIVE_POWER'] - df_merged['LINE_LOSS']
df_merged['NET_PREDICTED_POWER'] = df_merged['NET_PREDICTED_POWER'].clip(lower=0)

# ========== 评估净功率 vs 全局模型 ==========
def evaluate_predictions(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"📊 {label} 评估 - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    return {'label': label, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

y_true = df_merged['ACTIVE_POWER_BING_PROCESS']
y_global_pred = df_merged['PREDICTED_ACTIVE_POWER_BING_PROCESS']
y_net_sum_pred = df_merged['NET_PREDICTED_POWER']

global_metrics = evaluate_predictions(y_true, y_global_pred, '全局模型预测')
net_sum_metrics = evaluate_predictions(y_true, y_net_sum_pred, '风机净功率求和（扣除线损）')

with open(metrics_output_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['label', 'RMSE', 'MAE', 'R2'])
    writer.writeheader()
    writer.writerow(global_metrics)
    writer.writerow(net_sum_metrics)

print(f"✅ 已保存净功率评估结果到：{metrics_output_path}")

# ========== 保存最终调整结果 ==========
df_merged.to_csv(adjusted_output_path, index=False, encoding='utf-8-sig')
print(f"✅ 线损调整后结果已保存到：{adjusted_output_path}")


# import matplotlib.pyplot as plt
# # 中文字体配置
# plt.rcParams['font.family'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# # 准备数据
# timestamps = df_merged['timestamp']
# y_true = df_merged['ACTIVE_POWER_BING_PROCESS']
# y_global_pred = df_merged['PREDICTED_ACTIVE_POWER_BING_PROCESS']
# y_sum_pred = df_merged['SUM_PREDICTED_ACTIVE_POWER']
# y_net_pred = df_merged['NET_PREDICTED_POWER']

# # 绘制对比图
# plt.figure(figsize=(16, 8))
# plt.plot(timestamps, y_true, label='真实功率 (ACTIVE_POWER_BING_PROCESS)', linewidth=2)
# plt.plot(timestamps, y_global_pred, label='全局模型预测 (PREDICTED_ACTIVE_POWER_BING_PROCESS)', linewidth=2)
# plt.plot(timestamps, y_sum_pred, label='风机预测求和 (SUM_PREDICTED_ACTIVE_POWER)', linewidth=2)
# plt.plot(timestamps, y_net_pred, label='风机净功率（扣线损） (NET_PREDICTED_POWER)', linewidth=2)

# plt.title('功率预测对比曲线')
# plt.xlabel('时间')
# plt.ylabel('功率 (kW)')
# plt.legend(loc='upper left')
# plt.xticks(rotation=45)
# plt.tight_layout()

# # 保存和展示
# output_chart_path = r"G:\WindPowerForecast\输电线丙分风机预测\预测结果对比\power_prediction_comparison.png"
# plt.savefig(output_chart_path)
# plt.show()

# print(f"✅ 功率对比曲线已保存到：{output_chart_path}")

import plotly.graph_objects as go

# 创建图形
fig = go.Figure()

# 添加真实功率曲线
fig.add_trace(go.Scatter(x=df_merged['timestamp'], y=df_merged['ACTIVE_POWER_BING_PROCESS'],
                         mode='lines', name='真实功率 (ACTIVE_POWER_BING_PROCESS)'))

# 添加全局模型预测曲线
fig.add_trace(go.Scatter(x=df_merged['timestamp'], y=df_merged['PREDICTED_ACTIVE_POWER_BING_PROCESS'],
                         mode='lines', name='全局模型预测 (PREDICTED_ACTIVE_POWER_BING_PROCESS)'))

# 添加风机预测求和曲线
fig.add_trace(go.Scatter(x=df_merged['timestamp'], y=df_merged['SUM_PREDICTED_ACTIVE_POWER'],
                         mode='lines', name='风机预测求和 (SUM_PREDICTED_ACTIVE_POWER)'))

# 添加净功率曲线
fig.add_trace(go.Scatter(x=df_merged['timestamp'], y=df_merged['NET_PREDICTED_POWER'],
                         mode='lines', name='风机净功率（扣线损） (NET_PREDICTED_POWER)'))

# 布局设置
fig.update_layout(
    title='功率预测对比曲线',
    xaxis_title='时间',
    yaxis_title='功率 (kW)',
    hovermode='x unified',
    template='plotly_white'
)

# 保存为 HTML
output_html_path = r"G:\WindPowerForecast\输电线丙分风机预测\预测结果对比\power_prediction_comparison_M60_N240_无限电.html"
fig.write_html(output_html_path)

print(f"✅ 交互式 HTML 报告已保存到：{output_html_path}")