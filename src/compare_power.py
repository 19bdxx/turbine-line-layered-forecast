"""
输电线路功率对比分析 (Transmission Line Power Comparison Analysis)

对比风机功率求和与线路实测功率，计算误差分布并生成可视化图表。

运行方式：
  python compare_power.py

注意：运行前请将下方"配置"部分的 DATA_FILE 修改为实际路径，
      或通过环境变量 DATA_FILE 指定。
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

# 中文字体配置
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 配置 (Configuration)
# ============================================================
DATA_FILE = os.environ.get(
    "DATA_FILE",
    r"G:\WindPowerForecast\输电线丙分风机预测\峡阳峡沙点位数据20240315-20241201_峡阳B_V4_去重.csv",
)

CLIP_FAN_POWER  = True      # 将风机功率负值置零
CLIP_LINE_POWER = True      # 将线路测量功率负值置零
PLOT_STYLE      = "scatter" # 'scatter' 或 'line'
SAVE_PLOTS      = True      # 是否保存图片到当前目录

# 线路 → 风机编号列表映射
LINE_TO_FANS = {
    "戊": list(range(63, 110)),
    "丁": list(range(110, 153)),
    "丙": list(range(153, 200)),
}

# 线路 → 实测功率列名映射
LINE_MEASURE_COLS = {
    "戊": "ACTIVE_POWER_WU_PROCESS",
    "丁": "ACTIVE_POWER_DING_PROCESS",
    "丙": "ACTIVE_POWER_BING_PROCESS",
}


# ============================================================
# 主程序 (Main)
# ============================================================

def main() -> None:
    df = pd.read_csv(DATA_FILE, encoding="gbk", parse_dates=["timestamp"])

    # 负功率归零
    if CLIP_FAN_POWER:
        for fans in LINE_TO_FANS.values():
            for fan in fans:
                col = f"ACTIVE_POWER_#{fan}"
                if col in df.columns:
                    df[col] = df[col].clip(lower=0)
        print("✅ 所有风机负值已置零")

    if CLIP_LINE_POWER:
        for measure_col in LINE_MEASURE_COLS.values():
            if measure_col in df.columns:
                df[measure_col] = df[measure_col].clip(lower=0)
        print("✅ 所有线路测量功率负值已置零")

    # 逐线路分析
    for line, fans in LINE_TO_FANS.items():
        fan_cols = [f"ACTIVE_POWER_#{fan}" for fan in fans if f"ACTIVE_POWER_#{fan}" in df.columns]
        if not fan_cols:
            print(f"⚠️ 线路 {line} 无有效风机数据，跳过")
            continue

        measure_col = LINE_MEASURE_COLS.get(line)
        if measure_col not in df.columns:
            print(f"⚠️ 线路 {line} 无测量列 {measure_col}，跳过")
            continue

        fan_sum_col = f"{line}_fan_sum"
        error_col   = f"{line}_error"
        df[fan_sum_col] = df[fan_cols].sum(axis=1) / 1000  # 与线路测量功率列保持量纲一致
        df[error_col]   = df[fan_sum_col] - df[measure_col]

        total = len(df)
        pos = (df[error_col] > 0).sum()
        neg = (df[error_col] < 0).sum()
        zer = (df[error_col] == 0).sum()

        print(f"\n📊 线路 {line} 误差分布:")
        print(
            f"  正误差: {pos} ({pos / total:.1%})  "
            f"负误差: {neg} ({neg / total:.1%})  "
            f"零误差: {zer} ({zer / total:.1%})"
        )
        print(df[error_col].describe().to_string())

        # 绘图
        fig, ax = plt.subplots(figsize=(12, 5))
        if PLOT_STYLE == "scatter":
            ax.scatter(df["timestamp"], df[fan_sum_col],  label="风机功率总和 (kW)", color="blue",  s=1)
            ax.scatter(df["timestamp"], df[measure_col],  label="线路测量功率 (kW)", color="green", s=1)
            ax.scatter(df["timestamp"], df[error_col],    label="功率误差 (kW)",     color="red",   s=1)
        else:
            ax.plot(df["timestamp"], df[fan_sum_col], label="风机功率总和 (kW)", color="blue")
            ax.plot(df["timestamp"], df[measure_col], label="线路测量功率 (kW)", color="green")
            ax.plot(df["timestamp"], df[error_col],   label="功率误差 (kW)",     color="red")

        ax.set_title(f"{line} 线路：风机总和 vs 测量功率 vs 误差")
        ax.set_xlabel("时间")
        ax.set_ylabel("功率 (kW)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        if SAVE_PLOTS:
            png_path = f"{line}_功率分析结果.png"
            fig.savefig(png_path, dpi=150)
            print(f"📝 图表已保存: {png_path}")

        plt.show()

        # 保存 CSV
        output_csv = f"{line}_功率分析结果.csv"
        df[["timestamp", fan_sum_col, measure_col, error_col]].to_csv(
            output_csv, index=False, encoding="utf-8-sig"
        )
        print(f"📝 线路 {line} 分析结果已保存: {output_csv}")


if __name__ == "__main__":
    main()
