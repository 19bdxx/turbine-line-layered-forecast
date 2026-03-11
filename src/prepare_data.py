"""
数据准备脚本 (Data Preparation Script)

步骤：
  1. 从原始 CSV 读取数据，过滤重复时间戳，选取所需列并保存清洗后数据集
  2. 对清洗后数据执行各风机多字段联合重复段检测（数据质量检查）

运行方式：
  python prepare_data.py

注意：运行前请将下方"配置"部分的路径修改为实际路径，
      或通过环境变量 RAW_DATA_FILE / DUPLICATE_TS_FILE / CLEAN_OUTPUT_FILE 指定。
"""

import os
import pandas as pd

# ============================================================
# 配置 (Configuration)
# ============================================================
RAW_DATA_FILE     = os.environ.get(
    "RAW_DATA_FILE",
    r"G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\按场站划分\峡阳B"
    r"\峡阳峡沙点位数据20240315-20241201_峡阳B_V4.csv",
)
DUPLICATE_TS_FILE = os.environ.get(
    "DUPLICATE_TS_FILE",
    r"输电线丙_联合重复时间戳并集.csv",
)
CLEAN_OUTPUT_FILE = os.environ.get(
    "CLEAN_OUTPUT_FILE",
    r"G:\WindPowerForecast\输电线丙分风机预测\原始数据"
    r"\峡阳峡沙点位数据20240315-20241201_峡阳B_V4_去重.csv",
)
QC_OUTPUT_DIR     = os.environ.get(
    "QC_OUTPUT_DIR",
    r"G:\WindPowerForecast\输电线丙分风机预测\联合重复值检测结果",
)

# 风机编号范围
TURBINE_ID_RANGE = range(153, 200)

# 联合重复检测的字段前缀
FIELD_PREFIXES = [
    "STATUS_",
    "ACTIVE_POWER_",
    "REACTIVE_POWER_",
    "WINDSPEED_",
    "WINDDIRECTION_",
]

# 连续相同值超过此阈值才算重复段
MIN_REPEAT = 5


# ============================================================
# 步骤 1：数据清洗 (Data Cleaning)
# ============================================================

def clean_dataset(raw_file: str, dup_ts_file: str, output_file: str) -> pd.DataFrame:
    """
    读取原始数据，过滤重复时间戳，选取所需列，保存清洗结果。

    Returns:
        清洗后的 DataFrame
    """
    print("=" * 55)
    print("步骤 1: 数据清洗")
    print("=" * 55)

    df = pd.read_csv(raw_file)
    print(f"原始数据行数: {len(df)}")

    # 过滤重复时间戳
    if os.path.exists(dup_ts_file):
        df_dup = pd.read_csv(dup_ts_file)
        print(f"需过滤时间戳数量: {len(df_dup)}")
        matched = df["timestamp"].isin(df_dup["timestamp"]).sum()
        print(f"实际匹配删除行数: {matched}")
        df = df[~df["timestamp"].isin(df_dup["timestamp"])]
        print(f"过滤后剩余行数: {len(df)}")
    else:
        print(f"⚠️  重复时间戳文件不存在，跳过过滤: {dup_ts_file}")

    # 选取所需列
    base_cols = ["timestamp", "LIMIT_POWER", "ACTIVE_POWER_BING_PROCESS", "ACTIVE_POWER_BING"]
    extra_cols = []
    for prefix in ["STATUS_#", "ACTIVE_POWER_#", "REACTIVE_POWER_#", "WINDSPEED_#", "WINDDIRECTION_#"]:
        extra_cols += [f"{prefix}{i}" for i in TURBINE_ID_RANGE]
    available = [c for c in base_cols + extra_cols if c in df.columns]
    df = df[available]

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"✅ 清洗后数据已保存: {output_file}")
    return df


# ============================================================
# 步骤 2：数据质量检查——联合重复段检测 (Quality Check)
# ============================================================

def _extract_fan_numbers(df: pd.DataFrame, prefixes: list) -> list:
    fan_numbers = set()
    for prefix in prefixes:
        for col in df.columns:
            if prefix in col and "#" in col:
                try:
                    fan_numbers.add(int(col.split("#")[1]))
                except (ValueError, IndexError):
                    pass
    return sorted(fan_numbers)


def _detect_joint_repeats(
    df: pd.DataFrame,
    fan_numbers: list,
    field_prefixes: list,
    timestamp_col: str = "timestamp",
    min_repeat: int = 5,
) -> list:
    """检测各风机多字段联合不变的连续重复段。"""
    results = []
    df = df.reset_index(drop=True)

    for fan_num in fan_numbers:
        col_names = [f"{p}#{fan_num}" for p in field_prefixes]
        if not all(c in df.columns for c in col_names):
            continue

        current_val = None
        count = 0
        start_idx = None

        for idx in range(len(df)):
            joint_val = tuple(df.loc[idx, col_names])
            if joint_val == current_val:
                count += 1
            else:
                if count >= min_repeat:
                    results.append(
                        {
                            "风机编号": fan_num,
                            "重复值组合": current_val,
                            "开始时间": df.loc[start_idx, timestamp_col],
                            "结束时间": df.loc[idx - 1, timestamp_col],
                            "持续长度": count,
                        }
                    )
                current_val = joint_val
                count = 1
                start_idx = idx

        if count >= min_repeat:
            results.append(
                {
                    "风机编号": fan_num,
                    "重复值组合": current_val,
                    "开始时间": df.loc[start_idx, timestamp_col],
                    "结束时间": df.loc[len(df) - 1, timestamp_col],
                    "持续长度": count,
                }
            )

    return results


def run_quality_check(df: pd.DataFrame, output_dir: str) -> None:
    """对清洗后数据执行联合重复段检测并保存结果。"""
    print("\n" + "=" * 55)
    print("步骤 2: 数据质量检查（联合重复段检测）")
    print("=" * 55)

    fan_numbers = _extract_fan_numbers(df, FIELD_PREFIXES)
    print(f"提取风机编号: {fan_numbers}")

    results = _detect_joint_repeats(
        df, fan_numbers, FIELD_PREFIXES, timestamp_col="timestamp", min_repeat=MIN_REPEAT
    )

    os.makedirs(output_dir, exist_ok=True)

    if results:
        df_res = pd.DataFrame(results)

        excel_path = os.path.join(output_dir, "联合重复值检测结果.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_res.to_excel(writer, sheet_name="联合重复检测结果", index=False)
        print(f"✅ 联合重复检测结果已保存: {excel_path}")

        summary = (
            df_res.groupby("风机编号")["持续长度"]
            .sum()
            .reset_index()
            .rename(columns={"持续长度": "联合重复总长度"})
        )
        summary_csv = os.path.join(output_dir, "联合重复值总时长汇总.csv")
        summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        print(f"✅ 联合重复时长汇总已保存: {summary_csv}")
    else:
        print("📭 未检测到联合重复值段。")


# ============================================================
# 主程序 (Main)
# ============================================================

if __name__ == "__main__":
    df_clean = clean_dataset(RAW_DATA_FILE, DUPLICATE_TS_FILE, CLEAN_OUTPUT_FILE)
    run_quality_check(df_clean, QC_OUTPUT_DIR)
