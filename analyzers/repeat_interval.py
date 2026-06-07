"""
repeat_interval.py - 工具2：重复间隔分布统计

统计 [0, N-1] 中每个数重复出现的间隔的分布情况。
间隔定义：同一数值两次出现之间的位置距离。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict


class RepeatInterval:
    """
    重复间隔分布统计工具
    
    对于每个数值 v ∈ [0, N-1]，统计其在序列中
    重复出现时的间隔（距离）分布。
    """

    def __init__(self):
        self.results: Dict = {}

    def analyze(self, data: np.ndarray, N: int,
                max_interval: int = 1000) -> Dict:
        """
        分析随机数序列的重复间隔分布
        
        Args:
            data: 随机整数序列
            N: 取值范围 [0, N-1]
            max_interval: 统计的最大间隔值
        
        Returns:
            包含间隔统计结果的字典
        """
        sample_size = len(data)

        # 记录每个数值出现的位置
        positions = defaultdict(list)
        for idx, val in enumerate(data):
            positions[int(val)].append(idx)

        # 统计每个数值的重复间隔
        all_intervals = []
        intervals_by_value = {}

        for val in range(N):
            pos_list = positions.get(val, [])
            if len(pos_list) < 2:
                intervals_by_value[val] = np.array([], dtype=np.int64)
                continue

            # 计算相邻出现位置的间隔
            pos_arr = np.array(pos_list, dtype=np.int64)
            intervals = np.diff(pos_arr)
            # 过滤掉超过 max_interval 的间隔
            intervals = intervals[intervals <= max_interval]
            intervals_by_value[val] = intervals
            all_intervals.extend(intervals.tolist())

        all_intervals = np.array(all_intervals, dtype=np.int64)

        # 全局间隔统计
        if len(all_intervals) > 0:
            global_mean = float(np.mean(all_intervals))
            global_median = float(np.median(all_intervals))
            global_std = float(np.std(all_intervals))
            global_min = int(np.min(all_intervals))
            global_max = int(np.max(all_intervals))

            # 间隔直方图统计
            interval_bins = np.arange(0, max_interval + 1, dtype=np.int64)
            interval_hist, _ = np.histogram(all_intervals, bins=interval_bins)
        else:
            global_mean = global_median = global_std = 0.0
            global_min = global_max = 0
            interval_hist = np.zeros(max_interval, dtype=np.int64)

        # 按数值的间隔统计
        value_avg_intervals = {}
        value_interval_std = {}
        for val in range(N):
            intervals = intervals_by_value.get(val, np.array([], dtype=np.int64))
            if len(intervals) > 0:
                value_avg_intervals[val] = float(np.mean(intervals))
                value_interval_std[val] = float(np.std(intervals))
            else:
                value_avg_intervals[val] = 0.0
                value_interval_std[val] = 0.0

        self.results = {
            "all_intervals": all_intervals,
            "intervals_by_value": intervals_by_value,
            "interval_histogram": interval_hist,
            "global_mean_interval": global_mean,
            "global_median_interval": global_median,
            "global_std_interval": global_std,
            "global_min_interval": global_min,
            "global_max_interval": global_max,
            "value_avg_intervals": value_avg_intervals,
            "value_interval_std": value_interval_std,
            "total_intervals_count": len(all_intervals),
            "sample_size": sample_size,
            "N": N,
            "max_interval": max_interval,
        }

        return self.results

    def to_dataframe(self, top_n: int = 10) -> pd.DataFrame:
        """
        将部分结果转换为 DataFrame
        
        Args:
            top_n: 显示前N个数值的间隔统计
        
        Returns:
            DataFrame
        """
        if not self.results:
            raise ValueError("请先调用 analyze() 方法")

        data = []
        for val in range(min(top_n, self.results["N"])):
            data.append({
                "数值": val,
                "平均间隔": self.results["value_avg_intervals"].get(val, 0),
                "间隔标准差": self.results["value_interval_std"].get(val, 0),
            })

        return pd.DataFrame(data)

    def summary(self) -> str:
        """返回重复间隔分布统计摘要"""
        if not self.results:
            return "尚未进行分析"

        N = self.results["N"]
        sample_size = self.results["sample_size"]

        summary = (
            f"\n{'='*60}\n"
            f"  工具2：重复间隔分布统计\n"
            f"{'='*60}\n"
            f"  样本数量: {sample_size}\n"
            f"  取值范围: [0, {N-1}]\n"
            f"  有效间隔数: {self.results['total_intervals_count']}\n"
            f"  {'─'*50}\n"
            f"  全局平均间隔: {self.results['global_mean_interval']:.2f}\n"
            f"  全局中位间隔: {self.results['global_median_interval']:.2f}\n"
            f"  全局间隔标准差: {self.results['global_std_interval']:.2f}\n"
            f"  最小间隔: {self.results['global_min_interval']}\n"
            f"  最大间隔: {self.results['global_max_interval']}\n"
            f"  {'─'*50}\n"
            f"  理论平均间隔: {sample_size / (sample_size / N):.2f} (≈N={N})\n"
            f"{'='*60}\n"
        )
        return summary
