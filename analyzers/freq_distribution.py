"""
freq_distribution.py - 工具1：频率分布统计

统计 [0, N-1] 中每个数的出现概率，并与理论均匀分布对比。
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class FreqDistribution:
    """
    频率分布统计工具
    
    统计 [0, N-1] 中每个数的出现频次和概率，
    计算实际分布与理论均匀分布的偏差。
    """

    def __init__(self):
        self.results: Dict = {}

    def analyze(self, data: np.ndarray, N: int) -> Dict:
        """
        分析随机数序列的频率分布
        
        Args:
            data: 随机整数序列，shape=(sample_size,)
            N: 取值范围 [0, N-1]
        
        Returns:
            包含统计结果的字典:
            - "frequencies": 每个数的出现频次
            - "probabilities": 每个数的出现概率
            - "expected_freq": 理论期望频次
            - "expected_prob": 理论期望概率 (1/N)
            - "max_deviation": 最大偏差
            - "mean_deviation": 平均绝对偏差
            - "std_deviation": 偏差标准差
            - "min_count": 最小频次
            - "max_count": 最大频次
            - "range_ratio": 最大最小频次比
        """
        sample_size = len(data)

        # 统计每个数出现的频次
        frequencies = np.zeros(N, dtype=np.int64)
        for val in data:
            frequencies[val] += 1

        # 计算概率
        probabilities = frequencies / sample_size

        # 理论值
        expected_freq = sample_size / N
        expected_prob = 1.0 / N

        # 偏差分析
        deviations = probabilities - expected_prob
        abs_deviations = np.abs(deviations)

        # 存储结果
        self.results = {
            "frequencies": frequencies,
            "probabilities": probabilities,
            "expected_freq": expected_freq,
            "expected_prob": expected_prob,
            "max_deviation": float(np.max(abs_deviations)),
            "mean_deviation": float(np.mean(abs_deviations)),
            "std_deviation": float(np.std(deviations)),
            "min_count": int(np.min(frequencies)),
            "max_count": int(np.max(frequencies)),
            "range_ratio": float(np.max(frequencies) / np.maximum(np.min(frequencies), 1)),
            "sample_size": sample_size,
            "N": N,
        }

        return self.results

    def to_dataframe(self) -> pd.DataFrame:
        """
        将频率统计结果转换为 DataFrame
        
        Returns:
            包含每个数值统计信息的 DataFrame
        """
        if not self.results:
            raise ValueError("请先调用 analyze() 方法")

        N = self.results["N"]
        df = pd.DataFrame({
            "数值": range(N),
            "出现频次": self.results["frequencies"],
            "出现概率": self.results["probabilities"],
            "期望概率": self.results["expected_prob"],
            "偏差": self.results["probabilities"] - self.results["expected_prob"],
        })
        return df

    def summary(self) -> str:
        """返回频率分布统计摘要"""
        if not self.results:
            return "尚未进行分析"

        N = self.results["N"]
        sample_size = self.results["sample_size"]

        summary = (
            f"\n{'='*60}\n"
            f"  工具1：频率分布统计\n"
            f"{'='*60}\n"
            f"  样本数量: {sample_size}\n"
            f"  取值范围: [0, {N-1}]\n"
            f"  理论概率: 1/{N} = {self.results['expected_prob']:.6f}\n"
            f"  理论期望频次: {self.results['expected_freq']:.2f}\n"
            f"  {'─'*50}\n"
            f"  实际最小频次: {self.results['min_count']}\n"
            f"  实际最大频次: {self.results['max_count']}\n"
            f"  最大/最小频次比: {self.results['range_ratio']:.4f}\n"
            f"  {'─'*50}\n"
            f"  最大概率偏差: {self.results['max_deviation']:.6f}\n"
            f"  平均绝对偏差: {self.results['mean_deviation']:.6f}\n"
            f"  偏差标准差:   {self.results['std_deviation']:.6f}\n"
            f"{'='*60}\n"
        )
        return summary
