"""
runs_test.py - 工具4：游程检验（Runs Test）

游程检验用于检测序列的随机性，判断序列中元素是否独立。

游程定义：序列中连续出现的相同值或连续单调的片段。

本实现采用 Wald-Wolfowitz 游程检验，基于序列中值将数据
分为两类（高于中位数/低于中位数），统计游程数量。

原假设 H0：序列是随机的（游程数量符合预期）
备择假设 H1：序列不是随机的
"""

import numpy as np
from scipy.stats import norm
from typing import Dict


class RunsTest:
    """
    游程检验工具
    
    检验 rand()%N 生成序列的随机性（独立性）。
    """

    def __init__(self):
        self.results: Dict = {}

    def analyze(self, data: np.ndarray, N: int, alpha: float = 0.05) -> Dict:
        """
        执行游程检验
        
        基于序列中值将数据分为高于中位数(1)和低于中位数(0)两类，
        统计游程数量并与期望值比较。
        
        Args:
            data: 随机整数序列
            N: 取值范围（用于计算理论中位数）
            alpha: 显著性水平
        
        Returns:
            包含检验结果的字典:
            - "runs_count": 实际游程数
            - "expected_runs": 期望游程数
            - "runs_std": 游程数标准差
            - "z_stat": Z统计量
            - "p_value": p值（双尾）
            - "is_random": 是否接受随机性假设
            - "above_median": 高于中位数的数量
            - "below_median": 低于中位数的数量
        """
        sample_size = len(data)

        # 计算中位数
        median = float(np.median(data))

        # 将数据转换为二元序列：1=高于中位数, 0=低于或等于中位数
        binary = np.where(data > median, 1, 0)

        n1 = int(np.sum(binary))       # 高于中位数的个数
        n0 = sample_size - n1           # 低于或等于中位数的个数

        # 计算游程数量
        runs = 1
        for i in range(1, sample_size):
            if binary[i] != binary[i - 1]:
                runs += 1

        # 期望游程数
        expected_runs = (2.0 * n0 * n1) / (n0 + n1) + 1

        # 游程数标准差
        numerator = 2.0 * n0 * n1 * (2.0 * n0 * n1 - n0 - n1)
        denominator = (n0 + n1) ** 2 * (n0 + n1 - 1)
        runs_std = np.sqrt(numerator / denominator) if denominator > 0 else 0.0

        # Z统计量
        if runs_std > 0:
            z_stat = (runs - expected_runs) / runs_std
        else:
            z_stat = 0.0

        # p值（双尾检验）
        p_value = 2.0 * (1.0 - norm.cdf(abs(z_stat)))

        # 结论
        is_random = bool(p_value >= alpha)

        self.results = {
            "runs_count": int(runs),
            "expected_runs": float(expected_runs),
            "runs_std": float(runs_std),
            "z_stat": float(z_stat),
            "p_value": float(p_value),
            "alpha": alpha,
            "is_random": is_random,
            "n1_above": n1,
            "n0_below": n0,
            "sample_size": sample_size,
            "N": N,
        }

        return self.results

    def summary(self) -> str:
        """返回游程检验结果摘要"""
        if not self.results:
            return "尚未进行分析"

        r = self.results
        conclusion = "接受 H0：序列具有随机性" if r["is_random"] else "拒绝 H0：序列不具有随机性"

        summary = (
            f"\n{'='*60}\n"
            f"  工具4：游程检验 (Wald-Wolfowitz Runs Test)\n"
            f"{'='*60}\n"
            f"  原假设 H0: 序列是随机的\n"
            f"  显著性水平 α = {r['alpha']}\n"
            f"  {'─'*50}\n"
            f"  高于中位数个数 (n1): {r['n1_above']}\n"
            f"  低于中位数个数 (n0): {r['n0_below']}\n"
            f"  实际游程数:   {r['runs_count']}\n"
            f"  期望游程数:   {r['expected_runs']:.2f}\n"
            f"  游程标准差:   {r['runs_std']:.4f}\n"
            f"  Z统计量:      {r['z_stat']:.4f}\n"
            f"  p-value:      {r['p_value']:.6f}\n"
            f"  {'─'*50}\n"
            f"  结论: {conclusion}\n"
            f"{'='*60}\n"
        )
        return summary
