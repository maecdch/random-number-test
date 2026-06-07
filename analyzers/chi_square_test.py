"""
chi_square_test.py - 工具3：卡方检验

使用卡方拟合优度检验判断随机数序列是否服从均匀分布。

原假设 H0：数据服从 [0, N-1] 上的均匀分布
备择假设 H1：数据不服从均匀分布

检验统计量：χ² = Σ (O_i - E_i)² / E_i
其中 O_i 为第 i 类的观测频数，E_i 为期望频数
"""

import numpy as np
from scipy.stats import chisquare
from typing import Dict


class ChiSquareTest:
    """
    卡方拟合优度检验工具
    
    检验 rand()%N 生成的序列是否服从均匀分布。
    """

    def __init__(self):
        self.results: Dict = {}

    def analyze(self, data: np.ndarray, N: int, alpha: float = 0.05) -> Dict:
        """
        执行卡方检验
        
        Args:
            data: 随机整数序列
            N: 取值范围 [0, N-1]
            alpha: 显著性水平（默认 0.05）
        
        Returns:
            包含检验结果的字典:
            - "chi2_stat": 卡方统计量
            - "p_value": p值
            - "critical_value": 临界值
            - "dof": 自由度 (N-1)
            - "alpha": 显著性水平
            - "is_uniform": 是否接受均匀分布假设 (p >= alpha)
            - "effect_size": 效应量 (Cramer's V)
        """
        sample_size = len(data)

        # 统计频数
        observed = np.zeros(N, dtype=np.int64)
        for val in data:
            observed[int(val)] += 1

        # 期望频数
        expected = np.full(N, sample_size / N)

        # 执行卡方检验
        chi2_stat, p_value = chisquare(observed, f_exp=expected)

        # 计算临界值
        from scipy.stats import chi2 as chi2_dist
        critical_value = chi2_dist.ppf(1 - alpha, N - 1)

        # 效应量 (Cramer's V)
        # V = sqrt(χ² / (n * min(k-1, 1))) = sqrt(χ² / n) 因为 k-1=1 或更大
        # 对于均匀性检验，使用 V = sqrt(χ² / (n * (k-1)))
        effect_size = np.sqrt(chi2_stat / (sample_size * (N - 1)))

        self.results = {
            "chi2_stat": float(chi2_stat),
            "p_value": float(p_value),
            "critical_value": float(critical_value),
            "dof": N - 1,
            "alpha": alpha,
            "is_uniform": bool(p_value >= alpha),
            "effect_size": float(effect_size),
            "sample_size": sample_size,
            "N": N,
        }

        return self.results

    def summary(self) -> str:
        """返回卡方检验结果摘要"""
        if not self.results:
            return "尚未进行分析"

        r = self.results
        conclusion = "接受 H0：数据服从均匀分布" if r["is_uniform"] else "拒绝 H0：数据不服从均匀分布"

        summary = (
            f"\n{'='*60}\n"
            f"  工具3：卡方拟合优度检验\n"
            f"{'='*60}\n"
            f"  原假设 H0: 数据服从 [0, {r['N']-1}] 上的均匀分布\n"
            f"  显著性水平 α = {r['alpha']}\n"
            f"  {'─'*50}\n"
            f"  卡方统计量 χ² = {r['chi2_stat']:.4f}\n"
            f"  自由度 df = {r['dof']}\n"
            f"  临界值 = {r['critical_value']:.4f}\n"
            f"  p-value = {r['p_value']:.6f}\n"
            f"  效应量 (Cramer's V) = {r['effect_size']:.6f}\n"
            f"  {'─'*50}\n"
            f"  结论: {conclusion}\n"
            f"{'='*60}\n"
        )
        return summary
