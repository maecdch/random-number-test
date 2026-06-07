"""
ks_test.py - 工具5：Kolmogorov-Smirnov 检验

KS检验用于比较样本的经验分布函数(EDF)与理论分布函数(CDF)，
判断样本是否来自指定的理论分布。

对于 [0, N-1] 离散均匀分布：
    理论CDF: F(x) = (x+1)/N, x = 0, 1, ..., N-1

检验统计量: D = max|F_n(x) - F(x)|
其中 F_n(x) 为经验分布函数，F(x) 为理论分布函数

原假设 H0：样本服从指定的理论分布
"""

import numpy as np
from scipy.stats import kstest, ksone
from typing import Dict


class KSTest:
    """
    Kolmogorov-Smirnov 检验工具
    
    检验 rand()%N 生成的序列是否服从 [0, N-1] 上的均匀分布。
    """

    def __init__(self):
        self.results: Dict = {}

    def analyze(self, data: np.ndarray, N: int, alpha: float = 0.05) -> Dict:
        """
        执行KS检验
        
        Args:
            data: 随机整数序列
            N: 取值范围 [0, N-1]
            alpha: 显著性水平
        
        Returns:
            包含检验结果的字典:
            - "d_stat": KS统计量 D
            - "p_value": p值
            - "critical_value": 临界值
            - "alpha": 显著性水平
            - "is_uniform": 是否接受均匀分布假设
            - "d_plus": D+ 统计量（正向最大偏差）
            - "d_minus": D- 统计量（负向最大偏差）
        """
        sample_size = len(data)

        # 对离散数据使用连续性校正
        # KS检验更适合连续分布，但对离散数据也可近似使用
        # 对每个整数点添加均匀抖动（jitter）以近似连续分布
        jittered = data + np.random.uniform(-0.5, 0.5, size=sample_size)

        # 理论分布：[-0.5, N-0.5] 上的均匀分布
        # 对应 [0, N-1] 离散均匀分布
        loc = -0.5
        scale = N  # 范围宽度

        # 执行KS检验
        d_stat, p_value = kstest(jittered, 'uniform', args=(loc, scale))

        # 计算临界值
        # 对于大样本，近似临界值: c(alpha) / sqrt(n)
        from scipy.stats import ksone
        critical_value = ksone.ppf(1 - alpha / 2, sample_size)

        # 计算 D+ 和 D-
        sorted_data = np.sort(data)
        n = sample_size

        # 经验分布函数 vs 理论分布函数
        # 对于离散均匀分布，理论CDF: F(x) = (x + 1) / N
        d_plus = 0.0
        d_minus = 0.0
        for i, x in enumerate(sorted_data):
            # 经验CDF: (i + 1) / n
            ecdf = (i + 1) / n
            # 理论CDF (x 处的值)
            tcdf = (x + 1) / N
            # x 之前的理论CDF
            tcdf_before = x / N if x > 0 else 0.0

            d_plus = max(d_plus, ecdf - tcdf_before)
            d_minus = max(d_minus, tcdf - (i) / n)

        conclusion = bool(p_value >= alpha)

        self.results = {
            "d_stat": float(d_stat),
            "p_value": float(p_value),
            "critical_value": float(critical_value),
            "d_plus": float(d_plus),
            "d_minus": float(d_minus),
            "alpha": alpha,
            "is_uniform": conclusion,
            "sample_size": sample_size,
            "N": N,
        }

        return self.results

    def summary(self) -> str:
        """返回KS检验结果摘要"""
        if not self.results:
            return "尚未进行分析"

        r = self.results
        conclusion = "接受 H0：服从均匀分布" if r["is_uniform"] else "拒绝 H0：不服从均匀分布"

        summary = (
            f"\n{'='*60}\n"
            f"  工具5：Kolmogorov-Smirnov 检验\n"
            f"{'='*60}\n"
            f"  原假设 H0: 样本服从 [0, {r['N']-1}] 上的均匀分布\n"
            f"  显著性水平 α = {r['alpha']}\n"
            f"  {'─'*50}\n"
            f"  KS统计量 D = {r['d_stat']:.6f}\n"
            f"  D+ = {r['d_plus']:.6f}\n"
            f"  D- = {r['d_minus']:.6f}\n"
            f"  临界值 = {r['critical_value']:.6f}\n"
            f"  p-value = {r['p_value']:.6f}\n"
            f"  {'─'*50}\n"
            f"  结论: {conclusion}\n"
            f"{'='*60}\n"
        )
        return summary
