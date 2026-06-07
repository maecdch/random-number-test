"""
autocorrelation.py - 工具6：自相关检验

自相关检验用于检测随机数序列中是否存在相关性。
如果序列是真正随机的，则不同滞后的自相关系数应接近零。

自相关系数定义（滞后 k）：
    r_k = Σ (x_t - μ)(x_{t+k} - μ) / Σ (x_t - μ)²

其中 μ 为序列均值，k 为滞后阶数。

原假设 H0：序列在滞后 k 处不存在自相关
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional


class AutocorrelationTest:
    """
    自相关检验工具
    
    检测 rand()%N 生成序列中不同滞后的自相关性。
    """

    def __init__(self):
        self.results: Dict = {}

    def analyze(self, data: np.ndarray, N: int,
                max_lag: int = 50, alpha: float = 0.05) -> Dict:
        """
        执行自相关检验
        
        计算并检验多个滞后的自相关系数，使用 Ljung-Box Q 检验
        判断序列整体是否存在显著的自相关。
        
        Args:
            data: 随机整数序列
            N: 取值范围
            max_lag: 最大滞后阶数
            alpha: 显著性水平
        
        Returns:
            包含检验结果的字典:
            - "autocorrelations": 各滞后阶数的自相关系数数组
            - "q_stat": Ljung-Box Q 统计量
            - "p_value": Q统计量的p值
            - "is_independent": 是否接受独立性假设
            - "significant_lags": 显著自相关的滞后阶数列表
            - "max_acf": 最大绝对自相关系数
        """
        sample_size = len(data)

        # 数据中心化
        centered = data - np.mean(data)
        variance = np.var(data)

        # 计算各滞后的自相关系数
        autocorrelations = np.zeros(max_lag + 1)
        for k in range(1, max_lag + 1):
            if variance > 0:
                # r_k = Σ (x_t - μ)(x_{t+k} - μ) / (n * σ²)
                cov = np.sum(centered[:-k] * centered[k:]) / sample_size
                autocorrelations[k] = cov / variance

        # Ljung-Box Q 检验
        # Q = n(n+2) * Σ (r_k² / (n-k))
        q_stat = 0.0
        for k in range(1, max_lag + 1):
            if sample_size > k:
                q_stat += autocorrelations[k] ** 2 / (sample_size - k)
        q_stat *= sample_size * (sample_size + 2)

        # Q统计量服从 χ²(max_lag) 分布
        from scipy.stats import chi2
        p_value = 1.0 - chi2.cdf(q_stat, max_lag)

        # 找出显著自相关的滞后（|r_k| > 2/sqrt(n) 的近似95%置信区间）
        threshold = 2.0 / np.sqrt(sample_size)
        significant_lags = [int(k) for k in range(1, max_lag + 1)
                            if abs(autocorrelations[k]) > threshold]

        is_independent = bool(p_value >= alpha)

        self.results = {
            "autocorrelations": autocorrelations,
            "q_stat": float(q_stat),
            "p_value": float(p_value),
            "alpha": alpha,
            "is_independent": is_independent,
            "significant_lags": significant_lags,
            "max_acf": float(np.max(np.abs(autocorrelations[1:]))),
            "threshold": float(threshold),
            "max_lag": max_lag,
            "sample_size": sample_size,
            "N": N,
        }

        return self.results

    def summary(self) -> str:
        """返回自相关检验结果摘要"""
        if not self.results:
            return "尚未进行分析"

        r = self.results
        conclusion = "接受 H0：序列不存在显著自相关" if r["is_independent"] else "拒绝 H0：序列存在显著自相关"

        sig_lags_str = ", ".join(map(str, r["significant_lags"][:10]))
        if len(r["significant_lags"]) > 10:
            sig_lags_str += f"... (共{len(r['significant_lags'])}个)"

        summary = (
            f"\n{'='*60}\n"
            f"  工具6：自相关检验 (Ljung-Box Q Test)\n"
            f"{'='*60}\n"
            f"  原假设 H0: 序列不存在自相关（独立）\n"
            f"  显著性水平 α = {r['alpha']}\n"
            f"  最大滞后阶数: {r['max_lag']}\n"
            f"  {'─'*50}\n"
            f"  Ljung-Box Q 统计量 = {r['q_stat']:.4f}\n"
            f"  p-value = {r['p_value']:.6f}\n"
            f"  最大|自相关系数| = {r['max_acf']:.4f}\n"
            f"  95%置信阈值 = {r['threshold']:.4f}\n"
            f"  显著自相关滞后数: {len(r['significant_lags'])}\n"
        )

        if r['significant_lags']:
            summary += f"  显著滞后阶数: {sig_lags_str}\n"

        summary += (
            f"  {'─'*50}\n"
            f"  结论: {conclusion}\n"
            f"{'='*60}\n"
        )
        return summary
