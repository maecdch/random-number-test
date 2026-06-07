"""
normal_fit_test.py - 工具7：正态分布拟合检验（辅助工具）

用于检验通过 Box-Muller 或 CLT 方法生成的正态分布随机数
是否确实服从正态分布。

使用 Shapiro-Wilk 检验和 Q-Q 图分析。
"""

import numpy as np
from scipy.stats import shapiro, normaltest, probplot
from typing import Dict, Optional


class NormalFitTest:
    """
    正态分布拟合检验工具
    
    检验生成的随机数是否服从正态分布。
    """

    def __init__(self):
        self.results: Dict = {}

    def analyze(self, data: np.ndarray, alpha: float = 0.05) -> Dict:
        """
        执行正态分布拟合检验
        
        使用 Shapiro-Wilk 检验和 D'Agostino-Pearson 检验。
        
        Args:
            data: 待检验的浮点数序列
            alpha: 显著性水平
        
        Returns:
            包含检验结果的字典:
            - "shapiro_stat": Shapiro-Wilk 统计量 W
            - "shapiro_p": Shapiro-Wilk p值
            - "dagostino_stat": D'Agostino-Pearson 统计量
            - "dagostino_p": D'Agostino-Pearson p值
            - "skewness": 偏度
            - "kurtosis": 峰度
            - "is_normal": 是否接受正态分布假设
            - "sample_mean": 样本均值
            - "sample_std": 样本标准差
        """
        sample_size = len(data)

        # Shapiro-Wilk 检验（适用于 n < 5000）
        if sample_size < 5000:
            shapiro_stat, shapiro_p = shapiro(data)
        else:
            # 大样本时随机子采样
            indices = np.random.choice(sample_size, 5000, replace=False)
            shapiro_stat, shapiro_p = shapiro(data[indices])

        # D'Agostino-Pearson 检验（综合偏度和峰度）
        dagostino_stat, dagostino_p = normaltest(data)

        # 偏度和峰度
        skewness = float(np.mean(((data - np.mean(data)) / np.std(data)) ** 3))
        kurtosis = float(np.mean(((data - np.mean(data)) / np.std(data)) ** 4) - 3)

        # 综合判断：两个检验都通过
        is_normal = bool(shapiro_p >= alpha and dagostino_p >= alpha)

        self.results = {
            "shapiro_stat": float(shapiro_stat),
            "shapiro_p": float(shapiro_p),
            "dagostino_stat": float(dagostino_stat),
            "dagostino_p": float(dagostino_p),
            "skewness": skewness,
            "kurtosis": kurtosis,
            "alpha": alpha,
            "is_normal": is_normal,
            "sample_mean": float(np.mean(data)),
            "sample_std": float(np.std(data)),
            "sample_size": sample_size,
        }

        return self.results

    def summary(self) -> str:
        """返回正态分布拟合检验结果摘要"""
        if not self.results:
            return "尚未进行分析"

        r = self.results
        conclusion = "接受 H0：数据服从正态分布" if r["is_normal"] else "拒绝 H0：数据不服从正态分布"

        summary = (
            f"\n{'='*60}\n"
            f"  工具7：正态分布拟合检验\n"
            f"{'='*60}\n"
            f"  原假设 H0: 数据服从正态分布\n"
            f"  显著性水平 α = {r['alpha']}\n"
            f"  {'─'*50}\n"
            f"  Shapiro-Wilk 检验:\n"
            f"    W统计量 = {r['shapiro_stat']:.6f}\n"
            f"    p-value = {r['shapiro_p']:.6f}\n"
            f"  D'Agostino-Pearson 检验:\n"
            f"    统计量 = {r['dagostino_stat']:.4f}\n"
            f"    p-value = {r['dagostino_p']:.6f}\n"
            f"  {'─'*50}\n"
            f"  样本均值: {r['sample_mean']:.4f} (目标: --)\n"
            f"  样本标准差: {r['sample_std']:.4f} (目标: --)\n"
            f"  偏度: {r['skewness']:.4f} (正态=0)\n"
            f"  峰度: {r['kurtosis']:.4f} (正态=0)\n"
            f"  {'─'*50}\n"
            f"  结论: {conclusion}\n"
            f"{'='*60}\n"
        )
        return summary
