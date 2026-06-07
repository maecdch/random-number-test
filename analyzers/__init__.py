"""
测评工具模块

包含6种针对 rand()%N 的测评工具 + 1种正态分布拟合检验：
1. 频率分布统计 (FreqDistribution)
2. 重复间隔分布统计 (RepeatInterval)
3. 卡方检验 (ChiSquareTest)
4. 游程检验 (RunsTest)
5. KS检验 (KSTest)
6. 自相关检验 (AutocorrelationTest)
7. 正态分布拟合检验 (NormalFitTest) - 辅助工具
"""

from .freq_distribution import FreqDistribution
from .repeat_interval import RepeatInterval
from .chi_square_test import ChiSquareTest
from .runs_test import RunsTest
from .ks_test import KSTest
from .autocorrelation import AutocorrelationTest
from .normal_fit_test import NormalFitTest

__all__ = [
    "FreqDistribution",
    "RepeatInterval",
    "ChiSquareTest",
    "RunsTest",
    "KSTest",
    "AutocorrelationTest",
    "NormalFitTest",
]
