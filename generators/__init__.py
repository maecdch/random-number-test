"""
随机数生成器模块
包含 rand()%N 模拟、均匀分布生成器、正态分布生成器
"""

from .basic_rand import BasicRand
from .uniform import UniformGenerator
from .normal import NormalGenerator

__all__ = ["BasicRand", "UniformGenerator", "NormalGenerator"]
