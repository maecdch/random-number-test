"""
basic_rand.py - 模拟C语言 rand()%N 随机数生成器

模拟C语言标准库中的 rand() 函数（线性同余生成器LCG），
并使用 % N 操作生成 [0, N-1] 范围内的随机整数。
"""

import numpy as np
from typing import List


class BasicRand:
    """
    模拟C语言 rand()%N 随机数生成器
    
    使用线性同余生成器(LCG)模拟C标准库的 rand():
        seed = (seed * 1103515245 + 12345) & 0x7fffffff
        return seed
    
    然后通过 % N 操作生成 [0, N-1] 范围内的整数。
    """

    def __init__(self, seed: int = 1):
        """
        初始化生成器
        
        Args:
            seed: 随机数种子
        """
        self._seed = seed & 0x7fffffff
        self._initial_seed = self._seed

    def _lcg_rand(self) -> int:
        """
        模拟C语言 rand() 的线性同余生成器
        
        LCG参数（与glibc相同）:
            multiplier: 1103515245
            increment: 12345
            modulus: 2^31
        
        Returns:
            [0, 2^31-1] 范围内的随机整数
        """
        self._seed = (self._seed * 1103515245 + 12345) & 0x7fffffff
        return self._seed

    def rand_int(self, N: int) -> int:
        """
        生成 [0, N-1] 范围内的随机整数
        
        模拟 rand() % N
        
        Args:
            N: 范围上限
        
        Returns:
            [0, N-1] 范围内的随机整数
        """
        return self._lcg_rand() % N

    def generate_sequence(self, size: int, N: int) -> np.ndarray:
        """
        生成随机整数序列
        
        Args:
            size: 序列长度
            N: 范围上限
        
        Returns:
            shape为(size,)的numpy数组，元素范围[0, N-1]
        """
        return np.array([self.rand_int(N) for _ in range(size)], dtype=np.int64)

    def reset(self):
        """重置生成器到初始状态"""
        self._seed = self._initial_seed

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int):
        self._seed = value & 0x7fffffff
        self._initial_seed = self._seed


class PythonRandWrapper:
    """
    Python内置 random 模块的包装器
    
    用于对比C风格 rand()%N 与 Python random 的差异
    """

    def __init__(self, seed: int = None):
        import random
        self._rng = random.Random(seed)

    def rand_int(self, N: int) -> int:
        return self._rng.randint(0, N - 1)

    def generate_sequence(self, size: int, N: int) -> np.ndarray:
        return np.array([self.rand_int(N) for _ in range(size)], dtype=np.int64)


class NumPyRandWrapper:
    """
    NumPy随机数生成器包装器
    
    用于对比C风格 rand()%N 与 NumPy 的差异
    """

    def __init__(self, seed: int = None):
        self._rng = np.random.default_rng(seed)

    def rand_int(self, N: int) -> int:
        return int(self._rng.integers(0, N))

    def generate_sequence(self, size: int, N: int) -> np.ndarray:
        return self._rng.integers(0, N, size=size)
