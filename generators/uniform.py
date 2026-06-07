"""
uniform.py - [0, N-1] 均匀分布随机数生成器

实现多种方法生成均匀分布随机整数：
1. rand() % N 法（C语言风格）
2. int(random() * N) 法（浮点乘法律）
3. 拒绝采样法
4. 线性同余法（LCG）
"""

import numpy as np
from typing import Callable, Optional


class UniformGenerator:
    """
    [0, N-1] 均匀分布随机整数生成器
    
    提供多种生成方法，支持对比分析不同方法的分布特性
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def method_modulo(N: int, size: int, seed: int = 42) -> np.ndarray:
        """
        方法1：rand() % N 法（C语言风格）
        
        原理：先生成 [0, RAND_MAX] 范围的整数，再对N取模
        
        缺陷：当 RAND_MAX+1 不能被 N 整除时，会产生偏差
        低位的随机性较差（LCG的低位周期更短）
        
        Args:
            N: 范围上限
            size: 序列长度
            seed: 随机种子
        
        Returns:
            [0, N-1] 范围内的随机整数数组
        """
        from .basic_rand import BasicRand
        generator = BasicRand(seed)
        return generator.generate_sequence(size, N)

    @staticmethod
    def method_float_multiply(N: int, size: int, seed: int = 42) -> np.ndarray:
        """
        方法2：int(random() * N) 法（浮点乘法律）
        
        原理：先生成 [0, 1) 浮点数，再乘以N取整
        
        优点：分布更均匀，没有模运算的偏差
        
        Args:
            N: 范围上限
            size: 序列长度
            seed: 随机种子
        
        Returns:
            [0, N-1] 范围内的随机整数数组
        """
        import random as std_random
        std_random.seed(seed)
        return np.array([int(std_random.random() * N) for _ in range(size)], dtype=np.int64)

    @staticmethod
    def method_rejection_sampling(N: int, size: int, seed: int = 42) -> np.ndarray:
        """
        方法3：拒绝采样法
        
        原理：生成 [0, 2^k-1] 范围的随机数（其中 2^k > N），
        如果结果 >= N 则拒绝并重新采样，直到落在 [0, N-1] 范围内
        
        优点：无模运算偏差，分布完全均匀
        
        Args:
            N: 范围上限
            size: 序列长度
            seed: 随机种子
        
        Returns:
            [0, N-1] 范围内的随机整数数组
        """
        import random as std_random
        std_random.seed(seed)

        # 计算需要的比特数 k，使得 2^k >= N
        k = 1
        while (1 << k) < N:
            k += 1
        mask = (1 << k) - 1

        result = np.empty(size, dtype=np.int64)
        for i in range(size):
            # 生成 [0, 2^k-1] 的随机数，拒绝 >= N 的值
            while True:
                # 使用 Python 的 getrandbits 生成 k 位随机数
                val = std_random.getrandbits(k)
                if val < N:
                    result[i] = val
                    break

        return result

    @staticmethod
    def method_lcg(N: int, size: int, seed: int = 42,
                   multiplier: int = 1664525,
                   increment: int = 1013904223,
                   modulus: int = 2 ** 32) -> np.ndarray:
        """
        方法4：线性同余法（自定义LCG参数）
        
        使用不同参数的LCG生成器，观察参数对随机性的影响
        
        Args:
            N: 范围上限
            size: 序列长度
            seed: 随机种子
            multiplier: 乘数
            increment: 增量
            modulus: 模数
        
        Returns:
            [0, N-1] 范围内的随机整数数组
        """
        result = np.empty(size, dtype=np.int64)
        state = seed
        for i in range(size):
            state = (state * multiplier + increment) % modulus
            result[i] = state % N
        return result

    def generate(self, N: int, size: int,
                 method: str = "modulo",
                 seed: int = 42) -> np.ndarray:
        """
        统一的生成接口
        
        Args:
            N: 范围上限
            size: 序列长度
            method: 生成方法 ("modulo", "float_multiply", "rejection", "lcg")
            seed: 随机种子
        
        Returns:
            [0, N-1] 范围内的随机整数数组
        """
        methods = {
            "modulo": self.method_modulo,
            "float_multiply": self.method_float_multiply,
            "rejection": self.method_rejection_sampling,
            "lcg": self.method_lcg,
        }

        if method not in methods:
            raise ValueError(f"未知方法: {method}，可选: {list(methods.keys())}")

        return methods[method](N, size, seed)
