"""
normal.py - 正态分布随机数生成器

实现多种方法生成正态分布的随机数，并转换为 [0, N-1] 范围：
1. Box-Muller 变换法
2. 中心极限定理法（CLT）
"""

import numpy as np
from typing import Optional


class NormalGenerator:
    """
    正态分布随机数生成器
    
    设计公式/函数获得正态分布随机数，并映射到 [0, N-1] 范围
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def box_muller(size: int, mu: float = 0.0, sigma: float = 1.0,
                   seed: int = 42) -> np.ndarray:
        """
        Box-Muller 变换法生成正态分布随机数
        
        原理：
        设 U1, U2 是 (0,1] 上的独立均匀分布随机数，则：
            Z0 = sqrt(-2 * ln(U1)) * cos(2 * pi * U2)
            Z1 = sqrt(-2 * ln(U1)) * sin(2 * pi * U2)
        Z0, Z1 服从标准正态分布 N(0,1)
        
        再通过线性变换：X = mu + sigma * Z 得到 N(mu, sigma^2)
        
        Args:
            size: 样本数量
            mu: 均值
            sigma: 标准差
            seed: 随机种子
        
        Returns:
            正态分布随机数数组
        """
        import random as std_random
        std_random.seed(seed)

        # Box-Muller 每次生成2个独立样本
        result = np.empty(size)
        for i in range(0, size, 2):
            u1 = std_random.random()
            # 避免 u1 = 0 导致 log(0)
            while u1 == 0.0:
                u1 = std_random.random()
            u2 = std_random.random()

            z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
            z1 = np.sqrt(-2.0 * np.log(u1)) * np.sin(2.0 * np.pi * u2)

            result[i] = mu + sigma * z0
            if i + 1 < size:
                result[i + 1] = mu + sigma * z1

        return result

    @staticmethod
    def central_limit_theorem(size: int, mu: float = 0.0, sigma: float = 1.0,
                              n_samples: int = 12, seed: int = 42) -> np.ndarray:
        """
        中心极限定理法生成正态分布随机数
        
        原理：
        根据中心极限定理，n个独立同分布的随机变量之和近似服从正态分布。
        对于 [0,1) 均匀分布 U_i，有：
            E[U_i] = 1/2, Var[U_i] = 1/12
            令 Z = (sum(U_i) - n/2) / sqrt(n/12)
        Z 近似服从 N(0,1)
        
        当 n=12 时，分母 sqrt(12/12) = 1，公式简化为：
            Z = sum(U_i) - 6
        
        Args:
            size: 样本数量
            mu: 均值
            sigma: 标准差
            n_samples: 每次求和使用的均匀随机数个数（推荐12）
            seed: 随机种子
        
        Returns:
            正态分布随机数数组
        """
        import random as std_random
        std_random.seed(seed)

        result = np.empty(size)
        for i in range(size):
            # 生成 n_samples 个 [0,1) 均匀随机数并求和
            s = sum(std_random.random() for _ in range(n_samples))
            # 标准化
            z = (s - n_samples / 2.0) / np.sqrt(n_samples / 12.0)
            result[i] = mu + sigma * z

        return result

    def generate_uniform_mapped(self, N: int, size: int,
                                method: str = "box_muller",
                                mu: Optional[float] = None,
                                sigma: Optional[float] = None,
                                seed: int = 42) -> np.ndarray:
        """
        生成正态分布随机数并映射到 [0, N-1] 范围
        
        映射策略：
        1. 生成 N(mu, sigma^2) 随机数
        2. 截断到 [mu - 3*sigma, mu + 3*sigma] 范围
        3. 线性映射到 [0, N-1] 并取整
        
        Args:
            N: 目标范围上限
            size: 样本数量
            method: 生成方法 ("box_muller" 或 "clt")
            mu: 正态分布均值（默认 N/2）
            sigma: 正态分布标准差（默认 N/6，使 99.7% 数据落在 [0, N-1]）
            seed: 随机种子
        
        Returns:
            [0, N-1] 范围内的随机整数数组
        """
        if mu is None:
            mu = (N - 1) / 2.0
        if sigma is None:
            sigma = (N - 1) / 6.0

        # 生成正态分布随机数
        if method == "box_muller":
            samples = self.box_muller(size, mu, sigma, seed)
        elif method == "clt":
            samples = self.central_limit_theorem(size, mu, sigma, seed=seed)
        else:
            raise ValueError(f"未知方法: {method}")

        # 截断到 [mu - 3*sigma, mu + 3*sigma]
        lower = mu - 3 * sigma
        upper = mu + 3 * sigma
        samples = np.clip(samples, lower, upper)

        # 线性映射到 [0, N-1] 并取整
        mapped = ((samples - lower) / (upper - lower) * (N - 1)).astype(np.int64)
        mapped = np.clip(mapped, 0, N - 1)

        return mapped

    def generate_raw(self, size: int, mu: float = 0.0, sigma: float = 1.0,
                     method: str = "box_muller", seed: int = 42) -> np.ndarray:
        """
        生成原始正态分布随机数（不映射）
        
        Args:
            size: 样本数量
            mu: 均值
            sigma: 标准差
            method: "box_muller" 或 "clt"
            seed: 随机种子
        
        Returns:
            正态分布浮点数数组
        """
        if method == "box_muller":
            return self.box_muller(size, mu, sigma, seed)
        elif method == "clt":
            return self.central_limit_theorem(size, mu, sigma, seed=seed)
        else:
            raise ValueError(f"未知方法: {method}")
