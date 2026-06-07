"""
全局配置文件
随机数的统计分布测试 - 配置参数
"""

# 随机数范围 [0, N-1]
N = 100

# 采样数量
SAMPLE_SIZE = 10000

# 随机数种子
SEED = 42

# 重复间隔统计的最大间隔
MAX_INTERVAL = 1000

# 自相关检验的最大滞后阶数
MAX_LAG = 50

# 显著性水平
ALPHA = 0.05

# 正态分布参数
NORMAL_MEAN = 50.0
NORMAL_STD = 15.0

# 输出路径
REPORT_DIR = "reports"
FIGURE_DIR = "reports/figures"
RESULT_DIR = "reports/test_results"

# 中科国昱工具箱输出格式
TOOLKIT_OUTPUT_FILE = "reports/test_results/toolkit_input.txt"
