# 随机数的统计分布测试

《密码学导论》课程大作业 — 测评 `rand() % N` 方法生成的随机数分布质量。

## 项目结构

```
random-number-test/
├── generators/               # 随机数生成器
│   ├── basic_rand.py         #   C风格 rand()%N 模拟
│   ├── uniform.py            #   均匀分布生成器（4种方法）
│   └── normal.py             #   正态分布生成器（Box-Muller / CLT）
├── analyzers/                # 测评工具（7种）
│   ├── freq_distribution.py  # ① 频率分布统计
│   ├── repeat_interval.py    # ② 重复间隔分布统计
│   ├── chi_square_test.py    # ③ 卡方拟合优度检验
│   ├── runs_test.py          # ④ 游程检验
│   ├── ks_test.py            # ⑤ Kolmogorov-Smirnov 检验
│   ├── autocorrelation.py    # ⑥ 自相关检验 (Ljung-Box Q)
│   └── normal_fit_test.py    # ⑦ 正态分布拟合检验
├── visualizer.py             # 可视化（直方图、QQ图、ACF图等）
├── main.py                   # 主程序入口
├── config.py                 # 全局配置
├── reports/                  # 输出结果
│   ├── figures/              # 图表
│   └── test_results/         # 数据
└── requirements.txt          # 依赖清单
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行完整测试
python main.py

# 自定义参数
python main.py --N 50 --size 5000

# 导出CSV数据
python main.py --export-csv

# 导出中科国昱密评工具箱格式
python main.py --export-toolkit
```

## 7种测评工具

| # | 工具 | 原理 | 检测目标 |
|---|------|------|---------|
| 1 | 频率分布统计 | 计数 + 频率计算 | 各数值出现概率是否均匀 |
| 2 | 重复间隔分布 | 间隔统计 + 直方图 | 重复出现间隔的规律性 |
| 3 | 卡方检验 | χ² = Σ(O-E)²/E | 分布拟合优度 |
| 4 | 游程检验 | Wald-Wolfowitz | 序列独立性 |
| 5 | KS检验 | D = max\|F-F₀\| | 经验分布 vs 理论分布 |
| 6 | 自相关检验 | Ljung-Box Q | 序列相关性 |
| 7 | 正态拟合检验 | Shapiro-Wilk | 正态分布生成器验证 |

## 中科国昱密评工具箱

使用 `--export-toolkit` 参数可导出多种格式（十进制、十六进制、二进制、CSV），供中科国昱密评工具箱进行随机性测评。
