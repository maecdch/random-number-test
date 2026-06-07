"""
visualizer.py - 可视化模块

生成所有测评结果的可视化图表，用于报告展示。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, Optional

# 设置中文字体 - 避免使用 SimHei 不支持的字符
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """
    可视化生成器

    为每个测评工具生成对应的图表，保存为PNG文件。
    """

    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_freq_distribution(self, freq_results: Dict, filename: str = "01_freq_distribution.png"):
        """绘制频率分布直方图"""
        N = freq_results["N"]
        frequencies = freq_results["frequencies"]
        expected = freq_results["expected_freq"]

        fig, ax = plt.subplots(figsize=(12, 5))

        # 实际频次
        ax.bar(range(N), frequencies, alpha=0.7, label='实际频次', color='steelblue')
        # 期望频次
        ax.axhline(y=expected, color='red', linestyle='--', linewidth=1.5,
                   label=f'期望频次 ({expected:.1f})')

        ax.set_xlabel('数值')
        ax.set_ylabel('频次')
        ax.set_title(f'频率分布图 (N={N}, 样本量={freq_results["sample_size"]})')
        ax.legend()
        ax.set_xlim(-0.5, N - 0.5)

        # 标注偏差最大的几个值
        deviations = np.abs(frequencies - expected)
        top_dev_indices = np.argsort(deviations)[-5:]
        for idx in top_dev_indices:
            ax.annotate(f'{int(frequencies[idx])}',
                        xy=(idx, frequencies[idx]),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=8, color='darkred')

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def plot_repeat_interval(self, interval_results: Dict, filename: str = "02_repeat_interval.png"):
        """绘制重复间隔分布直方图"""
        all_intervals = interval_results["all_intervals"]
        max_interval = interval_results["max_interval"]

        if len(all_intervals) == 0:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 图1：间隔分布直方图
        axes[0].hist(all_intervals, bins=min(50, max_interval),
                     alpha=0.7, color='steelblue', edgecolor='white')
        axes[0].axvline(x=interval_results["global_mean_interval"],
                        color='red', linestyle='--', linewidth=1.5,
                        label=f'均值={interval_results["global_mean_interval"]:.1f}')
        axes[0].axvline(x=interval_results["global_median_interval"],
                        color='green', linestyle=':', linewidth=1.5,
                        label=f'中位数={interval_results["global_median_interval"]:.1f}')
        axes[0].set_xlabel('重复间隔')
        axes[0].set_ylabel('频次')
        axes[0].set_title('重复间隔分布')
        axes[0].legend()

        # 图2：间隔分布箱线图（按数值分组，取前20个值）
        N = interval_results["N"]
        intervals_by_value = interval_results["intervals_by_value"]

        # 选择出现频率最高的20个数值
        value_interval_lengths = {v: len(iv) for v, iv in intervals_by_value.items()}
        top_values = sorted(value_interval_lengths.keys(),
                            key=lambda v: value_interval_lengths[v], reverse=True)[:20]

        data_for_box = [intervals_by_value[v] for v in top_values]
        if any(len(d) > 0 for d in data_for_box):
            bp = axes[1].boxplot(data_for_box, labels=top_values, showfliers=False)
            axes[1].set_xlabel('数值')
            axes[1].set_ylabel('重复间隔')
            axes[1].set_title('各数值重复间隔分布 (Top 20)')
            axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def plot_chi_square(self, chi_results: Dict, freq_results: Dict,
                        filename: str = "03_chi_square.png"):
        """绘制卡方检验可视化图"""
        N = chi_results["N"]
        frequencies = freq_results["frequencies"]
        expected_freq = freq_results["expected_freq"]

        fig, ax = plt.subplots(figsize=(12, 5))

        # 计算每个类别的卡方贡献
        chi_contrib = (frequencies - expected_freq) ** 2 / expected_freq

        colors = ['#e74c3c' if c > chi_results["critical_value"] / N
                  else '#3498db' for c in chi_contrib]

        ax.bar(range(N), chi_contrib, color=colors, alpha=0.7, edgecolor='white')
        ax.axhline(y=chi_results["chi2_stat"] / N, color='red',
                   linestyle='--', linewidth=1,
                   label=f'平均贡献 (χ²/N={chi_results["chi2_stat"]/N:.4f})')

        ax.set_xlabel('数值')
        ax.set_ylabel('卡方贡献 (O-E)²/E')
        ax.set_title(f'卡方检验 - 各数值贡献度 (χ²={chi_results["chi2_stat"]:.2f}, '
                     f'p={chi_results["p_value"]:.4f})')
        ax.legend()

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def plot_runs_test(self, data: np.ndarray, runs_results: Dict,
                       filename: str = "04_runs_test.png"):
        """绘制游程检验可视化图"""
        sample_size = len(data)
        median = float(np.median(data))

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # 图1：序列变化图（取前200个样本）
        subset = data[:min(200, sample_size)]
        axes[0].plot(subset, 'o-', markersize=3, linewidth=0.5, alpha=0.7, color='steelblue')
        axes[0].axhline(y=median, color='red', linestyle='--',
                        label=f'中位数={median:.1f}')
        axes[0].fill_between(range(len(subset)), median, subset,
                             where=(subset > median), color='green', alpha=0.1)
        axes[0].fill_between(range(len(subset)), median, subset,
                             where=(subset <= median), color='red', alpha=0.1)
        axes[0].set_xlabel('位置')
        axes[0].set_ylabel('数值')
        axes[0].set_title('序列变化图（前200个样本，绿色=高于中位数，红色=低于中位数）')
        axes[0].legend()

        # 图2：游程数量对比
        categories = ['实际游程数', '期望游程数']
        values = [runs_results["runs_count"], runs_results["expected_runs"]]
        colors_bar = ['#3498db' if runs_results["is_random"] else '#e74c3c', '#95a5a6']
        axes[1].bar(categories, values, color=colors_bar, alpha=0.7, width=0.4)
        axes[1].set_ylabel('游程数量')
        axes[1].set_title(f'游程检验 (Z={runs_results["z_stat"]:.2f}, '
                          f'p={runs_results["p_value"]:.4f})')

        # 标注判断结果
        result_text = '[通过] 随机性通过' if runs_results["is_random"] else '[未通过] 随机性未通过'
        axes[1].text(0.5, 0.95, result_text, transform=axes[1].transAxes,
                     ha='center', fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def plot_ks_test(self, data: np.ndarray, ks_results: Dict, N: int,
                     filename: str = "05_ks_test.png"):
        """绘制KS检验可视化图"""
        sample_size = len(data)

        fig, ax = plt.subplots(figsize=(10, 6))

        # 经验分布函数
        sorted_data = np.sort(data)
        ecdf = np.arange(1, sample_size + 1) / sample_size

        # 理论分布函数（均匀分布）
        tcdf_x = np.linspace(0, N - 1, 1000)
        tcdf_y = (tcdf_x + 1) / N

        # 绘制
        ax.step(sorted_data, ecdf, where='post', label='经验分布函数 (ECDF)',
                color='steelblue', linewidth=1.5)
        ax.plot(tcdf_x, tcdf_y, '--', label='理论分布函数 (均匀分布)',
                color='red', linewidth=1.5)

        # 标注最大偏差位置
        d_stat = ks_results["d_stat"]
        ax.annotate(f'D = {d_stat:.4f}',
                    xy=(0.7, 0.3), xycoords='axes fraction',
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        ax.set_xlabel('数值')
        ax.set_ylabel('累积概率')
        ax.set_title(f'Kolmogorov-Smirnov 检验 (D={d_stat:.4f}, '
                     f'p={ks_results["p_value"]:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def plot_autocorrelation(self, acf_results: Dict,
                             filename: str = "06_autocorrelation.png"):
        """绘制自相关函数图"""
        autocorrelations = acf_results["autocorrelations"]
        max_lag = acf_results["max_lag"]
        threshold = acf_results["threshold"]

        fig, ax = plt.subplots(figsize=(12, 5))

        lags = range(1, max_lag + 1)
        acf_values = autocorrelations[1:]

        # 绘制自相关系数柱状图
        colors = ['#e74c3c' if abs(v) > threshold else '#3498db' for v in acf_values]
        ax.bar(lags, acf_values, width=0.6, color=colors, alpha=0.7, edgecolor='white')

        # 置信区间
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=0.8,
                   label=f'95%置信上限 (+{threshold:.3f})')
        ax.axhline(y=-threshold, color='red', linestyle='--', linewidth=0.8,
                   label=f'95%置信下限 (-{threshold:.3f})')
        ax.axhline(y=0, color='black', linewidth=0.5)

        ax.set_xlabel('滞后阶数')
        ax.set_ylabel('自相关系数')
        ax.set_title(f'自相关函数图 (Q={acf_results["q_stat"]:.2f}, '
                     f'p={acf_results["p_value"]:.4f})')
        ax.legend()
        ax.set_xlim(0, max_lag + 1)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def plot_normal_qq(self, normal_data: np.ndarray, normal_results: Dict,
                       filename: str = "07_normal_qq.png"):
        """绘制正态Q-Q图"""
        import scipy.stats as stats

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 图1：Q-Q图
        stats.probplot(normal_data, dist="norm", plot=axes[0])
        axes[0].set_title('Q-Q 图')
        axes[0].get_lines()[0].set_marker('o')
        axes[0].get_lines()[0].set_markersize(3)
        axes[0].get_lines()[0].set_alpha(0.6)

        # 图2：直方图 + 拟合正态曲线
        axes[1].hist(normal_data, bins=50, density=True, alpha=0.7,
                     color='steelblue', edgecolor='white')

        # 拟合正态分布曲线
        mu, sigma = np.mean(normal_data), np.std(normal_data)
        x = np.linspace(min(normal_data), max(normal_data), 100)
        y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        axes[1].plot(x, y, 'r-', linewidth=2, label=f'N({mu:.1f}, {sigma:.1f}²)')
        axes[1].set_xlabel('数值')
        axes[1].set_ylabel('概率密度')
        axes[1].set_title('分布直方图与拟合曲线')
        axes[1].legend()

        # 标注检验结果
        if normal_results:
            result_text = '[通过] 正态分布通过' if normal_results.get("is_normal") else '[未通过] 正态分布未通过'
            fig.suptitle(f'正态分布拟合检验 (Shapiro-Wilk p={normal_results["shapiro_p"]:.4f})',
                         fontsize=12)
            axes[1].text(0.5, 0.95, result_text, transform=axes[1].transAxes,
                         ha='center', fontsize=11,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def plot_all_comparison(self, all_results: Dict, filename: str = "08_all_comparison.png"):
        """绘制所有测评结果的汇总对比图"""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')

        # 构建汇总表格
        table_data = []
        col_labels = ['测评工具', '统计量', 'p-value', '结论']

        for tool_name, result in all_results.items():
            if tool_name == "frequency":
                table_data.append([
                    '① 频率分布统计',
                    f'最大偏差={result["max_deviation"]:.4f}',
                    '--',
                    '--'
                ])
            elif tool_name == "repeat_interval":
                table_data.append([
                    '② 重复间隔分布',
                    f'平均间隔={result["global_mean_interval"]:.1f}',
                    '--',
                    '--'
                ])
            elif tool_name == "chi_square":
                table_data.append([
                    '③ 卡方检验',
                    f'X2={result["chi2_stat"]:.2f}',
                    f'{result["p_value"]:.4f}',
                    '[通过] 均匀' if result["is_uniform"] else '[未通过] 非均匀'
                ])
            elif tool_name == "runs_test":
                table_data.append([
                    '④ 游程检验',
                    f'Z={result["z_stat"]:.2f}',
                    f'{result["p_value"]:.4f}',
                    '[通过] 随机' if result["is_random"] else '[未通过] 非随机'
                ])
            elif tool_name == "ks_test":
                table_data.append([
                    '⑤ KS检验',
                    f'D={result["d_stat"]:.4f}',
                    f'{result["p_value"]:.4f}',
                    '[通过] 均匀' if result["is_uniform"] else '[未通过] 非均匀'
                ])
            elif tool_name == "autocorrelation":
                table_data.append([
                    '⑥ 自相关检验',
                    f'Q={result["q_stat"]:.2f}',
                    f'{result["p_value"]:.4f}',
                    '[通过] 独立' if result["is_independent"] else '[未通过] 相关'
                ])

        table = ax.table(cellText=table_data, colLabels=col_labels,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        ax.set_title('测评结果汇总', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
