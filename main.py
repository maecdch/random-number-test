"""
main.py - 随机数的统计分布测试 主程序入口

运行此程序将执行所有测评工具并生成可视化结果。

用法:
    python main.py                     # 使用默认配置
    python main.py --N 50 --size 5000  # 自定义参数
    python main.py --no-plot           # 不生成图表
"""

import os
import sys
import io

# 解决 Windows 终端 GBK 编码无法输出 Unicode 的问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import json
import time
import numpy as np

# 确保可以导入本项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from generators.basic_rand import BasicRand, PythonRandWrapper, NumPyRandWrapper
from generators.uniform import UniformGenerator
from generators.normal import NormalGenerator
from analyzers.freq_distribution import FreqDistribution
from analyzers.repeat_interval import RepeatInterval
from analyzers.chi_square_test import ChiSquareTest
from analyzers.runs_test import RunsTest
from analyzers.ks_test import KSTest
from analyzers.autocorrelation import AutocorrelationTest
from analyzers.normal_fit_test import NormalFitTest
from visualizer import Visualizer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='随机数的统计分布测试工具集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                         # 默认配置运行
  python main.py --N 50 --size 5000      # 自定义N和样本量
  python main.py --no-plot               # 不生成图表
  python main.py --export-csv            # 导出结果为CSV
        """
    )
    parser.add_argument('--N', type=int, default=config.N,
                        help=f'随机数范围 [0, N-1] (默认: {config.N})')
    parser.add_argument('--size', type=int, default=config.SAMPLE_SIZE,
                        help=f'样本数量 (默认: {config.SAMPLE_SIZE})')
    parser.add_argument('--seed', type=int, default=config.SEED,
                        help=f'随机种子 (默认: {config.SEED})')
    parser.add_argument('--no-plot', action='store_true',
                        help='不生成图表')
    parser.add_argument('--export-csv', action='store_true',
                        help='导出结果为CSV文件')
    parser.add_argument('--export-toolkit', action='store_true',
                        help='导出数据为 中科国昱密评工具箱 格式')
    return parser.parse_args()


def run_rand_mod_N_test(N: int, size: int, seed: int, no_plot: bool, export_csv: bool):
    """
    执行 rand()%N 的完整测评流程
    
    Args:
        N: 随机数范围
        size: 样本数量
        seed: 随机种子
        no_plot: 是否不生成图表
        export_csv: 是否导出CSV
    """
    print(f"\n{'='*70}")
    print(f"  《随机数的统计分布测试》")
    print(f"  测试对象: rand() % N")
    print(f"  参数: N={N}, 样本量={size}, 种子={seed}")
    print(f"{'='*70}")

    # ========== 生成随机序列 ==========
    print(f"\n  ▶ 生成随机序列...")
    generator = BasicRand(seed)
    data = generator.generate_sequence(size, N)
    print(f"  序列已生成，前20个值: {data[:20]}")

    # ========== 工具1：频率分布统计 ==========
    print(f"\n  ▶ 工具1：频率分布统计...")
    freq_analyzer = FreqDistribution()
    freq_results = freq_analyzer.analyze(data, N)
    print(freq_analyzer.summary())

    # ========== 工具2：重复间隔分布统计 ==========
    print(f"\n  ▶ 工具2：重复间隔分布统计...")
    interval_analyzer = RepeatInterval()
    interval_results = interval_analyzer.analyze(data, N,
                                                  max_interval=config.MAX_INTERVAL)
    print(interval_analyzer.summary())

    # ========== 工具3：卡方检验 ==========
    print(f"\n  ▶ 工具3：卡方拟合优度检验...")
    chi_analyzer = ChiSquareTest()
    chi_results = chi_analyzer.analyze(data, N, alpha=config.ALPHA)
    print(chi_analyzer.summary())

    # ========== 工具4：游程检验 ==========
    print(f"\n  ▶ 工具4：游程检验...")
    runs_analyzer = RunsTest()
    runs_results = runs_analyzer.analyze(data, N, alpha=config.ALPHA)
    print(runs_analyzer.summary())

    # ========== 工具5：KS检验 ==========
    print(f"\n  ▶ 工具5：Kolmogorov-Smirnov 检验...")
    ks_analyzer = KSTest()
    ks_results = ks_analyzer.analyze(data, N, alpha=config.ALPHA)
    print(ks_analyzer.summary())

    # ========== 工具6：自相关检验 ==========
    print(f"\n  ▶ 工具6：自相关检验...")
    acf_analyzer = AutocorrelationTest()
    acf_results = acf_analyzer.analyze(data, N, max_lag=config.MAX_LAG,
                                        alpha=config.ALPHA)
    print(acf_analyzer.summary())

    # ========== 生成可视化图表 ==========
    all_results = {
        "frequency": freq_results,
        "repeat_interval": interval_results,
        "chi_square": chi_results,
        "runs_test": runs_results,
        "ks_test": ks_results,
        "autocorrelation": acf_results,
    }

    if not no_plot:
        print(f"\n  ▶ 生成可视化图表...")
        viz = Visualizer(config.FIGURE_DIR)

        path1 = viz.plot_freq_distribution(freq_results)
        print(f"  已保存: {path1}")

        path2 = viz.plot_repeat_interval(interval_results)
        if path2:
            print(f"  已保存: {path2}")

        path3 = viz.plot_chi_square(chi_results, freq_results)
        print(f"  已保存: {path3}")

        path4 = viz.plot_runs_test(data, runs_results)
        print(f"  已保存: {path4}")

        path5 = viz.plot_ks_test(data, ks_results, N)
        print(f"  已保存: {path5}")

        path6 = viz.plot_autocorrelation(acf_results)
        print(f"  已保存: {path6}")

        path8 = viz.plot_all_comparison(all_results)
        print(f"  已保存: {path8}")

    # ========== 导出CSV ==========
    if export_csv:
        print(f"\n  ▶ 导出CSV...")
        os.makedirs(config.RESULT_DIR, exist_ok=True)

        # 频率分布
        freq_df = freq_analyzer.to_dataframe()
        csv_path1 = os.path.join(config.RESULT_DIR, "freq_distribution.csv")
        freq_df.to_csv(csv_path1, index=False, encoding='utf-8-sig')
        print(f"  已导出: {csv_path1}")

        # 重复间隔（前N个值）
        interval_df = interval_analyzer.to_dataframe(N)
        csv_path2 = os.path.join(config.RESULT_DIR, "repeat_interval.csv")
        interval_df.to_csv(csv_path2, index=False, encoding='utf-8-sig')
        print(f"  已导出: {csv_path2}")

    return all_results


def run_uniform_generator_test(N: int, size: int, seed: int, no_plot: bool):
    """
    测试均匀分布生成器（多种方法对比）
    """
    print(f"\n{'='*70}")
    print(f"  【扩展测试】均匀分布随机数生成器对比")
    print(f"{'='*70}")

    methods = {
        "modulo": "rand() % N (C风格)",
        "float_multiply": "int(random() * N) (浮点乘法)",
        "rejection": "拒绝采样法",
        "lcg": "线性同余法 (自定义参数)",
    }

    for method_key, method_name in methods.items():
        print(f"\n  ▶ 方法: {method_name}")
        data = UniformGenerator.method_lcg(N, size, seed) if method_key == "lcg" else \
               UniformGenerator.method_rejection_sampling(N, size, seed) if method_key == "rejection" else \
               UniformGenerator.method_float_multiply(N, size, seed) if method_key == "float_multiply" else \
               UniformGenerator.method_modulo(N, size, seed)

        # 频率统计
        freq = FreqDistribution()
        result = freq.analyze(data, N)
        print(f"    最大偏差: {result['max_deviation']:.6f}")
        print(f"    平均偏差: {result['mean_deviation']:.6f}")

        # 卡方检验
        chi = ChiSquareTest()
        chi_result = chi.analyze(data, N)
        print(f"    χ²={chi_result['chi2_stat']:.2f}, p={chi_result['p_value']:.4f}, "
              f"{'均匀' if chi_result['is_uniform'] else '不均匀'}")

    return True


def run_normal_generator_test(N: int, size: int, seed: int, no_plot: bool):
    """
    测试正态分布生成器
    """
    print(f"\n{'='*70}")
    print(f"  【扩展测试】正态分布随机数生成器")
    print(f"{'='*70}")

    normal_gen = NormalGenerator(seed)

    # 测试1：Box-Muller 法
    print(f"\n  ▶ Box-Muller 变换法")
    raw_data_bm = normal_gen.generate_raw(size, mu=0, sigma=1, method="box_muller", seed=seed)
    mapped_bm = normal_gen.generate_uniform_mapped(N, size, method="box_muller", seed=seed)

    normal_fit = NormalFitTest()
    bm_result = normal_fit.analyze(raw_data_bm)
    print(f"    偏度={bm_result['skewness']:.4f}, 峰度={bm_result['kurtosis']:.4f}")
    print(f"    Shapiro-Wilk p={bm_result['shapiro_p']:.6f}")
    print(f"    {'✅ 通过正态检验' if bm_result['is_normal'] else '❌ 未通过正态检验'}")

    # 测试2：CLT法
    print(f"\n  ▶ 中心极限定理法 (n=12)")
    raw_data_clt = normal_gen.generate_raw(size, mu=0, sigma=1, method="clt", seed=seed)
    mapped_clt = normal_gen.generate_uniform_mapped(N, size, method="clt", seed=seed)

    clt_result = normal_fit.analyze(raw_data_clt)
    print(f"    偏度={clt_result['skewness']:.4f}, 峰度={clt_result['kurtosis']:.4f}")
    print(f"    Shapiro-Wilk p={clt_result['shapiro_p']:.6f}")
    print(f"    {'✅ 通过正态检验' if clt_result['is_normal'] else '❌ 未通过正态检验'}")

    # 可视化
    if not no_plot:
        viz = Visualizer(config.FIGURE_DIR)
        path = viz.plot_normal_qq(raw_data_bm, bm_result)
        print(f"\n  正态分布Q-Q图已保存: {path}")

    return raw_data_bm, raw_data_clt


def export_toolkit_format(data: np.ndarray, N: int, filepath: str):
    """
    导出数据为 中科国昱密评工具箱 可读格式
    
    中科国昱工具箱通常接受以下格式之一：
    1. 二进制格式（每字节一个随机数）
    2. 文本格式（每行一个十六进制数）
    3. CSV格式（逗号分隔的整数）
    
    这里导出多种格式供工具箱测试使用。
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # 格式1：文本格式（每行一个十进制数）
    txt_path = filepath.replace('.txt', '_decimal.txt')
    np.savetxt(txt_path, data, fmt='%d', delimiter='\n')
    print(f"  已导出(十进制): {txt_path}")

    # 格式2：十六进制格式
    hex_path = filepath.replace('.txt', '_hex.txt')
    with open(hex_path, 'w', encoding='utf-8') as f:
        for val in data:
            f.write(f"{val:02X}\n")
    print(f"  已导出(十六进制): {hex_path}")

    # 格式3：二进制格式
    bin_path = filepath.replace('.txt', '.bin')
    data.astype(np.uint8).tofile(bin_path)
    print(f"  已导出(二进制): {bin_path}")

    # 格式4：CSV格式
    csv_path = filepath.replace('.txt', '.csv')
    np.savetxt(csv_path, data, fmt='%d', delimiter=',')
    print(f"  已导出(CSV): {csv_path}")

    return txt_path, hex_path, bin_path, csv_path


def main():
    """主函数"""
    start_time = time.time()
    args = parse_args()

    # 创建输出目录
    os.makedirs(config.FIGURE_DIR, exist_ok=True)
    os.makedirs(config.RESULT_DIR, exist_ok=True)

    # ========== 第一部分：rand()%N 测评 ==========
    print(f"\n{'#'*70}")
    print(f"  ## 第一部分：rand() % N 随机数测评")
    print(f"{'#'*70}")
    all_results = run_rand_mod_N_test(
        N=args.N, size=args.size, seed=args.seed,
        no_plot=args.no_plot, export_csv=args.export_csv
    )

    # ========== 第二部分：均匀分布生成器对比 ==========
    print(f"\n{'#'*70}")
    print(f"  ## 第二部分：均匀分布生成器对比")
    print(f"{'#'*70}")
    run_uniform_generator_test(
        N=args.N, size=args.size, seed=args.seed,
        no_plot=args.no_plot
    )

    # ========== 第三部分：正态分布生成器 ==========
    print(f"\n{'#'*70}")
    print(f"  ## 第三部分：正态分布随机数生成")
    print(f"{'#'*70}")
    raw_bm, raw_clt = run_normal_generator_test(
        N=args.N, size=args.size, seed=args.seed,
        no_plot=args.no_plot
    )

    # ========== 第四部分：导出工具箱格式 ==========
    if args.export_toolkit:
        print(f"\n{'#'*70}")
        print(f"  ## 第四部分：导出 中科国昱密评工具箱 格式")
        print(f"{'#'*70}")
        generator = BasicRand(args.seed)
        data = generator.generate_sequence(args.size, args.N)
        export_toolkit_format(data, args.N, config.TOOLKIT_OUTPUT_FILE)

    # ========== 输出汇总 ==========
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  ✅ 全部测试完成！")
    print(f"  总耗时: {elapsed:.2f} 秒")
    print(f"  图表目录: {os.path.abspath(config.FIGURE_DIR)}")
    print(f"  数据目录: {os.path.abspath(config.RESULT_DIR)}")
    print(f"{'='*70}")

    # 最终结论摘要
    chi = all_results.get("chi_square", {})
    runs = all_results.get("runs_test", {})
    ks = all_results.get("ks_test", {})
    acf = all_results.get("autocorrelation", {})

    print(f"\n  📊 rand()%N 测评结论:")
    print(f"    • 卡方检验:      {'✅ 通过' if chi.get('is_uniform') else '❌ 未通过'}")
    print(f"    • 游程检验:      {'✅ 通过' if runs.get('is_random') else '❌ 未通过'}")
    print(f"    • KS检验:        {'✅ 通过' if ks.get('is_uniform') else '❌ 未通过'}")
    print(f"    • 自相关检验:    {'✅ 通过' if acf.get('is_independent') else '❌ 未通过'}")
    print()


if __name__ == "__main__":
    main()
