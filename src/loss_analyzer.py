#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss Analyzer: 计算训练损失指标
计算指标：
  1. Loss Volatility - 对数损失差分的标准差
  2. Spike Count & Spike Magnitude - 使用 3σ 规则检测尖峰
  3. Residual Std - 去趋势后的损失标准差（使用移动平均）
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class LossAnalyzer:
    """损失分析类，计算多个损失指标"""
    
    def __init__(self, csv_path: str, ma_window: int = 5):
        """
        初始化分析器
        
        Args:
            csv_path: 损失 CSV 文件路径
            ma_window: 移动平均窗口大小（用于去趋势）
        """
        self.csv_path = Path(csv_path)
        self.ma_window = ma_window
        self.df = None
        self.metrics = {}
        self._load_data()
    
    def _load_data(self):
        """加载 CSV 数据"""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        print(f"✓ 已加载 CSV 文件: {self.csv_path}")
        print(f"  数据形状: {self.df.shape}")
        print(f"  列名: {list(self.df.columns)}")
    
    def _clean_loss(self, loss_series: pd.Series) -> np.ndarray:
        """
        清理损失数据
        - 移除 NaN 值
        - 移除非正数（log 需要正数）
        
        Args:
            loss_series: 损失 Series
            
        Returns:
            清理后的 numpy 数组
        """
        loss_clean = loss_series.dropna().values
        # 移除非正数
        loss_clean = loss_clean[loss_clean > 0]
        
        if len(loss_clean) == 0:
            raise ValueError("没有有效的损失数据")
        
        return loss_clean
    
    def compute_volatility(self, loss: np.ndarray) -> float:
        """
        计算 Loss Volatility
        = log 损失差分的标准差
        
        Args:
            loss: 损失数组
            
        Returns:
            波动性指标（标准差）
        """
        log_loss = np.log(loss)
        diff_log_loss = np.diff(log_loss)
        volatility = float(np.std(diff_log_loss))
        return volatility
    
    def compute_spikes(self, loss: np.ndarray, threshold: float = 3.0) -> Dict:
        """
        计算 Spike Count & Spike Magnitude
        使用 N-sigma 规则检测离群值
        
        Args:
            loss: 损失数组
            threshold: 标准差倍数阈值（默认 3σ）
            
        Returns:
            包含尖峰统计的字典
                {
                    'spike_count': 尖峰个数,
                    'spike_indices': 尖峰索引列表,
                    'spike_magnitude_mean': 平均幅度,
                    'spike_magnitude_max': 最大幅度,
                    'spike_magnitude_std': 幅度标准差
                }
        """
        mean = np.mean(loss)
        std = np.std(loss)
        
        # 计算离群阈值
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        
        # 检测尖峰（超出阈值的点）
        spike_mask = (loss < lower_bound) | (loss > upper_bound)
        spike_indices = np.where(spike_mask)[0]
        
        # 计算尖峰幅度（绝对偏差）
        spike_magnitudes = np.abs(loss[spike_mask] - mean)
        
        result = {
            'spike_count': int(np.sum(spike_mask)),
            'spike_indices': spike_indices.tolist(),
            'spike_magnitude_mean': float(np.mean(spike_magnitudes)) if len(spike_magnitudes) > 0 else 0.0,
            'spike_magnitude_max': float(np.max(spike_magnitudes)) if len(spike_magnitudes) > 0 else 0.0,
            'spike_magnitude_std': float(np.std(spike_magnitudes)) if len(spike_magnitudes) > 1 else 0.0,
            'spike_ratio': float(len(spike_magnitudes) / len(loss)) if len(loss) > 0 else 0.0,
        }
        
        return result
    
    def _moving_average(self, loss: np.ndarray, window: int) -> np.ndarray:
        """
        计算移动平均
        
        Args:
            loss: 损失数组
            window: 窗口大小
            
        Returns:
            移动平均数组
        """
        if window <= 1:
            return loss.copy()
        
        # 使用 pandas 的 rolling 功能
        ma = pd.Series(loss).rolling(window=window, center=True, min_periods=1).mean().values
        return ma
    
    def compute_residual_std(self, loss: np.ndarray) -> Dict:
        """
        计算 Residual Std（去趋势残差标准差）
        1. 计算移动平均作为趋势
        2. 计算残差 = loss - MA
        3. 计算残差的标准差
        
        Args:
            loss: 损失数组
            
        Returns:
            包含残差统计的字典
                {
                    'residual_std': 残差标准差,
                    'residual_mean': 残差均值,
                    'residual_max': 残差最大值,
                    'detrended_loss': 去趋势后的损失
                }
        """
        ma = self._moving_average(loss, self.ma_window)
        residual = loss - ma
        
        result = {
            'residual_std': float(np.std(residual)),
            'residual_mean': float(np.mean(residual)),
            'residual_max': float(np.max(np.abs(residual))),
            'ma_window': self.ma_window,
        }
        
        return result
    
    def analyze(self) -> Dict:
        """
        执行完整分析
        
        Returns:
            分析结果字典
        """
        results = {}
        
        # 分析 loss_train
        if 'loss_train' in self.df.columns:
            print("\n【分析 loss_train】")
            loss_train = self._clean_loss(self.df['loss_train'])
            print(f"  有效样本数: {len(loss_train)}")
            
            train_metrics = {
                'sample_count': len(loss_train),
                'mean': float(np.mean(loss_train)),
                'std': float(np.std(loss_train)),
                'min': float(np.min(loss_train)),
                'max': float(np.max(loss_train)),
                'volatility': self.compute_volatility(loss_train),
                **self.compute_spikes(loss_train),
                **self.compute_residual_std(loss_train),
            }
            results['loss_train'] = train_metrics
            print(f"  ✓ 波动性: {train_metrics['volatility']:.6f}")
            print(f"  ✓ 尖峰数量: {train_metrics['spike_count']}")
            print(f"  ✓ 残差标准差: {train_metrics['residual_std']:.6f}")
        
        # 分析 loss_test
        if 'loss_test' in self.df.columns:
            print("\n【分析 loss_test】")
            loss_test = self._clean_loss(self.df['loss_test'])
            print(f"  有效样本数: {len(loss_test)}")
            
            test_metrics = {
                'sample_count': len(loss_test),
                'mean': float(np.mean(loss_test)),
                'std': float(np.std(loss_test)),
                'min': float(np.min(loss_test)),
                'max': float(np.max(loss_test)),
                'volatility': self.compute_volatility(loss_test),
                **self.compute_spikes(loss_test),
                **self.compute_residual_std(loss_test),
            }
            results['loss_test'] = test_metrics
            print(f"  ✓ 波动性: {test_metrics['volatility']:.6f}")
            print(f"  ✓ 尖峰数量: {test_metrics['spike_count']}")
            print(f"  ✓ 残差标准差: {test_metrics['residual_std']:.6f}")
        
        self.metrics = results
        return results
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        将结果转换为 DataFrame（适合 CSV 输出）
        
        Returns:
            DataFrame（行为指标，列为 loss_train/loss_test）
        """
        if not self.metrics:
            raise ValueError("请先运行 analyze() 方法")
        
        # 提取所有唯一的指标名称
        all_keys = set()
        for loss_type in self.metrics.values():
            all_keys.update(loss_type.keys())
        
        all_keys = sorted(list(all_keys))
        
        # 构建 DataFrame
        data = {}
        for key in all_keys:
            row_data = {}
            for loss_type, metrics in self.metrics.items():
                row_data[loss_type] = metrics.get(key, np.nan)
            data[key] = row_data
        
        df_result = pd.DataFrame(data).T
        return df_result
    
    def save_csv(self, output_path: str):
        """
        保存结果为 CSV
        
        Args:
            output_path: 输出 CSV 路径
        """
        df_result = self.to_dataframe()
        df_result.to_csv(output_path, encoding='utf-8')
        print(f"\n✓ 结果已保存到: {output_path}")
    
    def print_summary(self):
        """打印结果摘要"""
        if not self.metrics:
            raise ValueError("请先运行 analyze() 方法")
        
        print("\n" + "="*70)
        print("损失分析结果摘要")
        print("="*70)
        
        df_result = self.to_dataframe()
        
        # 关键指标
        key_metrics = ['volatility', 'spike_count', 'spike_ratio', 'residual_std']
        
        print("\n【关键指标】")
        print(df_result.loc[key_metrics].to_string())
        
        print("\n【详细统计】")
        print(df_result.to_string())


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='损失分析工具：计算训练损失的多个指标'
    )
    parser.add_argument(
        '--csv-path', '-c',
        type=str,
        required=True,
        help='损失 CSV 文件路径'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出 CSV 路径（可选）'
    )
    parser.add_argument(
        '--ma-window', '-w',
        type=int,
        default=5,
        help='移动平均窗口大小（默认：5）'
    )
    parser.add_argument(
        '--spike-threshold', '-t',
        type=float,
        default=3.0,
        help='尖峰检测阈值（标准差倍数，默认：3.0）'
    )
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = LossAnalyzer(args.csv_path, ma_window=args.ma_window)
    
    # 执行分析
    analyzer.analyze()
    
    # 打印摘要
    analyzer.print_summary()
    
    # 保存结果
    if args.output:
        analyzer.save_csv(args.output)
    else:
        # 默认输出文件名
        csv_stem = Path(args.csv_path).stem
        output_path = Path(args.csv_path).parent / f"{csv_stem}_metrics.csv"
        analyzer.save_csv(str(output_path))


if __name__ == "__main__":
    main()
