"""
用户行为分析系统

基于Pandas的电商用户行为数据分析，包含：
1. 用户转化漏斗分析
2. 高价值用户识别（RFM模型）- 5分位数分箱 + 8分类法
3. 异常行为检测 - 多维度 + 时间模式检测
4. 商品推荐系统（协同过滤）- TF-IDF加权 + 冷启动处理



数据集来源：
- 名称：User Behavior Data from Taobao for Recommendation
- 来源：阿里云天池
- 链接：https://tianchi.aliyun.com/dataset/649
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import gc
from collections import defaultdict

# ============================================================================
# 配置参数
# ============================================================================

CONFIG = {
    # 时间窗口配置
    'funnel_window_days': 30,      # 漏斗分析时间窗口
    'rfm_window_days': 90,         # RFM分析时间窗口
    'anomaly_window_days': 7,      # 异常检测时间窗口
    'cf_window_days': 180,         # 协同过滤时间窗口

    # 内存优化配置
    'chunk_size': 500000,          # 每块处理行数
    'n_sampling_passes': 5,        # 采样次数
    'sample_size_per_pass': 50000, # 每次采样大小
    'gc_interval': 10,             # GC间隔（每处理多少chunk）

    # 热门商品降权配置
    'idf_smoothing': 1,            # IDF平滑参数
    'popularity_alpha': 0.5,       # 热度惩罚系数

    # RFM配置
    'rfm_n_bins': 5,               # RFM分箱数量
    'rfm_binning_method': 'quantile',  # 分箱方法

    # 协同过滤配置
    'behavior_weights': {'buy': 5, 'cart': 3, 'fav': 2, 'pv': 1},
    'time_decay': 0.95,            # 时间衰减因子
    'min_interactions': 2,         # 最小交互次数
    'top_similar_items': 5,        # 每个商品的相似商品数
    'max_recommendations': 10,     # 最大推荐数

    # 动态阈值配置
    'threshold_config': {
        'percentile': 95,              # 统一使用95%分位数
        'use_dynamic_thresholds': True,  # 启用动态阈值
    },

    # 异常检测配置（静态后备值，当动态阈值禁用时使用）
    'frequency_thresholds': {
        'pv': 200,
        'fav': 50,
        'cart': 30,
        'buy': 20
    },
    'abnormal_hours': [1, 2, 3, 4, 5],  # 异常时间段（凌晨）
    'night_activity_threshold': 10,     # 凌晨活动阈值
    'z_score_threshold': 3,             # Z-Score阈值
    'buy_pv_ratio_threshold': 0.5,      # 购买/浏览比阈值（疑似刷单）
    'crawler_pv_threshold': 1000,       # 爬虫PV阈值

    # 数据路径
    'csv_path': r"C:\Users\liuch\Desktop\bUbUUdUd\UserBehavior.csv\UserBehavior.csv",
    'output_dir': "output",
}


# ============================================================================
# 时间窗口管理器
# ============================================================================

class TimeWindowManager:
    """时间窗口管理器，支持固定窗口和滑动窗口"""

    def __init__(self, reference_date=None):
        self.reference_date = reference_date
        self.data_min_date = None
        self.data_max_date = None

    def set_reference_date(self, date):
        """设置参考日期"""
        if isinstance(date, str):
            self.reference_date = pd.to_datetime(date)
        else:
            self.reference_date = date

    def set_data_range(self, min_date, max_date):
        """设置数据的实际日期范围"""
        if isinstance(min_date, str):
            self.data_min_date = pd.to_datetime(min_date)
        else:
            self.data_min_date = min_date

        if isinstance(max_date, str):
            self.data_max_date = pd.to_datetime(max_date)
        else:
            self.data_max_date = max_date

    def get_actual_window_days(self, requested_days):
        """获取实际可用的窗口天数（不超过数据范围）"""
        if self.data_min_date is None or self.data_max_date is None:
            return requested_days

        data_span = (self.data_max_date - self.data_min_date).days
        return min(requested_days, data_span)

    def filter_by_window(self, df, date_col, window_days, fallback_to_all=True):
        """根据时间窗口过滤数据

        Args:
            df: 数据框
            date_col: 日期列名
            window_days: 时间窗口天数
            fallback_to_all: 如果窗口内无数据，是否回退到使用全量数据

        Returns:
            过滤后的数据框
        """
        if self.reference_date is None:
            raise ValueError("Reference date not set")

        # 自动调整窗口大小，确保不超过数据范围
        actual_window_days = self.get_actual_window_days(window_days)
        start_date = self.reference_date - timedelta(days=actual_window_days)

        if df[date_col].dtype == 'object':
            df_dates = pd.to_datetime(df[date_col])
        else:
            df_dates = df[date_col]

        mask = (df_dates >= start_date) & (df_dates <= self.reference_date)
        filtered_df = df[mask].copy()

        # 如果过滤后无数据且允许回退，则使用全量数据
        if len(filtered_df) == 0 and fallback_to_all:
            print(f"  [警告] 时间窗口({window_days}天)内无数据，使用全量数据")
            return df.copy()

        return filtered_df

    def get_window_bounds(self, window_days):
        """获取时间窗口边界"""
        if self.reference_date is None:
            raise ValueError("Reference date not set")

        actual_window_days = self.get_actual_window_days(window_days)
        start_date = self.reference_date - timedelta(days=actual_window_days)
        return start_date, self.reference_date


# ============================================================================
# 流式数据处理器（内存优化）
# ============================================================================

class StreamingDataProcessor:
    """流式数据处理器，支持分块处理和增量聚合"""

    def __init__(self, csv_path, chunk_size=500000):
        self.csv_path = csv_path
        self.chunk_size = chunk_size
        self.dtypes = {
            'user_id': 'int32',
            'item_id': 'int32',
            'category_id': 'int32',
            'behavior_type': 'category',
            'timestamp': 'int32'
        }

    def load_data_streaming(self, preprocess_func=None):
        """流式加载数据，支持预处理函数"""
        chunks = []
        chunk_count = 0

        print("\n正在流式读取数据...")
        for chunk in pd.read_csv(
            self.csv_path,
            names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp'],
            dtype=self.dtypes,
            chunksize=self.chunk_size
        ):
            if preprocess_func:
                chunk = preprocess_func(chunk)
            chunks.append(chunk)
            chunk_count += 1
            print(f"已读取 {chunk_count * self.chunk_size:,} 行", end='\r')

            # 定期GC
            if chunk_count % CONFIG['gc_interval'] == 0:
                gc.collect()

        df = pd.concat(chunks, ignore_index=True)
        gc.collect()
        print(f"\n数据加载完成，共 {len(df):,} 行")
        return df

    def process_in_chunks(self, process_func, aggregator_func):
        """分块处理数据，支持增量聚合"""
        results = []
        chunk_count = 0

        for chunk in pd.read_csv(
            self.csv_path,
            names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp'],
            dtype=self.dtypes,
            chunksize=self.chunk_size
        ):
            chunk_result = process_func(chunk)
            results.append(chunk_result)
            chunk_count += 1

            if chunk_count % CONFIG['gc_interval'] == 0:
                gc.collect()

        return aggregator_func(results)

    def multi_pass_sampling(self, df, sample_func, n_passes=5, sample_size=50000):
        """多次少量采样，最后聚合结果"""
        all_results = []

        for i in range(n_passes):
            sample = df.sample(n=min(sample_size, len(df)), random_state=42 + i)
            result = sample_func(sample)
            all_results.append(result)
            gc.collect()

        return all_results


# ============================================================================
# RFM分析器（改进版）
# ============================================================================

class RFMAnalyzer:
    """RFM分析器，支持分位数分箱和8类用户分层"""

    def __init__(self, n_bins=5):
        self.n_bins = n_bins

    def calculate_rfm(self, buy_df, max_date):
        """计算RFM特征"""
        user_rfm = buy_df.groupby('user_id', observed=True).agg({
            'date': lambda x: (max_date - pd.to_datetime(x.max())).days,
            'item_id': 'count'
        }).reset_index()
        user_rfm.columns = ['user_id', 'recency', 'frequency']
        user_rfm['monetary'] = user_rfm['frequency']
        return user_rfm

    def _safe_qcut(self, series, n_bins, ascending=True):
        """安全的分位数分箱，自动处理数据不足的情况"""
        if len(series) == 0:
            return pd.Series(dtype='int64')

        # 计算实际可用的分箱数（不能超过唯一值数量）
        n_unique = series.nunique()
        actual_bins = min(n_bins, n_unique)

        if actual_bins < 2:
            # 数据太少，全部给中间分数
            return pd.Series([n_bins // 2 + 1] * len(series), index=series.index)

        # 生成标签
        if ascending:
            labels = list(range(1, actual_bins + 1))
        else:
            labels = list(range(actual_bins, 0, -1))

        try:
            result = pd.qcut(
                series.rank(method='first'),
                q=actual_bins,
                labels=labels,
                duplicates='drop'
            )
            return result.astype(int)
        except ValueError:
            # 如果还是失败，使用等宽分箱作为后备
            try:
                result = pd.cut(
                    series,
                    bins=actual_bins,
                    labels=labels,
                    duplicates='drop'
                )
                return result.astype(int)
            except ValueError:
                # 最后的后备方案
                return pd.Series([n_bins // 2 + 1] * len(series), index=series.index)

    def quantile_binning(self, user_rfm):
        """分位数分箱（替代等宽分箱）"""
        if len(user_rfm) == 0:
            user_rfm['r_score'] = pd.Series(dtype='int64')
            user_rfm['f_score'] = pd.Series(dtype='int64')
            user_rfm['m_score'] = pd.Series(dtype='int64')
            return user_rfm

        # R分数：recency越小越好，所以分数反转（ascending=False）
        user_rfm['r_score'] = self._safe_qcut(user_rfm['recency'], self.n_bins, ascending=False)

        # F分数：frequency越大越好
        user_rfm['f_score'] = self._safe_qcut(user_rfm['frequency'], self.n_bins, ascending=True)

        # M分数：monetary越大越好
        user_rfm['m_score'] = self._safe_qcut(user_rfm['monetary'], self.n_bins, ascending=True)

        return user_rfm

    def segment_users_8class(self, row):
        """8类用户分层（RFM经典分类法）"""
        r, f, m = row['r_score'], row['f_score'], row['m_score']

        # 阈值设为中间值（n_bins=5时，阈值为3）
        r_mid = self.n_bins // 2 + 1  # 3
        f_mid = self.n_bins // 2 + 1  # 3
        m_mid = self.n_bins // 2 + 1  # 3

        r_high = r >= r_mid
        f_high = f >= f_mid
        m_high = m >= m_mid

        if r_high and f_high and m_high:
            return '重要价值客户'
        elif r_high and not f_high and m_high:
            return '重要发展客户'
        elif not r_high and f_high and m_high:
            return '重要保持客户'
        elif not r_high and not f_high and m_high:
            return '重要挽留客户'
        elif r_high and f_high and not m_high:
            return '一般价值客户'
        elif r_high and not f_high and not m_high:
            return '一般发展客户'
        elif not r_high and f_high and not m_high:
            return '一般保持客户'
        else:
            return '一般挽留客户'

    def analyze(self, buy_df, max_date):
        """执行完整的RFM分析"""
        # 计算RFM
        user_rfm = self.calculate_rfm(buy_df, max_date)

        # 分位数分箱
        user_rfm = self.quantile_binning(user_rfm)

        # 计算总分
        user_rfm['rfm_score'] = user_rfm['r_score'] + user_rfm['f_score'] + user_rfm['m_score']

        # 8类用户分层
        user_rfm['user_segment'] = user_rfm.apply(self.segment_users_8class, axis=1)

        return user_rfm


# ============================================================================
# 异常检测器（增强版）
# ============================================================================

class AnomalyDetector:
    """异常检测器，支持多维度检测"""

    def __init__(self, config):
        self.frequency_thresholds = config['frequency_thresholds']
        self.abnormal_hours = config['abnormal_hours']
        self.night_activity_threshold = config['night_activity_threshold']
        self.z_score_threshold = config['z_score_threshold']
        self.buy_pv_ratio_threshold = config['buy_pv_ratio_threshold']
        self.crawler_pv_threshold = config['crawler_pv_threshold']

    def detect_high_frequency(self, df):
        """检测分行为类型的高频异常"""
        daily_behavior = df.groupby(
            ['user_id', 'date', 'behavior_type'],
            observed=True
        ).size().reset_index(name='count')

        anomalies = []
        for behavior, threshold in self.frequency_thresholds.items():
            behavior_data = daily_behavior[daily_behavior['behavior_type'] == behavior]
            anomaly = behavior_data[behavior_data['count'] > threshold].copy()
            if len(anomaly) > 0:
                anomaly['anomaly_type'] = f'高频{behavior}'
                anomaly['threshold'] = threshold
                anomalies.append(anomaly)

        if anomalies:
            return pd.concat(anomalies, ignore_index=True)
        return pd.DataFrame()

    def detect_time_pattern_anomaly(self, df):
        """检测时间模式异常（凌晨活动）"""
        if 'hour' not in df.columns:
            df = df.copy()
            df['hour'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour

        night_activity = df[df['hour'].isin(self.abnormal_hours)].groupby(
            ['user_id', 'date'],
            observed=True
        ).size().reset_index(name='night_count')

        anomaly = night_activity[night_activity['night_count'] > self.night_activity_threshold].copy()
        if len(anomaly) > 0:
            anomaly['anomaly_type'] = '凌晨异常活动'
            anomaly['threshold'] = self.night_activity_threshold

        return anomaly

    def detect_behavior_sequence_anomaly(self, df):
        """检测行为序列异常"""
        anomalies = []

        # 计算用户行为统计
        user_behavior = df.groupby(['user_id', 'behavior_type'], observed=True).size().unstack(fill_value=0)

        # 确保所有行为类型列存在
        for col in ['pv', 'buy', 'fav', 'cart']:
            if col not in user_behavior.columns:
                user_behavior[col] = 0

        # 疑似刷单：购买/浏览比过高
        user_behavior['buy_pv_ratio'] = user_behavior['buy'] / (user_behavior['pv'] + 1)
        suspicious_orders = user_behavior[
            (user_behavior['buy_pv_ratio'] > self.buy_pv_ratio_threshold) &
            (user_behavior['buy'] > 5)
        ].reset_index()

        if len(suspicious_orders) > 0:
            suspicious_orders['anomaly_type'] = '疑似刷单'
            suspicious_orders['detail'] = suspicious_orders.apply(
                lambda x: f"购买{x['buy']}次,浏览{x['pv']}次,比例{x['buy_pv_ratio']:.2f}",
                axis=1
            )
            anomalies.append(suspicious_orders[['user_id', 'anomaly_type', 'detail']])

        # 疑似爬虫：只有大量PV，无其他行为
        crawlers = user_behavior[
            (user_behavior['pv'] > self.crawler_pv_threshold) &
            (user_behavior['buy'] == 0) &
            (user_behavior['cart'] == 0) &
            (user_behavior['fav'] == 0)
        ].reset_index()

        if len(crawlers) > 0:
            crawlers['anomaly_type'] = '疑似爬虫'
            crawlers['detail'] = crawlers.apply(
                lambda x: f"浏览{x['pv']}次,无其他行为",
                axis=1
            )
            anomalies.append(crawlers[['user_id', 'anomaly_type', 'detail']])

        if anomalies:
            return pd.concat(anomalies, ignore_index=True)
        return pd.DataFrame()

    def detect_statistical_anomaly(self, df):
        """基于Z-Score的统计异常检测"""
        user_activity = df.groupby('user_id', observed=True).size().reset_index(name='total_actions')

        mean_actions = user_activity['total_actions'].mean()
        std_actions = user_activity['total_actions'].std()

        if std_actions > 0:
            user_activity['z_score'] = (user_activity['total_actions'] - mean_actions) / std_actions
            anomaly = user_activity[user_activity['z_score'] > self.z_score_threshold].copy()

            if len(anomaly) > 0:
                anomaly['anomaly_type'] = 'Z-Score异常'
                anomaly['detail'] = anomaly.apply(
                    lambda x: f"总行为{x['total_actions']},Z-Score={x['z_score']:.2f}",
                    axis=1
                )
                return anomaly[['user_id', 'anomaly_type', 'detail', 'z_score']]

        return pd.DataFrame()

    def detect_all(self, df):
        """执行所有异常检测"""
        print("  - 检测高频行为异常...")
        high_freq = self.detect_high_frequency(df)

        print("  - 检测时间模式异常...")
        time_pattern = self.detect_time_pattern_anomaly(df)

        print("  - 检测行为序列异常...")
        behavior_seq = self.detect_behavior_sequence_anomaly(df)

        print("  - 检测统计异常...")
        statistical = self.detect_statistical_anomaly(df)

        results = []
        for result in [high_freq, time_pattern, behavior_seq, statistical]:
            if len(result) > 0:
                results.append(result)

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()


# ============================================================================
# 协同过滤推荐器（增强版）
# ============================================================================

class CollaborativeFilteringRecommender:
    """协同过滤推荐器，支持IDF加权、冷启动处理、时间衰减"""

    def __init__(self, config):
        self.behavior_weights = config['behavior_weights']
        self.time_decay = config['time_decay']
        self.idf_smoothing = config['idf_smoothing']
        self.popularity_alpha = config['popularity_alpha']
        self.top_similar_items = config['top_similar_items']
        self.max_recommendations = config['max_recommendations']

        self.item_idf = {}
        self.item_popularity_penalty = {}
        self.global_popular = []
        self.category_popular = {}
        self.item_similarity = None

    def calculate_item_idf(self, df):
        """计算商品IDF权重：热门商品权重低，冷门商品权重高"""
        total_users = df['user_id'].nunique()
        item_user_count = df.groupby('item_id', observed=True)['user_id'].nunique()

        # IDF公式：log(总用户数 / (商品用户数 + 平滑因子)) + 1
        item_idf = np.log(total_users / (item_user_count + self.idf_smoothing)) + 1
        self.item_idf = item_idf.to_dict()

        return self.item_idf

    def calculate_popularity_penalty(self, df):
        """计算热度惩罚系数"""
        item_counts = df.groupby('item_id', observed=True).size()

        # 惩罚公式：1 / (1 + alpha * log(1 + count))
        penalty = 1 / (1 + self.popularity_alpha * np.log1p(item_counts))
        self.item_popularity_penalty = penalty.to_dict()

        return self.item_popularity_penalty

    def build_global_popular(self, df, top_n=100):
        """构建全局热门商品列表"""
        # 综合考虑购买行为
        buy_counts = df[df['behavior_type'] == 'buy'].groupby('item_id', observed=True).size()
        self.global_popular = buy_counts.nlargest(top_n).index.tolist()
        return self.global_popular

    def build_category_popular(self, df, top_n=20):
        """构建分类目热门商品列表"""
        buy_df = df[df['behavior_type'] == 'buy']

        for category_id in buy_df['category_id'].unique():
            category_buys = buy_df[buy_df['category_id'] == category_id]
            item_counts = category_buys.groupby('item_id', observed=True).size()
            self.category_popular[category_id] = item_counts.nlargest(top_n).index.tolist()

        return self.category_popular

    def calculate_time_decay_score(self, days_ago, base_score):
        """计算时间衰减分数"""
        return base_score * (self.time_decay ** days_ago)

    def build_user_item_matrix(self, df, max_date):
        """构建加权的用户-商品交互矩阵"""
        # 计算每条记录的时间衰减
        df = df.copy()
        df['days_ago'] = (max_date - pd.to_datetime(df['date'])).dt.days

        # 应用行为权重和时间衰减
        df['score'] = df.apply(
            lambda x: self.behavior_weights.get(x['behavior_type'], 1) *
                      (self.time_decay ** x['days_ago']),
            axis=1
        )

        # 应用IDF权重
        df['idf_weight'] = df['item_id'].map(self.item_idf).fillna(1)
        df['weighted_score'] = df['score'] * df['idf_weight']

        # 聚合用户-商品评分
        user_item = df.groupby(['user_id', 'item_id'], observed=True).agg({
            'weighted_score': 'sum',
            'score': 'sum'
        }).reset_index()

        return user_item

    def build_item_similarity_matrix(self, user_item_sample):
        """构建商品相似度矩阵（应用IDF加权）"""
        if len(user_item_sample) == 0:
            self.item_similarity = pd.DataFrame(columns=[
                'item_id_1', 'item_id_2', 'weighted_similarity', 'co_occurrence',
                'penalty_1', 'penalty_2', 'final_similarity'
            ])
            return self.item_similarity

        # 基于用户的协同过滤：找出同一用户交互的商品对
        item_pairs = user_item_sample.merge(
            user_item_sample,
            on='user_id',
            suffixes=('_1', '_2')
        )
        item_pairs = item_pairs[item_pairs['item_id_1'] < item_pairs['item_id_2']]

        if len(item_pairs) == 0:
            self.item_similarity = pd.DataFrame(columns=[
                'item_id_1', 'item_id_2', 'weighted_similarity', 'co_occurrence',
                'penalty_1', 'penalty_2', 'final_similarity'
            ])
            return self.item_similarity

        # 计算加权共现
        item_pairs['weighted_co_occurrence'] = (
            item_pairs['weighted_score_1'] * item_pairs['weighted_score_2']
        )

        # 聚合共现次数
        item_similarity = item_pairs.groupby(
            ['item_id_1', 'item_id_2'],
            observed=True
        ).agg({
            'weighted_co_occurrence': 'sum',
            'user_id': 'count'  # 原始共现次数
        }).reset_index()

        item_similarity.columns = ['item_id_1', 'item_id_2', 'weighted_similarity', 'co_occurrence']

        # 应用热度惩罚
        item_similarity['penalty_1'] = item_similarity['item_id_1'].map(
            self.item_popularity_penalty
        ).fillna(1)
        item_similarity['penalty_2'] = item_similarity['item_id_2'].map(
            self.item_popularity_penalty
        ).fillna(1)
        item_similarity['final_similarity'] = (
            item_similarity['weighted_similarity'] *
            item_similarity['penalty_1'] *
            item_similarity['penalty_2']
        )

        item_similarity = item_similarity.sort_values('final_similarity', ascending=False)
        self.item_similarity = item_similarity

        return item_similarity

    def get_top_similar_items(self):
        """获取每个商品的TOP相似商品"""
        if self.item_similarity is None:
            return pd.DataFrame()

        top_similar = self.item_similarity.groupby(
            'item_id_1',
            observed=True
        ).head(self.top_similar_items).reset_index(drop=True)

        top_similar['rank'] = top_similar.groupby('item_id_1', observed=True).cumcount() + 1

        return top_similar

    def recommend_for_new_user(self, user_context=None, n=10):
        """新用户冷启动推荐"""
        if user_context and 'category_id' in user_context:
            category_id = user_context['category_id']
            if category_id in self.category_popular:
                return {
                    'recommendations': self.category_popular[category_id][:n],
                    'strategy': 'category_popular'
                }

        return {
            'recommendations': self.global_popular[:n],
            'strategy': 'global_popular'
        }

    def get_recommendation(self, user_id, user_history, top_similar, n=10):
        """
        获取用户推荐，支持冷启动降级

        降级策略：
        1. 有足够CF推荐 -> 纯CF
        2. CF不足 -> CF + 热门补充
        3. 无历史 -> 冷启动推荐
        """
        if not user_history or len(user_history) == 0:
            return self.recommend_for_new_user(n=n)

        purchased = set(user_history)

        # 获取CF推荐（去重）
        cf_recs = top_similar[top_similar['item_id_1'].isin(purchased)]
        cf_recs = cf_recs[~cf_recs['item_id_2'].isin(purchased)]
        cf_recs = cf_recs.drop_duplicates(subset='item_id_2').nlargest(n, 'final_similarity')
        cf_items = cf_recs['item_id_2'].tolist()

        if len(cf_items) >= n:
            return {
                'recommendations': cf_items[:n],
                'strategy': 'collaborative_filtering',
                'cf_count': len(cf_items)
            }

        # 混合策略：CF + 热门补充
        popular_supplement = [i for i in self.global_popular if i not in purchased and i not in cf_items]
        combined = cf_items + popular_supplement[:n - len(cf_items)]

        return {
            'recommendations': combined[:n],
            'strategy': 'hybrid',
            'cf_count': len(cf_items),
            'popular_count': len(combined) - len(cf_items)
        }

    def stratified_sampling(self, df, sample_size):
        """分层采样：按用户活跃度分层，避免只采样活跃用户"""
        if len(df) == 0:
            return df

        user_activity = df.groupby('user_id', observed=True).size().reset_index(name='activity')

        # 如果用户数太少，直接返回原数据
        if len(user_activity) < 3:
            return df

        # 按活跃度分为3层
        try:
            user_activity['activity_tier'] = pd.qcut(
                user_activity['activity'].rank(method='first'),
                q=3,
                labels=['low', 'medium', 'high'],
                duplicates='drop'
            )
        except ValueError:
            # 如果分层失败，返回原数据
            return df

        # 每层等比例采样
        sampled_users = []
        tier_sample_size = sample_size // 3

        for tier in ['low', 'medium', 'high']:
            tier_users = user_activity[user_activity['activity_tier'] == tier]['user_id']
            n_sample = min(tier_sample_size, len(tier_users))
            if n_sample > 0:
                sampled_users.extend(tier_users.sample(n=n_sample, random_state=42).tolist())

        if len(sampled_users) == 0:
            return df

        return df[df['user_id'].isin(sampled_users)]


# ============================================================================
# 主程序
# ============================================================================

def preprocess_chunk(chunk):
    """预处理数据块"""
    chunk = chunk.drop_duplicates()
    chunk = chunk[chunk['behavior_type'].isin(['pv', 'fav', 'cart', 'buy'])]

    # 过滤异常时间戳（只保留2017年的数据，这是淘宝数据集的实际时间范围）
    # 时间戳范围：2017-11-25 到 2017-12-03
    min_ts = 1511539200  # 2017-11-25 00:00:00
    max_ts = 1512345600  # 2017-12-04 00:00:00
    chunk = chunk[(chunk['timestamp'] >= min_ts) & (chunk['timestamp'] <= max_ts)]

    return chunk


def main():
    print("=" * 80)
    print("用户行为分析系统 (增强版)")
    print("=" * 80)

    # 创建输出目录
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # ========================================================================
    # Step 1: 数据加载（流式处理）
    # ========================================================================
    print("\n" + "-" * 40)
    print("Step 1: 流式加载数据")
    print("-" * 40)

    processor = StreamingDataProcessor(
        CONFIG['csv_path'],
        chunk_size=CONFIG['chunk_size']
    )

    df = processor.load_data_streaming(preprocess_func=preprocess_chunk)

    # 添加时间特征
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
    df['hour'] = df['datetime'].dt.hour

    print(f"预处理后: {len(df):,} 条记录")
    print(f"用户数: {df['user_id'].nunique():,}")
    print(f"商品数: {df['item_id'].nunique():,}")

    # 设置时间窗口管理器
    time_manager = TimeWindowManager()
    min_date = pd.to_datetime(df['date'].min())
    max_date = pd.to_datetime(df['date'].max())
    time_manager.set_reference_date(max_date)
    time_manager.set_data_range(min_date, max_date)

    data_span_days = (max_date - min_date).days
    print(f"数据日期范围: {min_date.date()} 至 {max_date.date()} (共{data_span_days}天)")

    gc.collect()

    # ========================================================================
    # Step 2: 用户转化漏斗分析
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. 用户转化漏斗分析")
    print("=" * 80)

    # 应用时间窗口
    actual_funnel_days = time_manager.get_actual_window_days(CONFIG['funnel_window_days'])
    funnel_df = time_manager.filter_by_window(df, 'date', CONFIG['funnel_window_days'])
    print(f"时间窗口: 配置{CONFIG['funnel_window_days']}天, 实际{actual_funnel_days}天, 数据量: {len(funnel_df):,}")

    # 计算各行为类型的用户数
    funnel = funnel_df.groupby('behavior_type', observed=True)['user_id'].nunique().reset_index()
    funnel.columns = ['behavior_type', 'user_count']

    # 计算转化率
    pv_users = funnel[funnel['behavior_type'] == 'pv']['user_count'].values[0] if 'pv' in funnel['behavior_type'].values else 1
    fav_users = funnel[funnel['behavior_type'] == 'fav']['user_count'].values[0] if 'fav' in funnel['behavior_type'].values else 0
    cart_users = funnel[funnel['behavior_type'] == 'cart']['user_count'].values[0] if 'cart' in funnel['behavior_type'].values else 0
    buy_users = funnel[funnel['behavior_type'] == 'buy']['user_count'].values[0] if 'buy' in funnel['behavior_type'].values else 0

    print(f"\n浏览(pv)用户数: {pv_users:,}")
    print(f"收藏(fav)用户数: {fav_users:,} - 转化率: {fav_users/pv_users*100:.2f}%")
    print(f"加购(cart)用户数: {cart_users:,} - 转化率: {cart_users/pv_users*100:.2f}%")
    print(f"购买(buy)用户数: {buy_users:,} - 转化率: {buy_users/pv_users*100:.2f}%")

    funnel.to_csv(f"{CONFIG['output_dir']}/conversion_funnel.csv", index=False, encoding='utf-8-sig')
    print("转化漏斗分析完成")

    del funnel_df
    gc.collect()

    # ========================================================================
    # Step 3: RFM分析（改进版）
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. 高价值用户识别 (RFM模型 - 5分位数分箱 + 8分类法)")
    print("=" * 80)

    # 应用时间窗口
    actual_rfm_days = time_manager.get_actual_window_days(CONFIG['rfm_window_days'])
    rfm_df = time_manager.filter_by_window(df, 'date', CONFIG['rfm_window_days'])
    buy_df = rfm_df[rfm_df['behavior_type'] == 'buy'].copy()
    print(f"时间窗口: 配置{CONFIG['rfm_window_days']}天, 实际{actual_rfm_days}天, 购买记录: {len(buy_df):,}")

    if len(buy_df) > 0:
        # RFM分析
        rfm_analyzer = RFMAnalyzer(n_bins=CONFIG['rfm_n_bins'])
        user_rfm = rfm_analyzer.analyze(buy_df, max_date)

        print(f"\n分析基准日期: {max_date.date()}")
        print(f"RFM分箱方法: 分位数分箱 ({CONFIG['rfm_n_bins']}分箱)")

        print("\n用户8类分层统计:")
        segment_counts = user_rfm['user_segment'].value_counts()
        for segment, count in segment_counts.items():
            print(f"  {segment}: {count:,} ({count/len(user_rfm)*100:.1f}%)")

        print("\nTOP 10 高价值用户:")
        print(user_rfm.nlargest(10, 'rfm_score')[['user_id', 'recency', 'frequency', 'r_score', 'f_score', 'm_score', 'rfm_score', 'user_segment']])
    else:
        print("\n[警告] 无购买数据，跳过RFM分析")
        user_rfm = pd.DataFrame(columns=['user_id', 'recency', 'frequency', 'monetary', 'r_score', 'f_score', 'm_score', 'rfm_score', 'user_segment'])

    user_rfm.to_csv(f"{CONFIG['output_dir']}/user_rfm.csv", index=False, encoding='utf-8-sig')
    print("高价值用户识别完成")

    del rfm_df
    gc.collect()

    # ========================================================================
    # Step 4: 异常检测（增强版）
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. 异常行为检测 (多维度检测)")
    print("=" * 80)

    # 应用时间窗口
    actual_anomaly_days = time_manager.get_actual_window_days(CONFIG['anomaly_window_days'])
    anomaly_df = time_manager.filter_by_window(df, 'date', CONFIG['anomaly_window_days'])
    print(f"时间窗口: 配置{CONFIG['anomaly_window_days']}天, 实际{actual_anomaly_days}天, 数据量: {len(anomaly_df):,}")

    if len(anomaly_df) > 0:
        # 异常检测
        detector = AnomalyDetector(CONFIG)
        anomaly_results = detector.detect_all(anomaly_df)

        if len(anomaly_results) > 0:
            print(f"\n检测到异常记录数: {len(anomaly_results)}")
            print(f"涉及异常用户数: {anomaly_results['user_id'].nunique()}")

            print("\n异常类型分布:")
            if 'anomaly_type' in anomaly_results.columns:
                for anomaly_type, count in anomaly_results['anomaly_type'].value_counts().items():
                    print(f"  {anomaly_type}: {count}")

            print("\n异常行为示例:")
            print(anomaly_results.head(15))
        else:
            print("\n未检测到异常行为")
    else:
        print("\n[警告] 时间窗口内无数据，跳过异常检测")
        anomaly_results = pd.DataFrame(columns=['user_id', 'anomaly_type', 'detail'])

    anomaly_results.to_csv(f"{CONFIG['output_dir']}/anomaly_users.csv", index=False, encoding='utf-8-sig')
    print("异常行为检测完成")

    del anomaly_df
    gc.collect()

    # ========================================================================
    # Step 5: 协同过滤推荐（增强版）
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. 商品推荐系统 (协同过滤 - TF-IDF加权 + 冷启动处理)")
    print("=" * 80)

    # 应用时间窗口
    actual_cf_days = time_manager.get_actual_window_days(CONFIG['cf_window_days'])
    cf_df = time_manager.filter_by_window(df, 'date', CONFIG['cf_window_days'])
    print(f"时间窗口: 配置{CONFIG['cf_window_days']}天, 实际{actual_cf_days}天, 数据量: {len(cf_df):,}")

    # 初始化推荐器
    recommender = CollaborativeFilteringRecommender(CONFIG)
    top_similar = pd.DataFrame()  # 默认空DataFrame

    if len(cf_df) > 0:
        # 计算IDF权重
        print("\n计算IDF权重...")
        recommender.calculate_item_idf(cf_df)

        # 计算热度惩罚
        print("计算热度惩罚...")
        recommender.calculate_popularity_penalty(cf_df)

        # 构建热门商品列表
        print("构建热门商品列表...")
        recommender.build_global_popular(cf_df)
        recommender.build_category_popular(cf_df)

        # 构建用户-商品交互矩阵
        print("构建用户-商品交互矩阵...")
        interaction_df = cf_df[cf_df['behavior_type'].isin(['buy', 'cart', 'fav'])].copy()

        if len(interaction_df) > 0:
            user_item = recommender.build_user_item_matrix(interaction_df, max_date)
            print(f"用户-商品交互记录数: {len(user_item):,}")

            if len(user_item) > 0:
                # 分层采样
                print("执行分层采样...")
                sample_size = min(CONFIG['sample_size_per_pass'], len(user_item))
                user_item_sample = recommender.stratified_sampling(user_item, sample_size)
                print(f"采样后记录数: {len(user_item_sample):,}")

                if len(user_item_sample) > 0:
                    # 构建商品相似度矩阵
                    print("构建商品相似度矩阵（应用IDF加权）...")
                    item_similarity = recommender.build_item_similarity_matrix(user_item_sample)

                    # 获取TOP相似商品
                    top_similar = recommender.get_top_similar_items()

                    if len(top_similar) > 0:
                        print("\n商品推荐示例（含IDF加权）:")
                        display_cols = ['item_id_1', 'item_id_2', 'co_occurrence', 'weighted_similarity', 'final_similarity', 'rank']
                        available_cols = [c for c in display_cols if c in top_similar.columns]
                        print(top_similar[available_cols].head(20))
                    else:
                        print("\n[警告] 无法生成商品相似度")
        else:
            print("\n[警告] 无交互数据（购买/加购/收藏）")
    else:
        print("\n[警告] 时间窗口内无数据，跳过协同过滤")

    # 保存商品相似度
    if len(top_similar) == 0:
        top_similar = pd.DataFrame(columns=['item_id_1', 'item_id_2', 'co_occurrence', 'weighted_similarity', 'final_similarity', 'rank'])
    top_similar.to_csv(f"{CONFIG['output_dir']}/item_recommendations.csv", index=False, encoding='utf-8-sig')
    print("商品推荐完成")

    # ========================================================================
    # Step 6: 用户个性化推荐（含冷启动处理）
    # ========================================================================
    print("\n" + "-" * 40)
    print("生成用户个性化推荐（含冷启动处理）")
    print("-" * 40)

    user_rec_df = pd.DataFrame()

    if len(buy_df) > 0:
        # 获取用户购买历史
        user_purchased = buy_df.groupby('user_id', observed=True)['item_id'].apply(list).reset_index()
        user_purchased.columns = ['user_id', 'purchased_items']

        # 为用户生成推荐
        user_recommendations = []
        strategy_counts = defaultdict(int)

        # 处理前1000个用户
        for idx, row in user_purchased.head(1000).iterrows():
            user_id = row['user_id']
            purchased = row['purchased_items']

            result = recommender.get_recommendation(
                user_id,
                purchased,
                top_similar,
                n=CONFIG['max_recommendations']
            )

            strategy_counts[result['strategy']] += 1

            for rank, item_id in enumerate(result['recommendations'], 1):
                user_recommendations.append({
                    'user_id': user_id,
                    'recommended_item': item_id,
                    'rank': rank,
                    'strategy': result['strategy']
                })

        user_rec_df = pd.DataFrame(user_recommendations)

        if len(user_rec_df) > 0:
            print(f"\n生成用户推荐数: {len(user_rec_df):,}")
            print("\n推荐策略统计:")
            for strategy, count in strategy_counts.items():
                print(f"  {strategy}: {count} 用户")

            print("\n用户推荐示例:")
            print(user_rec_df.head(20))
        else:
            print("\n未生成用户推荐")
    else:
        print("\n[警告] 无购买数据，跳过用户推荐生成")

    # 保存结果
    if len(user_rec_df) == 0:
        user_rec_df = pd.DataFrame(columns=['user_id', 'recommended_item', 'rank', 'strategy'])
    user_rec_df.to_csv(f"{CONFIG['output_dir']}/user_recommendations.csv", index=False, encoding='utf-8-sig')
    print("用户推荐完成")



if __name__ == "__main__":
    main()
