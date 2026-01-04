"""
用户行为分析系统

基于Pandas的电商用户行为数据分析，包含：
1. 用户转化漏斗分析
2. 高价值用户识别（RFM模型）
3. 异常行为检测
4. 商品推荐系统（协同过滤）

数据集来源：
- 名称：User Behavior Data from Taobao for Recommendation
- 来源：阿里云天池
- 链接：https://tianchi.aliyun.com/dataset/649?spm=a2ty_o01.29997173.0.0.744f5171A9wDHy
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

print("=" * 80)
print("开始加载数据...")
print("=" * 80)

# 优化数据类型，减少内存占用
dtypes = {
    'user_id': 'int32',
    'item_id': 'int32',
    'category_id': 'int32',
    'behavior_type': 'category',
    'timestamp': 'int32'
}

# 分块读取大文件
chunk_size = 1000000
chunks = []

csv_path = r"C:\Users\liuch\Desktop\bUbUUdUd\UserBehavior.csv\UserBehavior.csv"

print("\n正在读取数据...")
for chunk in pd.read_csv(csv_path,
                         names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp'],
                         dtype=dtypes,
                         chunksize=chunk_size):
    chunks.append(chunk)
    print(f"已读取 {len(chunks) * chunk_size:,} 行", end='\r')

df = pd.concat(chunks, ignore_index=True)
print(f"\n数据加载完成，共 {len(df):,} 行")

# 数据预处理
df = df.drop_duplicates()
df = df[df['behavior_type'].isin(['pv', 'fav', 'cart', 'buy'])]
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')  # 转为字符串，避免内存问题

print(f"预处理后: {len(df):,} 条记录")
print(f"用户数: {df['user_id'].nunique():,}")
print(f"商品数: {df['item_id'].nunique():,}")

# 创建输出目录
os.makedirs("output", exist_ok=True)

print("\n" + "=" * 80)
print("1. 用户转化漏斗分析")
print("=" * 80)

# 计算各行为类型的用户数
funnel = df.groupby('behavior_type', observed=True)['user_id'].nunique().reset_index()
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

funnel.to_csv("output/conversion_funnel.csv", index=False)
print("转化漏斗分析完成")

print("\n" + "=" * 80)
print("2. 高价值用户识别 (RFM模型)")
print("=" * 80)

# 获取购买数据
buy_df = df[df['behavior_type'] == 'buy'].copy()
max_date = pd.to_datetime(df['date'].max())

print(f"分析基准日期: {max_date.date()}")

# RFM特征计算
user_rfm = buy_df.groupby('user_id', observed=True).agg({
    'date': lambda x: (max_date - pd.to_datetime(x.max())).days,  # Recency
    'item_id': 'count'  # Frequency & Monetary
}).reset_index()
user_rfm.columns = ['user_id', 'recency', 'frequency']
user_rfm['monetary'] = user_rfm['frequency']

# RFM评分
user_rfm['r_score'] = pd.cut(user_rfm['recency'], bins=3, labels=[3, 2, 1])
user_rfm['f_score'] = pd.cut(user_rfm['frequency'], bins=3, labels=[1, 2, 3])
user_rfm['m_score'] = pd.cut(user_rfm['monetary'], bins=3, labels=[1, 2, 3])

user_rfm['r_score'] = user_rfm['r_score'].astype(int)
user_rfm['f_score'] = user_rfm['f_score'].astype(int)
user_rfm['m_score'] = user_rfm['m_score'].astype(int)

# 计算总分
user_rfm['rfm_score'] = user_rfm['r_score'] + user_rfm['f_score'] + user_rfm['m_score']

# 用户分层
user_rfm['user_segment'] = pd.cut(user_rfm['rfm_score'],
                                   bins=[0, 5, 7, 9],
                                   labels=['低价值用户', '中价值用户', '高价值用户'])

print("\n用户分层统计:")
print(user_rfm['user_segment'].value_counts())

print("\nTOP 10 高价值用户:")
print(user_rfm.nlargest(10, 'rfm_score')[['user_id', 'recency', 'frequency', 'rfm_score', 'user_segment']])

user_rfm.to_csv("output/user_rfm.csv", index=False)
print("高价值用户识别完成")

print("\n" + "=" * 80)
print("3. 异常行为检测")
print("=" * 80)

# 检测单日异常高频操作（优化内存使用）
daily_behavior = df.groupby(['user_id', 'date', 'behavior_type'], observed=True).size().reset_index(name='count')
anomaly = daily_behavior[daily_behavior['count'] > 50]

print(f"\n检测到异常高频用户数: {anomaly['user_id'].nunique()}")
if len(anomaly) > 0:
    print("\n异常行为示例:")
    print(anomaly.nlargest(10, 'count'))
else:
    print("\n未检测到异常行为")

anomaly.to_csv("output/anomaly_users.csv", index=False)
print("异常行为检测完成")

print("\n" + "=" * 80)
print("4. 商品推荐系统 (基于协同过滤)")
print("=" * 80)

# 构建用户-商品交互矩阵（只用购买和加购）
interaction = df[df['behavior_type'].isin(['buy', 'cart'])].copy()
interaction['score'] = interaction['behavior_type'].map({'buy': 3, 'cart': 2})

# 聚合用户-商品评分
user_item = interaction.groupby(['user_id', 'item_id'], observed=True)['score'].sum().reset_index()

print(f"\n用户-商品交互记录数: {len(user_item):,}")

# 计算商品共现矩阵（简化版本，避免内存爆炸）
# 随机采样部分数据进行推荐计算
sample_size = min(100000, len(user_item))
user_item_sample = user_item.sample(n=sample_size, random_state=42)

# 基于用户的协同过滤：找出同一用户购买的商品对
item_pairs = user_item_sample.merge(user_item_sample, on='user_id', suffixes=('_1', '_2'))
item_pairs = item_pairs[item_pairs['item_id_1'] < item_pairs['item_id_2']]

# 计算商品共现次数
item_similarity = item_pairs.groupby(['item_id_1', 'item_id_2'], observed=True).size().reset_index(name='co_occurrence')
item_similarity = item_similarity.sort_values('co_occurrence', ascending=False)

# 为每个商品找出TOP5相似商品
top_similar = item_similarity.groupby('item_id_1', observed=True).head(5).reset_index(drop=True)
top_similar['rank'] = top_similar.groupby('item_id_1', observed=True).cumcount() + 1

print("\n商品推荐示例:")
print(top_similar.head(20))

top_similar.to_csv("output/item_recommendations.csv", index=False)
print("商品推荐完成")

# 生成用户个性化推荐
user_purchased = buy_df.groupby('user_id', observed=True)['item_id'].apply(list).reset_index()
user_purchased.columns = ['user_id', 'purchased_items']

# 为每个用户推荐商品（基于其购买过的商品）
user_recommendations = []

for idx, row in user_purchased.head(1000).iterrows():  # 只为前1000个用户生成推荐
    user_id = row['user_id']
    purchased = set(row['purchased_items'])

    # 找出用户购买商品的相似商品
    recs = top_similar[top_similar['item_id_1'].isin(purchased)]

    if len(recs) > 0:
        # 排除已购买商品
        recs = recs[~recs['item_id_2'].isin(purchased)]
        recs = recs.nlargest(5, 'co_occurrence')

        for _, rec in recs.iterrows():
            user_recommendations.append({
                'user_id': user_id,
                'recommended_item': rec['item_id_2'],
                'score': rec['co_occurrence'],
                'rank': rec['rank']
            })

user_rec_df = pd.DataFrame(user_recommendations)

if len(user_rec_df) > 0:
    print(f"\n生成用户推荐数: {len(user_rec_df):,}")
    print("\n用户推荐示例:")
    print(user_rec_df.head(20))
    user_rec_df.to_csv("output/user_recommendations.csv", index=False)
else:
    print("\n未生成用户推荐")
    pd.DataFrame(columns=['user_id', 'recommended_item', 'score', 'rank']).to_csv("output/user_recommendations.csv", index=False)

print("用户推荐完成")

print("\n" + "=" * 80)
print("分析完成！结果已保存到 output/ 目录")
print("=" * 80)
