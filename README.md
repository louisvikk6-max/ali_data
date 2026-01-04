# 用户行为分析系统 / User Behavior Analysis System

[中文](#中文) | [English](#english)

---

## 中文

基于Pandas的电商用户行为数据分析系统，提供用户转化分析、高价值用户识别、异常行为检测和智能推荐功能。

### 项目简介

本项目针对电商平台用户行为数据进行深度分析，帮助运营人员：
- 了解用户转化路径，优化运营策略
- 识别高价值用户，实现精准营销
- 发现异常行为，防范刷单风险
- 生成个性化推荐，提升用户体验

**技术栈：** Python + Pandas + NumPy

**适用场景：** 电商平台、在线零售、内容推荐等

### 核心功能

#### 1. 用户转化漏斗分析
分析用户从浏览到购买的完整路径，计算各环节转化率：
- **浏览(pv)** → **收藏(fav)** → **加购(cart)** → **购买(buy)**
- 输出每个环节的用户数量和转化率
- 帮助识别转化瓶颈

#### 2. 高价值用户识别（RFM模型）
基于经典的RFM模型进行用户价值评估：
- **R (Recency)**: 最近一次购买距今天数
- **F (Frequency)**: 购买频次
- **M (Monetary)**: 购买金额（用购买次数代替）

自动将用户分为三个层级：
- 高价值用户（RFM总分≥8）
- 中价值用户（6≤RFM总分<8）
- 低价值用户（RFM总分<6）

#### 3. 异常行为检测
识别潜在的刷单、作弊等异常行为：
- 检测单日异常高频操作（阈值：50次/天）
- 输出异常用户列表及其行为详情
- 支持自定义检测阈值

#### 4. 商品推荐系统
基于协同过滤算法的智能推荐：
- **商品相似度推荐**：找出与每个商品最相似的TOP5商品
- **用户个性化推荐**：基于用户购买历史生成个性化推荐列表
- 自动过滤用户已购买商品

### 数据集说明

#### 数据来源
本项目使用阿里巴巴公开的淘宝用户行为数据集：
- **数据集名称**：User Behavior Data from Taobao for Recommendation
- **数据来源**：阿里云天池
- **下载地址**：https://tianchi.aliyun.com/dataset/649

#### 输入数据格式
数据集（UserBehavior.csv）每行包含5个字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| user_id | int | 用户ID |
| item_id | int | 商品ID |
| category_id | int | 商品类目ID |
| behavior_type | string | 行为类型（pv/fav/cart/buy）|
| timestamp | int | Unix时间戳 |

#### 行为类型说明
- `pv`: 浏览（Page View）
- `fav`: 收藏（Favorite）
- `cart`: 加入购物车（Add to Cart）
- `buy`: 购买（Buy）

### 快速开始

#### 环境要求
- Python 3.7 或更高版本
- 16GB 内存（推荐）
- 10GB 可用磁盘空间

#### 安装步骤

**1. 克隆项目**
```bash
git clone https://github.com/your-username/user-behavior-analysis.git
cd user-behavior-analysis
```

**2. 安装依赖**
```bash
pip install pandas numpy
```

或使用requirements.txt：
```bash
pip install -r requirements.txt
```

**3. 准备数据**
- 下载UserBehavior.csv数据集
- 将文件放到合适的位置
- 修改`user_behavior_analysis.py`第37行，设置正确的文件路径：
```python
csv_path = r"你的文件路径\UserBehavior.csv"
```

#### 运行程序

```bash
python user_behavior_analysis.py
```

程序将自动完成以下步骤：
1. 分块加载数据（进度显示）
2. 数据预处理和清洗
3. 执行4项分析任务
4. 保存结果到output目录
5. 输出分析摘要到控制台

### 输出结果

程序会在`output/`目录下生成5个CSV文件：

#### 1. conversion_funnel.csv - 转化漏斗数据
```csv
behavior_type,user_count
pv,987654
fav,123456
cart,234567
buy,98765
```

#### 2. user_rfm.csv - 用户RFM分析
```csv
user_id,recency,frequency,monetary,r_score,f_score,m_score,rfm_score,user_segment
12345,5,10,10,3,3,3,9,高价值用户
67890,30,3,3,2,2,2,6,中价值用户
```

#### 3. anomaly_users.csv - 异常用户行为
```csv
user_id,date,behavior_type,count
88888,2017-12-01,pv,156
99999,2017-12-02,buy,78
```

#### 4. item_recommendations.csv - 商品推荐
```csv
item_id_1,item_id_2,co_occurrence,rank
1001,1002,450,1
1001,1003,380,2
```

#### 5. user_recommendations.csv - 用户个性化推荐
```csv
user_id,recommended_item,score,rank
12345,5678,450,1
12345,9012,380,2
```

### 性能优化

项目采用多项优化技术，确保能处理GB级大文件：

1. **分块读取**：每次读取100万行，避免内存溢出
2. **数据类型优化**：使用int32和category类型，减少50%内存占用
3. **采样计算**：推荐系统采样10万条数据，平衡效率和准确性
4. **增量处理**：边读边处理，不一次性加载全部数据

**实测性能**（16GB内存）：
- 数据量：3.41GB（约1亿条记录）
- 运行时间：约5-10分钟
- 峰值内存：约8-10GB

### 参数调整

可根据实际需求调整以下参数：

#### 异常检测阈值（第135行）
```python
anomaly = daily_behavior[daily_behavior['count'] > 50]  # 改为其他值
```

#### 推荐采样大小（第159行）
```python
sample_size = min(100000, len(user_item))  # 增大或减小采样数
```

#### 推荐数量（第174、197行）
```python
top_similar = item_similarity.groupby('item_id_1', observed=True).head(5)  # 改为TOP10
recs = recs.nlargest(5, 'co_occurrence')  # 改为其他数量
```

#### RFM分层阈值（第116行）
```python
bins=[0, 5, 7, 9]  # 调整分层边界
```

### 常见问题

**Q1: 内存不足怎么办？**

减少分块大小或推荐采样数：
```python
chunk_size = 500000  # 第34行，默认100万
sample_size = min(50000, len(user_item))  # 第159行，默认10万
```

**Q2: 运行太慢怎么办？**

增大分块大小（如果内存充足）：
```python
chunk_size = 2000000  # 第34行
```

**Q3: 如何只运行部分分析？**

注释掉不需要的部分，每个分析模块独立，可单独运行。

**Q4: 支持其他数据集吗？**

支持，只需确保CSV格式包含5个必要字段即可。

### 项目结构

```
user-behavior-analysis/
├── user_behavior_analysis.py  # 主程序
├── README.md                   # 项目说明
├── requirements.txt            # 依赖列表
├── .gitignore                  # Git忽略文件
└── output/                     # 输出目录（自动创建）
    ├── conversion_funnel.csv
    ├── user_rfm.csv
    ├── anomaly_users.csv
    ├── item_recommendations.csv
    └── user_recommendations.csv
```

### 技术文档

#### 算法说明

**RFM模型**：
- 使用`pd.cut()`进行等频分箱，将连续值离散化为1-3分
- Recency分数与天数负相关（越近越高）
- Frequency和Monetary分数与次数正相关

**协同过滤**：
- 基于物品的协同过滤（Item-based CF）
- 通过用户共同购买行为计算商品相似度
- 使用共现次数（co-occurrence）作为相似度度量

### 后续优化方向

- [ ] 支持增量更新，无需重新分析全部数据
- [ ] 添加可视化图表（转化漏斗图、用户分布图等）
- [ ] 引入ALS矩阵分解，提升推荐准确度
- [ ] 增加更多异常检测规则（如频繁退货、恶意评论等）
- [ ] 支持多种数据源（MySQL、PostgreSQL等）

### 贡献指南

欢迎提交Issue和Pull Request！

### 开源协议

MIT License

### 联系方式

- 作者：louisvikk6-max
- 邮箱：Liuchenyan102938@163.com

---

⭐ 如果这个项目对你有帮助，请给个Star支持一下！

---

## English

A Pandas-based e-commerce user behavior analysis system providing conversion analysis, high-value user identification, anomaly detection, and intelligent recommendation features.

### Project Overview

This project provides in-depth analysis of e-commerce user behavior data, helping operators to:
- Understand user conversion paths and optimize operational strategies
- Identify high-value users for precision marketing
- Detect abnormal behaviors and prevent fraud risks
- Generate personalized recommendations to improve user experience

**Tech Stack:** Python + Pandas + NumPy

**Use Cases:** E-commerce platforms, online retail, content recommendation, etc.

### Core Features

#### 1. User Conversion Funnel Analysis
Analyze the complete path from browsing to purchase and calculate conversion rates at each stage:
- **Browse(pv)** → **Favorite(fav)** → **Add to Cart(cart)** → **Purchase(buy)**
- Output user count and conversion rate for each stage
- Help identify conversion bottlenecks

#### 2. High-Value User Identification (RFM Model)
User value assessment based on the classic RFM model:
- **R (Recency)**: Days since last purchase
- **F (Frequency)**: Purchase frequency
- **M (Monetary)**: Purchase amount (replaced by purchase count)

Automatically classify users into three tiers:
- High-value users (RFM score ≥ 8)
- Medium-value users (6 ≤ RFM score < 8)
- Low-value users (RFM score < 6)

#### 3. Anomaly Detection
Identify potential fraudulent behaviors such as fake orders:
- Detect abnormally high-frequency operations per day (threshold: 50 times/day)
- Output list of anomalous users with behavior details
- Support customizable detection thresholds

#### 4. Product Recommendation System
Intelligent recommendations based on collaborative filtering:
- **Product Similarity Recommendation**: Find TOP5 most similar products for each item
- **User Personalized Recommendation**: Generate personalized recommendation lists based on user purchase history
- Automatically filter out products already purchased by users

### Dataset Description

#### Data Source
This project uses Alibaba's public Taobao user behavior dataset:
- **Dataset Name**: User Behavior Data from Taobao for Recommendation
- **Source**: Alibaba Cloud Tianchi
- **Download**: https://tianchi.aliyun.com/dataset/649

#### Input Data Format
Dataset (UserBehavior.csv) contains 5 fields per row:

| Field | Type | Description |
|-------|------|-------------|
| user_id | int | User ID |
| item_id | int | Product ID |
| category_id | int | Category ID |
| behavior_type | string | Behavior type (pv/fav/cart/buy) |
| timestamp | int | Unix timestamp |

#### Behavior Type Description
- `pv`: Page View
- `fav`: Favorite
- `cart`: Add to Cart
- `buy`: Purchase

### Quick Start

#### Requirements
- Python 3.7 or higher
- 16GB RAM (recommended)
- 10GB available disk space

#### Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/user-behavior-analysis.git
cd user-behavior-analysis
```

**2. Install dependencies**
```bash
pip install pandas numpy
```

Or using requirements.txt:
```bash
pip install -r requirements.txt
```

**3. Prepare data**
- Download UserBehavior.csv dataset
- Place the file in an appropriate location
- Modify line 37 in `user_behavior_analysis.py` to set the correct file path:
```python
csv_path = r"your_file_path\UserBehavior.csv"
```

#### Run the Program

```bash
python user_behavior_analysis.py
```

The program will automatically:
1. Load data in chunks (with progress display)
2. Preprocess and clean data
3. Execute 4 analysis tasks
4. Save results to output directory
5. Output analysis summary to console

### Output Results

The program generates 5 CSV files in the `output/` directory:

#### 1. conversion_funnel.csv - Conversion Funnel Data
```csv
behavior_type,user_count
pv,987654
fav,123456
cart,234567
buy,98765
```

#### 2. user_rfm.csv - User RFM Analysis
```csv
user_id,recency,frequency,monetary,r_score,f_score,m_score,rfm_score,user_segment
12345,5,10,10,3,3,3,9,High-value User
67890,30,3,3,2,2,2,6,Medium-value User
```

#### 3. anomaly_users.csv - Anomalous User Behavior
```csv
user_id,date,behavior_type,count
88888,2017-12-01,pv,156
99999,2017-12-02,buy,78
```

#### 4. item_recommendations.csv - Product Recommendations
```csv
item_id_1,item_id_2,co_occurrence,rank
1001,1002,450,1
1001,1003,380,2
```

#### 5. user_recommendations.csv - User Personalized Recommendations
```csv
user_id,recommended_item,score,rank
12345,5678,450,1
12345,9012,380,2
```

### Performance Optimization

The project employs multiple optimization techniques to handle GB-level large files:

1. **Chunked Reading**: Read 1 million rows at a time to avoid memory overflow
2. **Data Type Optimization**: Use int32 and category types to reduce memory usage by 50%
3. **Sampled Computation**: Sample 100k records for recommendations to balance efficiency and accuracy
4. **Incremental Processing**: Process data while reading, not loading all at once

**Performance Test** (16GB RAM):
- Data size: 3.41GB (about 100 million records)
- Runtime: About 5-10 minutes
- Peak memory: About 8-10GB

### Parameter Adjustment

Adjust the following parameters according to actual needs:

#### Anomaly Detection Threshold (Line 135)
```python
anomaly = daily_behavior[daily_behavior['count'] > 50]  # Change to other value
```

#### Recommendation Sample Size (Line 159)
```python
sample_size = min(100000, len(user_item))  # Increase or decrease sample size
```

#### Recommendation Count (Lines 174, 197)
```python
top_similar = item_similarity.groupby('item_id_1', observed=True).head(5)  # Change to TOP10
recs = recs.nlargest(5, 'co_occurrence')  # Change to other count
```

#### RFM Segmentation Threshold (Line 116)
```python
bins=[0, 5, 7, 9]  # Adjust segmentation boundaries
```

### FAQ

**Q1: What if memory is insufficient?**

Reduce chunk size or recommendation sample size:
```python
chunk_size = 500000  # Line 34, default 1 million
sample_size = min(50000, len(user_item))  # Line 159, default 100k
```

**Q2: What if the program runs too slowly?**

Increase chunk size (if memory is sufficient):
```python
chunk_size = 2000000  # Line 34
```

**Q3: How to run only partial analysis?**

Comment out unnecessary parts. Each analysis module is independent and can run separately.

**Q4: Does it support other datasets?**

Yes, as long as the CSV format contains the 5 required fields.

### Project Structure

```
user-behavior-analysis/
├── user_behavior_analysis.py  # Main program
├── README.md                   # Project documentation
├── requirements.txt            # Dependency list
├── .gitignore                  # Git ignore file
└── output/                     # Output directory (auto-created)
    ├── conversion_funnel.csv
    ├── user_rfm.csv
    ├── anomaly_users.csv
    ├── item_recommendations.csv
    └── user_recommendations.csv
```

### Technical Documentation

#### Algorithm Description

**RFM Model**:
- Use `pd.cut()` for equal-frequency binning, discretizing continuous values into 1-3 scores
- Recency score is negatively correlated with days (more recent = higher)
- Frequency and Monetary scores are positively correlated with count

**Collaborative Filtering**:
- Item-based Collaborative Filtering (Item-based CF)
- Calculate product similarity through users' co-purchase behavior
- Use co-occurrence count as similarity metric

### Future Improvements

- [ ] Support incremental updates without reanalyzing all data
- [ ] Add visualization charts (funnel charts, user distribution charts, etc.)
- [ ] Introduce ALS matrix factorization to improve recommendation accuracy
- [ ] Add more anomaly detection rules (frequent returns, malicious reviews, etc.)
- [ ] Support multiple data sources (MySQL, PostgreSQL, etc.)

### Contributing

Issues and Pull Requests are welcome!

### License

MIT License

### Contact

- Author: louisvikk6-max
- Email: Liuchenyan102938@163.com

---

⭐ If this project helps you, please give it a Star!
