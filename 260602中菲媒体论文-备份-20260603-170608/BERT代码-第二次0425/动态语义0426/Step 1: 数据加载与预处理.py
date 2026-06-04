"""
Step 1: 数据加载与预处理
- 读取Excel文件
- 解析日期
- 按事件窗口切割数据
- 输出预处理结果供后续步骤使用
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. 读取数据
# ─────────────────────────────────────────────
FILE_PATH = "/Users/keying/Desktop/中菲媒体对比论文撰写/BERT/大表T1T2.xlsx"

print("正在读取数据...")
df = pd.read_excel(FILE_PATH)

# 检查字段
print(f"数据维度: {df.shape}")
print(f"字段名: {df.columns.tolist()}")
print(f"Country取值: {df['Country'].unique()}")
print(f"Source取值: {df['Source'].unique()}\n")

# ─────────────────────────────────────────────
# 2. 日期解析
# ─────────────────────────────────────────────
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
invalid_dates = df['Date'].isna().sum()
if invalid_dates > 0:
    print(f"警告：有 {invalid_dates} 行日期解析失败，已丢弃")
df = df.dropna(subset=['Date'])

# 去掉正文为空的行
df = df.dropna(subset=['Content'])
df['Content'] = df['Content'].astype(str).str.strip()
df = df[df['Content'].str.len() > 50]  # 过滤过短文本

print(f"清洗后数据量: {df.shape[0]} 篇")
print(f"中国媒体: {(df['Country'] == 'China').sum()} 篇")
print(f"菲律宾媒体: {(df['Country'] == 'Philippine').sum()} 篇\n")

# ─────────────────────────────────────────────
# 3. 定义五个关键事件节点（±14天窗口）
# ─────────────────────────────────────────────
EVENTS = [
    {
        "id": "E1",
        "name": "激光照射事件",
        "name_en": "Laser Incident",
        "date": "2023-02-06",
        "note": "中国海警对菲BRP Malapascua使用军用级激光，菲方透明度策略起点"
    },
    {
        "id": "E2",
        "name": "首次大规模水炮+马科斯声明",
        "name_en": "Water Cannon & Marcos Statement",
        "date": "2023-08-05",
        "note": "菲公开发布视频，马科斯声明从未承诺撤走马德雷山号，报道峰值月"
    },
    {
        "id": "E3",
        "name": "中菲上海磋商",
        "name_en": "Shanghai Consultations",
        "date": "2024-01-17",
        "note": "双边磋商机制第八次会议，最重要缓和节点，检验去安全化是否真实发生"
    },
    {
        "id": "E4",
        "name": "登船夺械事件",
        "name_en": "Boarding & Weapons Seizure",
        "date": "2024-06-17",
        "note": "中国海警登上菲海军橡皮艇，用刀斧毁船夺械，烈度最高单一事件"
    },
    {
        "id": "E5",
        "name": "仁爱礁临时安排达成",
        "name_en": "Provisional Arrangement",
        "date": "2024-07-21",
        "note": "中菲达成仁爱礁补给临时安排，检验缓和后话语是否真的松动"
    },
]

WINDOW_DAYS = 14  # 事件前后各14天

# ─────────────────────────────────────────────
# 4. 为每篇文章标记所属事件窗口
# ─────────────────────────────────────────────
def assign_event_windows(df, events, window_days):
    """
    为每篇文章标记它落在哪个事件的哪个阶段：
    - 'pre'  = 事件前window_days天（基线期）
    - 'post' = 事件后window_days天（响应期）
    一篇文章可能属于多个事件窗口（保留所有匹配）
    """
    records = []
    for _, row in df.iterrows():
        article_date = row['Date']
        for event in events:
            event_date = pd.to_datetime(event['date'])
            pre_start  = event_date - timedelta(days=window_days)
            pre_end    = event_date - timedelta(days=1)
            post_start = event_date
            post_end   = event_date + timedelta(days=window_days)

            if pre_start <= article_date <= pre_end:
                record = row.to_dict()
                record['event_id']   = event['id']
                record['event_name'] = event['name']
                record['event_name_en'] = event['name_en']
                record['event_date'] = event['date']
                record['phase']      = 'pre'
                records.append(record)

            elif post_start <= article_date <= post_end:
                record = row.to_dict()
                record['event_id']   = event['id']
                record['event_name'] = event['name']
                record['event_name_en'] = event['name_en']
                record['event_date'] = event['date']
                record['phase']      = 'post'
                records.append(record)

    return pd.DataFrame(records)


print("正在切割事件窗口...")
df_events = assign_event_windows(df, EVENTS, WINDOW_DAYS)

# 统计各窗口数据量
print("\n各事件窗口文章数量:")
print("-" * 60)
for event in EVENTS:
    eid = event['id']
    subset = df_events[df_events['event_id'] == eid]
    cn_pre  = len(subset[(subset['Country']=='China') & (subset['phase']=='pre')])
    cn_post = len(subset[(subset['Country']=='China') & (subset['phase']=='post')])
    ph_pre  = len(subset[(subset['Country']=='Philippine') & (subset['phase']=='pre')])
    ph_post = len(subset[(subset['Country']=='Philippine') & (subset['phase']=='post')])
    print(f"{eid} {event['name']}")
    print(f"    中国  - 事件前:{cn_pre:3d}篇  事件后:{cn_post:3d}篇")
    print(f"    菲律宾 - 事件前:{ph_pre:3d}篇  事件后:{ph_post:3d}篇")

# ─────────────────────────────────────────────
# 5. 保存预处理结果
# ─────────────────────────────────────────────
OUTPUT_PATH = "/Users/keying/Desktop/中菲媒体对比论文撰写/BERT/preprocessed_events.pkl"
df_events.to_pickle(OUTPUT_PATH)
# 同时保存完整清洗后数据
df.to_pickle("/Users/keying/Desktop/中菲媒体对比论文撰写/BERT/preprocessed_full.pkl")

print(f"\n✅ Step 1 完成")
print(f"   事件窗口数据已保存至: {OUTPUT_PATH}")
print(f"   共 {len(df_events)} 条记录（含跨窗口重复）")
print(f"\n请运行 step2_encode.py")