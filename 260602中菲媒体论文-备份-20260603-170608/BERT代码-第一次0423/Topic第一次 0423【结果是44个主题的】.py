import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import os
from datetime import datetime

# 学术严谨性设置：全局随机种子固定，保证结果可复现
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("步骤 1/5：数据加载与预处理")
print("="*80)

# 请确保这里的路径指向你已经删掉 T3 的纯净版 Excel 文件
file_path = '/Users/keying/Desktop/中菲媒体对比论文撰写/BERT/大表T1T2.xlsx'
output_dir = '/Users/keying/Desktop/中菲媒体对比论文撰写/BERT'

if not os.path.exists(file_path):
    raise FileNotFoundError(f"未找到文件，请检查路径：{file_path}")

try:
    df = pd.read_excel(file_path)
    print(f"✅ 数据加载成功！当前数据量：{len(df)} 条")
except Exception as e:
    raise IOError(f"Excel文件读取失败：{e}")

# 核心数据完整性检查
required_columns = ['Content', 'Country', 'Date']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"数据缺少必要列：{col}，请检查Excel表头")

df = df.dropna(subset=['Content', 'Country', 'Date']).reset_index(drop=True)
df['Content'] = df['Content'].astype(str)
df['Country'] = df['Country'].astype(str)

try:
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
except Exception as e:
    raise ValueError(f"日期列解析失败：{e}")

docs = df['Content'].tolist()
countries = df['Country'].tolist()
timestamps = df['Date'].tolist()

print("\n" + "="*80)
print("步骤 2/5：核心算法组件初始化 (引入定制化领域停用词)")
print("="*80)

embedding_model = SentenceTransformer("all-mpnet-base-v2")

# 降维（固定，适配新闻文本）
umap_model = UMAP(
    n_neighbors=20,       # 放大！解决文本碎拆，核心修复
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)

hdbscan_model = HDBSCAN(
    min_cluster_size=30,  # 5815样本黄金门槛
    min_samples=20,       # 强过滤碎主题
    cluster_selection_method="eom",
    prediction_data=True
)

# 构建南海领域专属停用词表，剔除无区分度的高频词
custom_stop_words = list(ENGLISH_STOP_WORDS) + [
    'china', 'philippine', 'philippines', 'sea', 'south', 'chinese',
    'said', 'maritime', 'water', 'beijing', 'manila', 'island',
    'south china', 'china sea'
]

vectorizer_model = CountVectorizer(
    stop_words=custom_stop_words,
    ngram_range=(1, 2)
)

print("✅ 算法组件初始化完成")

print("\n" + "="*80)
print("步骤 3/5：BERTopic 模型训练")
print("="*80)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    calculate_probabilities=False,
    verbose=True
)

topics, _ = topic_model.fit_transform(docs)
print("✅ 模型训练完成！")

print("\n" + "="*80)
print("步骤 4/5：结果分析与数据导出")
print("="*80)

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# 4.1 基础主题分布
topic_info = topic_model.get_topic_info()
save_path_1 = os.path.join(output_dir, f"1_总体主题分布_{timestamp_str}.xlsx")
topic_info.to_excel(save_path_1, index=False)
print(f"✅ 已保存总体主题分布至：{save_path_1}")

# 4.2 全量数据贴标签
df_result = df.copy()
df_result['Topic'] = topics
df_result['Topic_Name'] = df_result['Topic'].map(topic_model.get_topic_info().set_index('Topic')['Name'])
save_path_2 = os.path.join(output_dir, f"2_全量数据_带主题标签_{timestamp_str}.xlsx")
df_result.to_excel(save_path_2, index=False)

# 4.3 分国家主题分布与特征词对比
country_topic_dist = df_result.groupby(['Country', 'Topic']).size().reset_index(name='Count')
country_topic_dist['Percentage'] = country_topic_dist['Count'] / country_topic_dist.groupby('Country')['Count'].transform('sum') * 100
save_path_3 = os.path.join(output_dir, f"3_分国家主题数量分布_{timestamp_str}.xlsx")
country_topic_dist.to_excel(save_path_3, index=False)

topics_per_class = topic_model.topics_per_class(docs, classes=countries)
save_path_class_words = os.path.join(output_dir, f"3_1_中菲两国特征词汇对比_{timestamp_str}.xlsx")
topics_per_class.to_excel(save_path_class_words, index=False)

fig_classes = topic_model.visualize_topics_per_class(topics_per_class, top_n_topics=10)
save_path_fig_classes = os.path.join(output_dir, f"3_2_中菲词汇对比可视化_{timestamp_str}.html")
fig_classes.write_html(save_path_fig_classes)

# 4.4 时间序列主题演变
topics_over_time = topic_model.topics_over_time(
    docs,
    timestamps=timestamps,
    global_tuning=True,
    evolution_tuning=True
)
save_path_4 = os.path.join(output_dir, f"4_主题时间序列演变_{timestamp_str}.xlsx")
topics_over_time.to_excel(save_path_4, index=False)

fig_time = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
save_path_5 = os.path.join(output_dir, f"5_主题时间演变可视化_{timestamp_str}.html")
fig_time.write_html(save_path_5)

print("\n" + "="*80)
print("步骤 5/5：模型保存")
print("="*80)

model_save_path = os.path.join(output_dir, f"BERTopic_中菲媒体模型_定制版_{timestamp_str}")
topic_model.save(model_save_path, serialization="safetensors")
print(f"✅ 模型已安全保存至：{model_save_path}")
print("\n大功告成！")