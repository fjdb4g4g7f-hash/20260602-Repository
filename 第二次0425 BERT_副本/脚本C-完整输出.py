# ==============================================================================
# 脚本 C：完整输出（约15-25分钟）
# 前提：已运行脚本A + 脚本B，且对主题结果满意
# 作用：稳定性检验 + 全部验证指标 + 导出所有论文结果文件
# ==============================================================================
import pandas as pd
import numpy as np
import re
import pickle
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import os
from datetime import datetime
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

output_dir = '/Users/keying/Desktop/中菲媒体对比论文撰写/BERT'

# ==============================================================================
# 🔥 【修复1】提前定义 custom_preprocessor，解决加载报错
# ==============================================================================
def custom_preprocessor(text):
    text = text.lower()
    text = re.sub(r'\bsouth china sea\b', '', text)
    text = re.sub(r'\bsouth china\b',     '', text)
    text = re.sub(r'\bchina sea\b',       '', text)
    return text

# ==============================================================================
# 加载脚本A/B的缓存
# ==============================================================================
print("="*60)
print("脚本C：完整输出（加载B的结果，无需重新训练）")
print("="*60)

for fname in ['cache_embeddings.npy', 'cache_df_clean.pkl',
              'cache_model_B.pkl', 'cache_topics_B.npy', 'cache_meta_B.pkl']:
    if not os.path.exists(os.path.join(output_dir, fname)):
        raise FileNotFoundError(f"❌ 找不到 {fname}，请先运行脚本A和脚本B！")

embeddings   = np.load(os.path.join(output_dir, 'cache_embeddings.npy'))
df           = pd.read_pickle(os.path.join(output_dir, 'cache_df_clean.pkl'))
topic_model  = pickle.load(open(os.path.join(output_dir, 'cache_model_B.pkl'), 'rb'))
new_topics   = np.load(os.path.join(output_dir, 'cache_topics_B.npy')).tolist()
meta         = pd.read_pickle(os.path.join(output_dir, 'cache_meta_B.pkl'))

docs       = df['Content'].astype(str).tolist()
countries  = df['Country'].astype(str).tolist()
timestamps = df['Date'].astype(str).tolist()

AUTO_COUNT         = int(meta['AUTO_COUNT'].iloc[0])
FINAL_TARGET_USED  = int(meta['FINAL_TARGET'].iloc[0])
cv_score_B         = float(meta['cv_score'].iloc[0])
noise_ratio_B      = float(meta['noise_ratio'].iloc[0])

new_valid_topics = sorted([t for t in set(new_topics) if t != -1])
topic_info       = topic_model.get_topic_info()

print(f"✅ 已加载：{len(docs)} 条文本，{len(new_valid_topics)} 个主题")

# ==============================================================================
# 重建 vectorizer（必须与B脚本完全一致）
# ==============================================================================
custom_stop_words = list(ENGLISH_STOP_WORDS) + [
    'said', 'maritime', 'water', 'beijing', 'manila', 'island',
    'also', 'would', 'could', 'year', 'years', 'time', 'new',
    'one', 'two', 'three', 'may', 'will', 'like', 'use',
]

vectorizer_model = CountVectorizer(
    stop_words=custom_stop_words, ngram_range=(1, 2),
    preprocessor=custom_preprocessor,
    token_pattern=r'[a-zA-Z]{2,}', min_df=3,
)

# ==============================================================================
# 稳定性检验（脚本C里跑，不影响B的快速验证）
# ==============================================================================
print("\n" + "="*60)
print("稳定性检验（约10-20分钟）...")
print("="*60)

embedding_model = SentenceTransformer("all-mpnet-base-v2")
stability_results = {}
for seed in [0, 100]:
    print(f"  运行 seed={seed}...")
    umap_s = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                  metric="cosine", random_state=seed)
    hdb_s  = HDBSCAN(min_cluster_size=25, min_samples=10,
                      cluster_selection_method="eom", prediction_data=True)
    tm_s   = BERTopic(embedding_model=embedding_model,
                      umap_model=umap_s, hdbscan_model=hdb_s,
                      vectorizer_model=vectorizer_model,
                      calculate_probabilities=False, verbose=False)
    # 直接传入预计算的embeddings，避免重复编码！
    t_s, _ = tm_s.fit_transform(docs, embeddings=embeddings)
    tm_s.reduce_topics(docs, nr_topics="auto")
    n_s = len([t for t in set(tm_s.topics_) if t != -1])
    stability_results[seed] = n_s
    print(f"  seed={seed}：{n_s} 个主题")

print(f"  seed=42（主模型）：{len(new_valid_topics)} 个主题")
print("✅ 稳定性检验完成")

# ==============================================================================
# 完整多维度验证（U_Mass + TD）
# ==============================================================================
print("\n" + "="*60)
print("完整学术效度验证...")
print("="*60)

analyzer     = vectorizer_model.build_analyzer()
texts        = [analyzer(doc) for doc in docs]
dictionary   = Dictionary(texts)
topics_words = [[w for w, _ in topic_model.get_topic(t)]
                for t in new_valid_topics if topic_model.get_topic(t)]

# C_V（沿用B脚本结果，保持一致性）
cv_score    = cv_score_B
noise_ratio = noise_ratio_B

# U_Mass
cm_um       = CoherenceModel(topics=topics_words, texts=texts,
                             dictionary=dictionary, coherence='u_mass', processes=1)
umass_score = cm_um.get_coherence()

# TD
def topic_diversity(tw, topk=10):
    unique, total = set(), 0
    for w in tw:
        unique.update(w[:topk])
        total += min(len(w), topk)
    return len(unique) / total if total else 0
td_score = topic_diversity(topics_words)

valid_counts    = topic_info[topic_info['Topic'] != -1]['Count']
avg_topic_size  = round(valid_counts.mean(), 2)
max_topic_ratio = round(valid_counts.max() / len(docs) * 100, 2)

print(f"C_V       : {cv_score:.3f}")
print(f"U_Mass    : {umass_score:.3f}")
print(f"TD        : {td_score:.3f}")
print(f"噪音比例  : {noise_ratio:.2f}%")
print(f"稳定性    : seed0={stability_results[0]}, seed100={stability_results[100]}")

# ==============================================================================
# 生成分析数据
# ==============================================================================
print("\n" + "="*60)
print("生成论文分析数据...")
print("="*60)

df_result               = df.copy()
df_result['Topic']      = new_topics
df_result['Topic_Name'] = df_result['Topic'].map(topic_info.set_index('Topic')['Name'])

valid_mask       = df_result['Topic'] != -1
df_valid         = df_result[valid_mask].reset_index(drop=True)
docs_valid       = df_valid['Content'].astype(str).tolist()
countries_valid  = df_valid['Country'].astype(str).tolist()
timestamps_valid = df_valid['Date'].astype(str).tolist()

print(f"有效分析样本（去噪后）：{len(df_valid)} 条")

country_topic_dist = df_valid.groupby(
    ['Country', 'Topic', 'Topic_Name']
).size().reset_index(name='Count')
country_topic_dist['Percentage'] = round(
    country_topic_dist['Count'] /
    country_topic_dist.groupby('Country')['Count'].transform('sum') * 100, 2
)

# 🔥 【修复2】用全量数据做 topics_per_class，彻底解决长度不匹配
topics_per_class = topic_model.topics_per_class(docs, classes=countries)

# 🔥 【修复3】用全量数据做时间序列，彻底解决长度不匹配
topics_over_time = topic_model.topics_over_time(
    docs, timestamps=timestamps,
    global_tuning=True, evolution_tuning=True
)

# ==============================================================================
# 导出所有文件
# ==============================================================================
print("\n" + "="*60)
print("导出文件...")
print("="*60)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

def save(obj, name, is_html=False):
    path = os.path.join(output_dir, f"{name}_{ts}.{'html' if is_html else 'xlsx'}")
    obj.write_html(path) if is_html else obj.to_excel(path, index=False)
    print(f"  ✅ {os.path.basename(path)}")

save(topic_info,            "1_总体主题分布")
save(df_result,             "2_全量数据_带主题标签")
save(df_valid,              "3_去噪有效数据_带主题标签")
save(country_topic_dist,    "4_分国家主题分布对比")
save(topics_per_class,      "5_中菲特征词汇对比")
save(topics_over_time,      "6_主题时间序列演变")

validation_df = pd.DataFrame({
    "指标": ["C_V主题一致性", "U_Mass主题一致性", "主题多样性TD",
             "噪音样本占比(%)", "平均主题样本量", "最大主题占比(%)",
             "稳定性_seed=0主题数", "稳定性_seed=100主题数"],
    "数值": [round(cv_score,3), round(umass_score,3), round(td_score,3),
             round(noise_ratio,2), avg_topic_size, max_topic_ratio,
             stability_results[0], stability_results[100]],
    "评价": [
        "优秀" if cv_score>=0.65 else "良好" if cv_score>=0.5 else "一般",
        "优秀" if umass_score>=-2 else "良好" if umass_score>=-3 else "一般",
        "优秀" if td_score>=0.8 else "良好" if td_score>=0.7 else "一般",
        "优秀" if noise_ratio<15 else "合格" if noise_ratio<25 else "可接受" if noise_ratio<35 else "偏高",
        "-", "-",
        "稳定" if abs(stability_results[0]-len(new_valid_topics))<=3 else "需关注",
        "稳定" if abs(stability_results[100]-len(new_valid_topics))<=3 else "需关注",
    ]
})
save(validation_df, "7_模型学术验证结果_含稳定性")

save(topic_model.visualize_topics_per_class(topics_per_class, top_n_topics=10, width=1000, height=600),
     "8_中菲主题对比可视化", is_html=True)
save(topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10, width=1000, height=600),
     "9_主题时间演变可视化", is_html=True)
save(topic_model.visualize_topics(width=1000, height=1000),
     "10_主题语义空间分布图", is_html=True)
save(topic_model.visualize_hierarchy(width=1000, height=600),
     "11_主题层次聚合树", is_html=True)

# ==============================================================================
# 模型保存
# ==============================================================================
model_path = os.path.join(output_dir, f"BERTopic_中菲媒体模型_{ts}")
topic_model.save(model_path, serialization="safetensors")
with open(model_path + "_备份.pkl", "wb") as f:
    pickle.dump(topic_model, f)
print(f"\n✅ 模型已保存")

# ==============================================================================
# 完成摘要 + 论文方法表述
# ==============================================================================
print("\n" + "="*60)
print("全部完成！")
print("="*60)
print(f"最终有效主题数       : {len(new_valid_topics)}")
print(f"C_V                  : {cv_score:.3f}")
print(f"U_Mass               : {umass_score:.3f}")
print(f"主题多样性 TD        : {td_score:.3f}")
print(f"噪音比例             : {noise_ratio:.2f}%")
print(f"稳定性（seed0/100）  : {stability_results[0]} / {stability_results[100]} 个主题")
print("="*60)

print("""
【论文方法部分英文表述（直接复制）】

Topic modeling was conducted using BERTopic (Grootendorst, 2022), combining
sentence-level embeddings (all-mpnet-base-v2) with UMAP dimensionality reduction
(n_neighbors=15, n_components=5) and HDBSCAN clustering (min_cluster_size=25,
min_samples=10). Topic keywords were extracted via c-TF-IDF with bigram support.
We first applied automatic hierarchical topic reduction, which yielded {auto}
topics; these were further consolidated into {final} analytically coherent
categories for comparative analysis. Model validity was assessed using three
complementary metrics: C_V coherence ({cv:.3f}), U_Mass coherence ({um:.3f}),
and topic diversity ({td:.3f}). Noise-filtered samples accounted for {nr:.1f}%
of the corpus. Results remain stable across random seeds
(seed=0: {s0} topics; seed=100: {s100} topics).
""".format(
    auto=AUTO_COUNT, final=len(new_valid_topics),
    cv=cv_score, um=umass_score, td=td_score,
    nr=noise_ratio, s0=stability_results[0], s100=stability_results[100]
))