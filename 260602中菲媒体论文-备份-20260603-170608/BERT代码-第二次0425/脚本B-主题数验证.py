# ==============================================================================
# 脚本 B：主题数验证（约10-15分钟）
# 前提：已运行脚本A，cache_embeddings.npy 存在
# 作用：验证主题数量和质量，人工确认后再运行脚本C输出全部结果
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
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

output_dir = '/Users/keying/Desktop/中菲媒体对比论文撰写/BERT'

# ── 加载缓存（脚本A的输出）─────────────────────────────────────────────────────
print("="*60)
print("脚本B：主题数验证")
print("="*60)

cache_emb = os.path.join(output_dir, 'cache_embeddings.npy')
cache_df  = os.path.join(output_dir, 'cache_df_clean.pkl')
if not os.path.exists(cache_emb) or not os.path.exists(cache_df):
    raise FileNotFoundError("❌ 找不到缓存文件，请先运行 脚本A！")

embeddings = np.load(cache_emb)
df         = pd.read_pickle(cache_df)
docs       = df['Content'].astype(str).tolist()
countries  = df['Country'].astype(str).tolist()
timestamps = df['Date'].astype(str).tolist()

print(f"✅ 已加载缓存：{len(docs)} 条文本，向量维度 {embeddings.shape}")

# ── 算法组件（与脚本C完全一致，保证结果可复现）──────────────────────────────
embedding_model = SentenceTransformer("all-mpnet-base-v2")  # 仅用于BERTopic接口

umap_model = UMAP(
    n_neighbors=15, n_components=5, min_dist=0.0,
    metric="cosine", random_state=RANDOM_STATE
)
hdbscan_model = HDBSCAN(
    min_cluster_size=25, min_samples=10,
    cluster_selection_method="eom", prediction_data=True
)

custom_stop_words = list(ENGLISH_STOP_WORDS) + [
    'said', 'maritime', 'water', 'beijing', 'manila', 'island',
    'also', 'would', 'could', 'year', 'years', 'time', 'new',
    'one', 'two', 'three', 'may', 'will', 'like', 'use',
]
def custom_preprocessor(text):
    text = text.lower()
    text = re.sub(r'\bsouth china sea\b', '', text)
    text = re.sub(r'\bsouth china\b',     '', text)
    text = re.sub(r'\bchina sea\b',       '', text)
    return text

vectorizer_model = CountVectorizer(
    stop_words=custom_stop_words, ngram_range=(1, 2),
    preprocessor=custom_preprocessor,
    token_pattern=r'[a-zA-Z]{2,}', min_df=3,
)

# ── 模型训练（直接传入预计算的embeddings，跳过编码，快！）──────────────────
print("\n开始训练（使用缓存向量，约3-5分钟）...")
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    calculate_probabilities=False,
    verbose=True
)

# 关键：传入 embeddings 参数，BERTopic 直接跳过编码步骤
topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
init_num  = len(set(topics)) - (1 if -1 in topics else 0)
print(f"\n✅ 初始主题数：{init_num} 个")

# ── 数据驱动合并 ──────────────────────────────────────────────────────────────
print("\n正在 auto 模式合并...")
topic_model.reduce_topics(docs, nr_topics="auto")
auto_topics = topic_model.topics_
AUTO_COUNT  = len([t for t in set(auto_topics) if t != -1])
print(f"✅ auto 合并结果：{AUTO_COUNT} 个主题")

if AUTO_COUNT > 20:
    FINAL_TARGET = 15
    print(f"  auto结果偏多，进一步合并至 {FINAL_TARGET} 个...")
    topic_model.reduce_topics(docs, nr_topics=FINAL_TARGET)
    print(f"  ⚠️  论文需注明：auto得到{AUTO_COUNT}个，分析需要合并至{FINAL_TARGET}个")
    FINAL_TARGET_USED = FINAL_TARGET
else:
    FINAL_TARGET_USED = AUTO_COUNT
    print(f"  auto结果在合理区间，直接使用")

topic_model.update_topics(docs, vectorizer_model=vectorizer_model)
new_topics       = topic_model.topics_
new_valid_topics = sorted([t for t in set(new_topics) if t != -1])
topic_info       = topic_model.get_topic_info()

# ── 打印主题列表，供人工核查 ──────────────────────────────────────────────────
print("\n" + "="*60)
print(f"📌 主题列表（共 {len(new_valid_topics)} 个）")
print("="*60)
print(topic_info[['Topic', 'Count', 'Name']].to_string(index=False))

# ── 快速C_V验证 ───────────────────────────────────────────────────────────────
print("\n计算 C_V 主题一致性（约2-3分钟）...")
analyzer     = vectorizer_model.build_analyzer()
texts        = [analyzer(doc) for doc in docs]
dictionary   = Dictionary(texts)
topics_words = [[w for w, _ in topic_model.get_topic(t)]
                for t in new_valid_topics if topic_model.get_topic(t)]

cm_cv    = CoherenceModel(topics=topics_words, texts=texts,
                          dictionary=dictionary, coherence='c_v', processes=1)
cv_score = cm_cv.get_coherence()
noise_ratio = (pd.Series(new_topics) == -1).mean() * 100

print("\n" + "="*60)
print("验证结果摘要")
print("="*60)
print(f"有效主题数  : {len(new_valid_topics)}")
print(f"C_V 一致性  : {cv_score:.3f}  （良好≥0.5，优秀≥0.65）")
print(f"噪音比例    : {noise_ratio:.2f}%  （合理范围 15-30%）")
print("="*60)

# ── 保存模型供脚本C使用（不用重跑！）────────────────────────────────────────
with open(os.path.join(output_dir, 'cache_model_B.pkl'), 'wb') as f:
    pickle.dump(topic_model, f)
# 同时保存关键变量
np.save(os.path.join(output_dir, 'cache_topics_B.npy'), np.array(new_topics))
pd.DataFrame({'AUTO_COUNT': [AUTO_COUNT],
              'FINAL_TARGET': [FINAL_TARGET_USED],
              'cv_score': [cv_score],
              'noise_ratio': [noise_ratio]}).to_pickle(
    os.path.join(output_dir, 'cache_meta_B.pkl')
)
print("\n✅ 模型已缓存，脚本C可直接加载，无需重新训练")

# ── 人工决策提示 ──────────────────────────────────────────────────────────────
print("""
══════════════════════════════════════════════════════════
请检查上方主题列表，确认：
  1. 主题数量是否合理？（建议 10-20 个）
  2. 各主题关键词是否有明确语义？
  3. C_V ≥ 0.5 且噪音 < 35%？

如果满意 → 直接运行 脚本C，输出全部论文结果
如果不满意 → 修改本脚本中的参数后重新运行（B脚本约10分钟）
             不需要重跑脚本A！
══════════════════════════════════════════════════════════
""")
