import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import os

# 基础设置
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 加载你的数据
file_path = '/Users/keying/Desktop/中菲媒体对比论文撰写/BERT/大表T1T2.xlsx'
df = pd.read_excel(file_path)
df = df.dropna(subset=['Content', 'Country', 'Date']).reset_index(drop=True)
docs = df['Content'].astype(str).tolist()

# 固定不变的组件
embedding_model = SentenceTransformer("all-mpnet-base-v2")
custom_stop_words = list(ENGLISH_STOP_WORDS) + [
    'china', 'philippine', 'philippines', 'sea', 'south', 'chinese',
    'said', 'maritime', 'water', 'beijing', 'manila', 'island',
    'south china', 'china sea'
]
vectorizer_model = CountVectorizer(stop_words=custom_stop_words, ngram_range=(1, 2))

# ==============================================================================
# 网格搜索：遍历参数组合，找最优解
# ==============================================================================
# 待测试的参数范围（覆盖你之前试过的所有值）
n_neighbors_list = [15, 20, 25]
min_cluster_size_list = [25, 30, 35]

# 存储结果
results = []

print("开始网格搜索最优参数...")
for n_neighbors in n_neighbors_list:
    for min_cluster_size in min_cluster_size_list:
        print(f"\n测试参数：n_neighbors={n_neighbors}, min_cluster_size={min_cluster_size}")

        # 初始化模型
        umap_model = UMAP(n_neighbors=n_neighbors, n_components=5, min_dist=0.0, metric='cosine',
                          random_state=RANDOM_STATE)
        hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=n_neighbors,
                                cluster_selection_method='eom', prediction_data=True)
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            calculate_probabilities=False,
            verbose=False
        )

        # 训练
        topics, _ = topic_model.fit_transform(docs)

        # 计算指标
        valid_topics = [t for t in set(topics) if t != -1]
        noise_ratio = (pd.Series(topics) == -1).mean() * 100

        # 计算C_V分数
        topics_words = []
        for topic in sorted(valid_topics):
            words = [word for word, _ in topic_model.get_topic(topic)]
            if words:
                topics_words.append(words)
        texts = [doc.split() for doc in docs]
        dictionary = Dictionary(texts)
        cm = CoherenceModel(topics=topics_words, texts=texts, dictionary=dictionary, coherence='c_v', processes=1)
        coherence_score = cm.get_coherence()

        # 保存结果
        results.append({
            "n_neighbors": n_neighbors,
            "min_cluster_size": min_cluster_size,
            "有效主题数": len(valid_topics),
            "C_V一致性分数": round(coherence_score, 3),
            "噪音占比(%)": round(noise_ratio, 2)
        })

        print(f"   有效主题数：{len(valid_topics)}, C_V分数：{coherence_score:.3f}, 噪音比：{noise_ratio:.2f}%")

# 输出最优结果
results_df = pd.DataFrame(results)
results_df.to_excel("/Users/keying/Desktop/中菲媒体对比论文撰写/BERT/参数网格搜索结果.xlsx", index=False)
print("\n✅ 网格搜索完成！结果已保存，你可以直接看哪一组参数最优！")
print("\n所有参数组合结果：")
print(results_df.sort_values("C_V一致性分数", ascending=False))