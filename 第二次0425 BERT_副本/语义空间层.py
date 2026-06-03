# ==============================================================================
# ✅ 学术标准：语境化词嵌入分析（用bert-base-uncased，词级差异必拉开）
# 适配你的数据：Country=China / Philippine（不带s）
# 包含功能：生成【图A：高维语义散点图】 + 【图B：中菲词义距离柱状图】
# ==============================================================================
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import torch
import umap  # 新增：用于生成散点图的降维工具
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
import os

# ===================== 固定配置 =====================
RANDOM_STATE = 42
output_dir = '/Users/keying/Desktop/中菲媒体对比论文撰写/BERT'
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 适配Mac，防止中文乱码
plt.rcParams['axes.unicode_minus'] = False

# 🔥 换用专门捕捉词级语境差异的模型（bert-base-uncased）
model_name = "bert-base-uncased"
print(f"正在加载词级专家模型 {model_name}...")
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

# ===================== 加载数据 =====================
print("正在加载数据缓存...")
df = pd.read_pickle(os.path.join(output_dir, 'cache_df_clean.pkl'))
df['Country'] = df['Country'].astype(str).str.strip()
df = df[df['Country'].isin(['China', 'Philippine'])]

TARGET_WORDS = ["sovereignty", "security", "peace", "defense", "threat", "cooperation", "law"]


# ===================== 提取核心词的语境向量 =====================
def get_word_context_embedding(sentence, target_word):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    embeddings = outputs.last_hidden_state[0].numpy()
    target_tokens = tokenizer.tokenize(target_word)
    target_len = len(target_tokens)
    # 精准寻找该词在句中的位置
    for i in range(len(tokens) - target_len + 1):
        if tokens[i:i + target_len] == target_tokens:
            return embeddings[i:i + target_len].mean(axis=0)  # 若被切分为多个sub-word则求平均
    return None


print("开始提取语境化词向量（大约需 10-20 分钟，请耐心等待）...")
word_embeddings = []
count = 0

for _, row in df.iterrows():
    text = str(row['Content']).strip()
    country = row['Country']
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) >= 40]
    for sent in sentences:
        sent_lower = sent.lower()
        for word in TARGET_WORDS:
            # 词边界匹配，防止抓错
            if re.search(r'\b' + re.escape(word) + r'\b', sent_lower):
                vec = get_word_context_embedding(sent, word)
                if vec is not None:
                    word_embeddings.append({
                        "vector": vec,
                        "keyword": word,
                        "country": country
                    })

    count += 1
    if count % 500 == 0:
        print(f"  已处理 {count} 篇新闻...")

df_vec = pd.DataFrame(word_embeddings)
print("✅ 提取完成 | 成功捕获词向量总数：", len(df_vec))

# ===================== 图 A：生成高维语义散点图 (UMAP) =====================
print("\n正在绘制图A：语义散点图...")
vectors = np.array(df_vec['vector'].tolist())
umap_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE, metric='cosine')
coords = umap_2d.fit_transform(vectors)
df_vec['x'] = coords[:, 0]
df_vec['y'] = coords[:, 1]

plt.figure(figsize=(12, 8))
color_map = {"China": "#2E86AB", "Philippine": "#F24236"}
for c in ['China', 'Philippine']:
    sub = df_vec[df_vec['country'] == c]
    plt.scatter(sub['x'], sub['y'], c=color_map[c], label=c, alpha=0.6, s=35)

# 给每个词的核心位置打上标签
for word in TARGET_WORDS:
    sub = df_vec[df_vec['keyword'] == word]
    if len(sub) > 0:
        # 找该词在空间中的中心点位置进行标注
        center_x = sub['x'].mean()
        center_y = sub['y'].mean()
        plt.annotate(word, (center_x, center_y), fontweight='bold', fontsize=13,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.title("Semantic Space of Core Words: China vs. Philippine Media", fontsize=15, pad=15)
plt.legend(loc="best")
plt.tight_layout()
scatter_path = os.path.join(output_dir, "最终版_图A_中菲语义错位散点图_bert.png")
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
plt.show()

# ===================== 图 B：计算语义距离并生成柱状图 =====================
print("\n正在绘制图B：语义距离对比柱状图...")


def calc_word_distance(df_vec, keyword):
    cn_vec = df_vec[(df_vec['keyword'] == keyword) & (df_vec['country'] == 'China')]['vector'].tolist()
    ph_vec = df_vec[(df_vec['keyword'] == keyword) & (df_vec['country'] == 'Philippine')]['vector'].tolist()
    if len(cn_vec) == 0 or len(ph_vec) == 0:
        return None
    cn_mean = np.array(cn_vec).mean(axis=0)
    ph_mean = np.array(ph_vec).mean(axis=0)
    # 计算余弦距离 (1 - 相似度)
    return round(1 - cosine_similarity([cn_mean], [ph_mean])[0][0], 4)


distance_results = {w: calc_word_distance(df_vec, w) for w in TARGET_WORDS}
result_df = pd.DataFrame(list(distance_results.items()), columns=['Core_Concept', 'China_PH_Semantic_Distance'])
result_df = result_df.sort_values('China_PH_Semantic_Distance', ascending=False).dropna()

plt.figure(figsize=(10, 6))
# 突出距离差异的排序
bars = plt.bar(result_df['Core_Concept'], result_df['China_PH_Semantic_Distance'], color='#4A90E2', edgecolor='black',
               alpha=0.8)

# 在柱子上显示具体数值
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 3), ha='center', va='bottom', fontsize=11)

plt.title("Semantic Distance of Core Concepts: China vs. Philippine Media", fontsize=15, pad=15)
plt.ylabel("Semantic Distance (1 - Cosine Similarity)", fontsize=12)
plt.xlabel("Core Concept", fontsize=12)
plt.xticks(rotation=30, fontsize=11)
plt.ylim(0, result_df['China_PH_Semantic_Distance'].max() * 1.2)  # 给顶部留点空间放数字
plt.tight_layout()
bar_path = os.path.join(output_dir, "最终版_图B_中菲核心词语义距离_bert.png")
plt.savefig(bar_path, dpi=300)
plt.show()

# 保存CSV表格
csv_path = os.path.join(output_dir, "最终版_中菲核心词语义距离数据_bert.csv")
result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

print("\n📊 中菲核心词语义距离（bert-base-uncased）：")
print(result_df)
print(f"\n🎉 全部完成！你的图A、图B和表格已经全部完美保存在 {output_dir} 文件夹下！")