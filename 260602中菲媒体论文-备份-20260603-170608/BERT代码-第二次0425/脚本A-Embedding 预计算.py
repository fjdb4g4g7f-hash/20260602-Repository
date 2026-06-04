# ==============================================================================
# 脚本 A：Embedding 预计算（只需跑一次，约25-40分钟）
# 作用：把5800条文本编码成向量并保存到磁盘
#       后续 B/C 脚本直接加载，无需重复编码
# ==============================================================================
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

file_path  = '/Users/keying/Desktop/中菲媒体对比论文撰写/BERT/大表T1T2.xlsx'
output_dir = '/Users/keying/Desktop/中菲媒体对比论文撰写/BERT'
os.makedirs(output_dir, exist_ok=True)

# ── 加载数据 ──────────────────────────────────────────────────────────────────
print("="*60)
print("脚本A：数据加载 + Embedding 预计算")
print("="*60)

df = pd.read_excel(file_path)
for col in ['Content', 'Country', 'Date']:
    if col not in df.columns:
        raise ValueError(f"数据缺少列：{col}")

df = df.dropna(subset=['Content', 'Country', 'Date']).reset_index(drop=True)
docs       = df['Content'].astype(str).tolist()
countries  = df['Country'].astype(str).tolist()
timestamps = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d').tolist()

print(f"✅ 有效样本：{len(docs)} 条")

# ── 保存清洗后的数据（供B/C使用）────────────────────────────────────────────
df_clean = df.copy()
df_clean['Content']   = docs
df_clean['Country']   = countries
df_clean['Date']      = timestamps
df_clean.to_pickle(os.path.join(output_dir, 'cache_df_clean.pkl'))
print("✅ 清洗后数据已保存")

# ── Embedding 编码（最耗时，约25-40分钟）────────────────────────────────────
print("\n开始 Embedding 编码，请耐心等待...")
embedding_model = SentenceTransformer("all-mpnet-base-v2")
embeddings = embedding_model.encode(
    docs,
    show_progress_bar=True,
    batch_size=32          # M系列芯片推荐32，显存不足时改16
)

# 保存向量（npy格式，加载极快）
np.save(os.path.join(output_dir, 'cache_embeddings.npy'), embeddings)
print(f"\n✅ Embedding 编码完成，维度：{embeddings.shape}")
print(f"✅ 已保存至：cache_embeddings.npy")
print("\n现在可以运行 脚本B（参数验证）")
