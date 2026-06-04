"""
Step 2: BERT语义编码
- 使用 sentence-transformers 将每篇文章编码为向量
- 模型: all-mpnet-base-v2（英文语料最优平衡，语义捕捉能力强）
- 输出: 每篇文章对应一个768维向量

依赖安装（在PyCharm终端运行）:
    pip install sentence-transformers torch pandas openpyxl
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import time

# ─────────────────────────────────────────────
# 1. 读取预处理数据
# ─────────────────────────────────────────────
print("读取预处理数据...")
df_events = pd.read_pickle(
    "/Users/keying/Desktop/中菲媒体对比论文撰写/BERT/preprocessed_events.pkl"
)
df_full = pd.read_pickle(
    "/Users/keying/Desktop/中菲媒体对比论文撰写/BERT/preprocessed_full.pkl"
)

# ─────────────────────────────────────────────
# 2. 加载模型
# ─────────────────────────────────────────────
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
print(f"加载模型: {MODEL_NAME}")
print("首次运行会自动下载模型（约420MB），请保持网络连接...")

model = SentenceTransformer(MODEL_NAME)

# 检测是否有GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")
model = model.to(device)

# ─────────────────────────────────────────────
# 3. 编码函数
# ─────────────────────────────────────────────
def encode_texts(texts, model, batch_size=32, desc="编码中"):
    """
    批量编码文本，带进度显示
    对长文本截取前512个词（BERT上限），Title+Content拼接
    """
    # 截断处理：取前500个单词，避免超出模型上限
    truncated = []
    for t in texts:
        words = str(t).split()
        truncated.append(" ".join(words[:500]))

    print(f"  {desc}: 共{len(truncated)}篇，batch_size={batch_size}")
    start = time.time()

    embeddings = model.encode(
        truncated,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # 归一化，余弦距离计算更准确
    )

    elapsed = time.time() - start
    print(f"  完成，耗时 {elapsed:.1f}秒，向量维度: {embeddings.shape}")
    return embeddings


# ─────────────────────────────────────────────
# 4. 对事件窗口数据编码
# ─────────────────────────────────────────────
print("\n开始编码事件窗口数据...")

# 拼接 Title + Content 作为输入（比单独用Content信息更完整）
df_events = df_events.reset_index(drop=True)
texts_events = (
    df_events['Title'].fillna('') + ". " + df_events['Content'].fillna('')
).tolist()

embeddings_events = encode_texts(texts_events, model, desc="事件窗口文章")

# ─────────────────────────────────────────────
# 5. 对全量数据编码（用于语义空间对比分析）
# ─────────────────────────────────────────────
print("\n开始编码全量数据（用于语义空间对比）...")
df_full = df_full.reset_index(drop=True)
texts_full = (
    df_full['Title'].fillna('') + ". " + df_full['Content'].fillna('')
).tolist()

embeddings_full = encode_texts(texts_full, model, desc="全量文章")

# ─────────────────────────────────────────────
# 6. 保存向量
# ─────────────────────────────────────────────
BASE = "/Users/keying/Desktop/中菲媒体对比论文撰写/BERT/"

np.save(BASE + "embeddings_events.npy", embeddings_events)
np.save(BASE + "embeddings_full.npy",   embeddings_full)

print(f"\n✅ Step 2 完成")
print(f"   事件窗口向量: {embeddings_events.shape} → embeddings_events.npy")
print(f"   全量向量:     {embeddings_full.shape} → embeddings_full.npy")
print(f"\n请运行 step3_analysis.py")