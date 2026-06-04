import pandas as pd
import numpy as np
from bertopic import BERTopic
import os
import glob
from datetime import datetime
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

# ==============================================================================
# 🔧 Mac 专属修复：强制单进程，解决报错
# ==============================================================================
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def main():
    BASE_DIR = '/Users/keying/Desktop/中菲媒体对比论文撰写/BERT'

    # ==========================================================================
    # 1. 加载本地数据
    # ==========================================================================
    print("=" * 60)
    print("加载数据中...")
    print("=" * 60)

    data_files = glob.glob(os.path.join(BASE_DIR, "2_全量数据_带主题标签_*.xlsx"))
    model_folders = glob.glob(os.path.join(BASE_DIR, "BERTopic_中菲媒体模型_*"))

    if not data_files or not model_folders:
        raise FileNotFoundError("未找到文件！")

    data_path = sorted(data_files, key=os.path.getmtime)[-1]
    model_path = sorted(model_folders, key=os.path.getmtime)[-1]

    topic_model = BERTopic.load(model_path, embedding_model=None)
    df_result = pd.read_excel(data_path)

    docs = df_result['Content'].astype(str).tolist()
    topics = df_result['Topic'].tolist()
    valid_topics = [t for t in set(topics) if t != -1]

    print(f"✅ 数据加载成功：{len(df_result)} 条新闻")
    print(f"✅ 有效主题数量：{len(valid_topics)} 个")

    # ==========================================================================
    # 2. 【核心】学界主流：C_V 主题一致性计算
    # ==========================================================================
    print("\n" + "=" * 60)
    print("【学术验证】计算 C_V 主题一致性...")
    print("=" * 60)

    # 提取主题关键词
    topics_words = []
    for topic in sorted(valid_topics)[:10]:
        words = [word for word, _ in topic_model.get_topic(topic)]
        if words:
            topics_words.append(words)

    # 预处理文本
    texts = [doc.split() for doc in docs]
    dictionary = Dictionary(texts)

    # ✅ 强制单进程计算 C_V 分数（Mac 不报错）
    cm = CoherenceModel(
        topics=topics_words,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v',
        processes=1  # 关键修复：强制单线程
    )
    coherence_score = cm.get_coherence()

    # 计算噪音比
    noise_ratio = (pd.Series(topics) == -1).mean() * 100

    # ==========================================================================
    # 3. 【要求】直接简单明了输出结果
    # ==========================================================================
    print("\n" + "=" * 60)
    print("🎉 模型验证结果（学界标准）")
    print("=" * 60)
    print(f"1. C_V 主题一致性分数：{coherence_score:.3f}")
    print(f"   评价：{'优秀 (≥0.7)' if coherence_score >= 0.7 else '良好 (≥0.5)' if coherence_score >= 0.5 else '一般'}")
    print(f"2. 噪音样本比例：{noise_ratio:.2f}%")
    print(f"   评价：{'优秀 (<10%)' if noise_ratio < 10 else '合格 (<20%)' if noise_ratio < 20 else '偏高'}")
    print("=" * 60)

    # 保存结果到 Excel
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_df = pd.DataFrame({
        "指标": ["C_V主题一致性", "噪音样本占比(%)"],
        "数值": [round(coherence_score, 3), round(noise_ratio, 2)],
        "评价": ["优秀" if coherence_score >= 0.7 else "良好" if coherence_score >= 0.5 else "一般",
                 "优秀" if noise_ratio < 10 else "合格" if noise_ratio < 20 else "偏高"]
    })
    save_path = os.path.join(BASE_DIR, f"6_模型验证结果_{timestamp_str}.xlsx")
    result_df.to_excel(save_path, index=False)
    print(f"\n✅ 结果已保存至：{save_path}")


# ==============================================================================
# 🔧 Mac 专属修复：必须加这个入口
# ==============================================================================
if __name__ == '__main__':
    main()