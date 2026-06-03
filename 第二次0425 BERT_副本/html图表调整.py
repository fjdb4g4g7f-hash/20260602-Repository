import pandas as pd
from bertopic import BERTopic
import os
import glob
import re
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==============================================================================
# 🔥 核心修复：必须在加载模型前，把这个预处理函数原封不动地放进来，否则模型会认不出来
# ==============================================================================
def custom_preprocessor(text):
    text = str(text).lower()
    text = re.sub(r'\bsouth china sea\b', '', text)
    text = re.sub(r'\bsouth china\b', '', text)
    text = re.sub(r'\bchina sea\b', '', text)
    return text


def main():
    BASE_DIR = '/Users/keying/Desktop/中菲媒体对比论文撰写/BERT'

    print("=" * 60)
    print("极简出图脚本：加载现成模型与数据（跳过训练）")
    print("=" * 60)

    # 1. 自动寻找最新的模型和全量数据文件
    model_folders = glob.glob(os.path.join(BASE_DIR, "BERTopic_中菲媒体模型_*_备份.pkl"))
    if not model_folders:
        model_folders = glob.glob(os.path.join(BASE_DIR, "BERTopic_中菲媒体模型_*"))

    data_files = glob.glob(os.path.join(BASE_DIR, "2_全量数据_带主题标签_*.xlsx"))

    if not model_folders or not data_files:
        raise FileNotFoundError("未找到模型或数据，请确认路径！")

    model_path = sorted(model_folders, key=os.path.getmtime)[-1]
    data_path = sorted(data_files, key=os.path.getmtime)[-1]

    print(f"正在加载模型：{os.path.basename(model_path)}")
    topic_model = BERTopic.load(model_path, embedding_model=None)

    print(f"正在加载数据：{os.path.basename(data_path)}")
    df = pd.read_excel(data_path)
    docs = df['Content'].astype(str).tolist()
    countries = df['Country'].astype(str).tolist()
    timestamps = df['Date'].astype(str).tolist()

    # ==============================================================================
    # 2. 🌟 精准控制要在图表中显示的主题列表
    # ==============================================================================
    all_topics = set(df['Topic'].tolist())

    # 你要求剔除的三个主题 ID：-1 (噪音), 11 (quick stories), 12 (barbie)
    exclude_topics = [-1, 11, 12]

    # 生成最终要在图上画出来的主题列表（去掉排斥的，剩下的全画）
    topics_to_keep = sorted([t for t in all_topics if t not in exclude_topics])

    print(f"\n✅ 准备绘制图表，将彻底隐藏 -1, 11, 12。")
    print(f"最终在图表上呈现的 {len(topics_to_keep)} 个主题是：\n{topics_to_keep}")

    # ==============================================================================
    # 3. 重新计算可视化基础数据
    # ==============================================================================
    print("\n" + "=" * 60)
    print("计算画图所需数据（约需十秒）...")
    print("=" * 60)

    topics_per_class = topic_model.topics_per_class(docs, classes=countries)
    topics_over_time = topic_model.topics_over_time(docs, timestamps=timestamps, global_tuning=True,
                                                    evolution_tuning=True)

    # ==============================================================================
    # 4. 生成并导出定制版图表
    # ==============================================================================
    print("\n" + "=" * 60)
    print("正在生成并导出完美图表...")
    print("=" * 60)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig_classes = topic_model.visualize_topics_per_class(
        topics_per_class,
        topics=topics_to_keep,
        width=1100, height=700
    )
    path_classes = os.path.join(BASE_DIR, f"定制图_8_中菲词汇对比_{ts}.html")
    fig_classes.write_html(path_classes)
    print(f"  ✅ {os.path.basename(path_classes)}")

    fig_time = topic_model.visualize_topics_over_time(
        topics_over_time,
        topics=topics_to_keep,
        width=1200, height=700
    )
    path_time = os.path.join(BASE_DIR, f"定制图_9_主题时间演变_{ts}.html")
    fig_time.write_html(path_time)
    print(f"  ✅ {os.path.basename(path_time)}")

    fig_barchart = topic_model.visualize_barchart(
        topics=topics_to_keep,
        n_words=8,
        width=1200, height=800
    )
    path_bar = os.path.join(BASE_DIR, f"定制图_额外_核心词条形图_{ts}.html")
    fig_barchart.write_html(path_bar)
    print(f"  ✅ {os.path.basename(path_bar)}")

    print("\n🎉 大功告成！图表已全部按你的要求净化更新，快去打开看看吧！")


if __name__ == '__main__':
    main()