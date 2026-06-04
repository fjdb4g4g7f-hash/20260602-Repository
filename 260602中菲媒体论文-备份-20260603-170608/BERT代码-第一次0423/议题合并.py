import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import os
import glob
from datetime import datetime

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def main():
    BASE_DIR = '/Users/keying/Desktop/中菲媒体对比论文撰写/BERT'

    print("=" * 60)
    print("步骤 1/3：加载现有模型和数据")
    print("=" * 60)

    model_folders = glob.glob(os.path.join(BASE_DIR, "BERTopic_中菲媒体模型_*"))
    data_files = glob.glob(os.path.join(BASE_DIR, "2_全量数据_带主题标签_*.xlsx"))

    if not model_folders or not data_files:
        raise FileNotFoundError("未找到模型或数据，请确认之前运行成功！")

    model_path = sorted(model_folders, key=os.path.getmtime)[-1]
    data_path = sorted(data_files, key=os.path.getmtime)[-1]

    topic_model = BERTopic.load(model_path, embedding_model=None)
    df_result = pd.read_excel(data_path)
    docs = df_result['Content'].astype(str).tolist()
    countries = df_result['Country'].astype(str).tolist()
    topics = df_result['Topic'].tolist()

    print("\n" + "=" * 60)
    print("步骤 2/3：模型自动合并并【修复特征词】")
    print("=" * 60)

    # 1. 强行合并到 15 个（包含噪音就是 14个有效主题）
    topic_model.reduce_topics(docs, nr_topics=15)
    new_topics = topic_model.topics_

    # 2. 【核心修复】把丢失的停用词表重新建出来
    custom_stop_words = list(ENGLISH_STOP_WORDS) + [
        'china', 'philippine', 'philippines', 'sea', 'south', 'chinese',
        'said', 'maritime', 'water', 'beijing', 'manila', 'island',
        'south china', 'china sea'
    ]
    vectorizer_model = CountVectorizer(stop_words=custom_stop_words, ngram_range=(1, 2))

    # 3. 【核心修复】强令模型用新的词表刷新主题代表词
    print("正在剔除the/and等无意义词汇，重新提取核心战略词汇...")
    topic_model.update_topics(docs, vectorizer_model=vectorizer_model)

    print("\n前10个新主题预览：")
    print(topic_model.get_topic_info().head(10))

    print("\n" + "=" * 60)
    print("步骤 3/3：重新生成并保存完美结果")
    print("=" * 60)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    df_result['Topic'] = new_topics
    df_result['Topic_Name'] = df_result['Topic'].map(topic_model.get_topic_info().set_index('Topic')['Name'])
    save_data_path = os.path.join(BASE_DIR, f"最终完美版_数据_{timestamp_str}.xlsx")
    df_result.to_excel(save_data_path, index=False)

    country_topic_dist = df_result.groupby(['Country', 'Topic']).size().reset_index(name='Count')
    country_topic_dist['Percentage'] = country_topic_dist['Count'] / country_topic_dist.groupby('Country')[
        'Count'].transform('sum') * 100
    save_path_dist = os.path.join(BASE_DIR, f"最终完美版_分国家数量分布_{timestamp_str}.xlsx")
    country_topic_dist.to_excel(save_path_dist, index=False)

    print("正在提取中菲专属特征词...")
    topics_per_class = topic_model.topics_per_class(docs, classes=countries)
    save_path_class_words = os.path.join(BASE_DIR, f"最终完美版_中菲特征词汇对比_{timestamp_str}.xlsx")
    topics_per_class.to_excel(save_path_class_words, index=False)

    print("\n🎉 大功告成！这次打开特征词汇表，绝对是漂亮的战略词汇！")


if __name__ == '__main__':
    main()