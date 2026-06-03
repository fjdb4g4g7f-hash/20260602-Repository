import pandas as pd
import numpy as np
import re


def advanced_semantic_proxy_filter(input_file, output_file):
    print(f'正在读取数据: {input_file} ...')
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f'读取报错: {e}')
        return

    df['Title'] = df['Title'].fillna('').astype(str)
    df['Content'] = df['Content'].fillna('').astype(str)

    print('正在配置正向与反向词库...')

    # ==========================================
    # 1. Tier 1: 核心冲突与具体实体 (加分项)
    # ==========================================
    tier1_acronyms = [
        r'\bUNCLOS\b', r'\bPCA\b', r'\bCOC\b', r'\bDOC\b',
        r'\bEEZ\b', r'\bFONOPs?\b', r'\bADIZ\b', r'\bBRP\b', r'\bCCG\b',
        r'\bPCG\b', r'\bAFP\b', r'\bLRAD\b', r'\bSCS\b', r'\bWPS\b',
        r'\bEDCA\b', r'\bASEAN\b', r'\bBCM\b', r'\bPLA[\-\s]?N\b'
    ]
    pattern_t1_acr = '|'.join(tier1_acronyms)

    tier1_words = [
        r'Scarborough', r'Panatag', r'Bajo de Masinloc', r'Huangyan',
        r'Second Thomas', r'Ayungin', r'Ren[\'\-]?ai', r'Sabina', r'Escoda', r'Xianbin',
        r'Spratly', r'Kalayaan', r'Nansha', r'Paracel', r'Hoang Sa', r'Xisha',
        r'Whitsun', r'Julian Felipe', r'Niu[\'\-]?e', r'Thitu', r'Pag[\-\s]?asa', r'Zhongye',
        r'Reed Bank', r'Recto Bank', r'Liyue', r'Mischief', r'Panganiban', r'Meiji',
        r'Subi', r'Zamora', r'Zhubi', r'Fiery Cross', r'Kagitingan', r'Yongshu',
        r'Vanguard Bank', r'Tu Chinh', r'Wan[\'\-]?an', r'Half Moon Shoal', r'Hasa[\-\s]?Hasa', r'Banyue',
        r'Iroquois', r'Rozul', r'Flat Island', r'Patag', r'Sandy Cay', r'Nanshan Island', r'Lawak',
        r'Loaita', r'Kota', r'West York', r'Likas', r'Northeast Cay', r'Parola',
        r'Southwest Cay', r'Pugad', r'Swallow Reef', r'Layang[\-\s]?Layang', r'Danwan',
        r'Commodore Reef', r'Rizal', r'Jialing', r'Woody Island', r'Macclesfield', r'Pratas', r'Dongsha',
        r'BRP Sierra Madre', r'water cannon', r'laser illumination', r'military[\-\s]?grade laser',
        r'radar lock', r'fire control radar', r'cyanide', r'Balikatan', r'Carta General',
        r'blocking maneuver', r'dangerous maneuver', r'collision', r'ramming',
        r'resupply mission', r'rotation and reprovisioning', r'airdrop',
        r'Coast Guard', r'maritime militia', r'PLA Navy', r'Chinese Navy', r'Chinese vessels', r'Chinese ships',
        r'Philippine Navy', r'Philippine vessels',
        r'gray zone', r'grey zone', r'swarming', r'coercion', r'encroachment', r'intimidation',
        r'joint patrol', r'artificial island', r'island building', r'militarization', r'reclamation',
        r'acoustic weapon', r'marine environment destruction', r'coral reef damage',
        r'Nine[\-\s]?dash', r'Ten[\-\s]?dash', r'9[\-\s]?dash', r'10[\-\s]?dash',
        r'Arbitral Tribunal', r'Arbitral Award', r'Hague ruling',
        r'Code of Conduct', r'Declaration on the Conduct'
    ]
    pattern_t1_words = '|'.join([r'\b' + word + r'\b' for word in tier1_words])

    # ==========================================
    # 2. Tier 2: 宏观地缘与外交词汇 (加分项)
    # ==========================================
    tier2_words = [
        r'tensions', r'standoff', r'flashpoint', r'provocative', r'provocation',
        r'harassment', r'Indo[\-\s]?Pacific', r'geopolitics', r'geopolitical',
        r'Exclusive Economic Zone', r'Freedom of Navigation',
        r'territorial waters', r'sovereign rights', r'historical rights', r'maritime boundary',
        r'Air Defense Identification Zone',
        r'sovereignty', r'territorial claim', r'maritime claim', r'diplomatic protest', r'note verbale',
        r'maritime dispute', r'territorial dispute', r'naval drill', r'military exercise', r'warship',
        r'maritime security', r'bilateral',
        r'oil and gas exploration', r'fishing ban', r'fishing vessel', r'fishermen'
    ]
    pattern_t2_words = '|'.join([r'\b' + word + r'\b' for word in tier2_words])

    # ==========================================
    # 3. 反向逻辑：干扰与噪音词汇 (扣分项)
    # ==========================================
    penalty_words = [
        # 天气气象类
        r'typhoon', r'hurricane', r'cyclone', r'storm', r'weather forecast', r'low pressure area',
        r'PAGASA', r'rainfall',
        # 经济股市类
        r'stock market', r'shares close', r'PSEi', r'trading day', r'index fell', r'inflation rate',
        r'dividend', r'earnings report', r'net income', r'shareholders',
        # 旅游生活体育类
        r'cruise ship', r'tourism', r'tourist', r'resort', r'basketball', r'PBA', r'championship'
    ]
    pattern_penalty = '|'.join([r'\b' + word + r'\b' for word in penalty_words])

    print('正在进行正反向交叉打分...')

    # 正向加分计算
    title_t1_acr_hits = df['Title'].str.count(pattern_t1_acr)
    title_t1_word_hits = df['Title'].str.count(pattern_t1_words, flags=re.IGNORECASE)
    title_t2_word_hits = df['Title'].str.count(pattern_t2_words, flags=re.IGNORECASE)
    df['Title_Score'] = (title_t1_acr_hits * 10) + (title_t1_word_hits * 10) + (title_t2_word_hits * 5)

    content_t1_acr_hits = df['Content'].str.count(pattern_t1_acr)
    content_t1_word_hits = df['Content'].str.count(pattern_t1_words, flags=re.IGNORECASE)
    content_t2_word_hits = df['Content'].str.count(pattern_t2_words, flags=re.IGNORECASE)
    df['Content_Score'] = (content_t1_acr_hits * 2) + (content_t1_word_hits * 2) + (content_t2_word_hits * 1)

    # 反向扣分计算 (标题扣10分，正文扣3分)
    title_penalty_hits = df['Title'].str.count(pattern_penalty, flags=re.IGNORECASE)
    content_penalty_hits = df['Content'].str.count(pattern_penalty, flags=re.IGNORECASE)
    df['Penalty_Score'] = (title_penalty_hits * 10) + (content_penalty_hits * 3)

    # 计算最终总分
    df['Total_Score'] = df['Title_Score'] + df['Content_Score'] - df['Penalty_Score']
    df['Content_Word_Count'] = df['Content'].apply(lambda x: len(str(x).split()))

    # 防止分母为0，同时如果分数为负，密度直接记为0
    df['Score_Density'] = np.where(
        df['Total_Score'] > 0,
        (df['Total_Score'] / df['Content_Word_Count'].replace(0, 1)) * 100,
        0
    )

    print('正在执行评级...')

    def assign_level(row):
        # 增加对负分的处理
        if row['Total_Score'] < 0:
            return 'Tier 4 (极大可能是干扰噪音)'
        elif row['Total_Score'] >= 15 or row['Score_Density'] >= 3.0:
            return 'Tier 1 (核心高相关)'
        elif row['Total_Score'] >= 5:
            return 'Tier 2 (宏观地缘相关)'
        elif row['Total_Score'] > 0:
            return 'Tier 3 (低相关/待定)'
        else:
            return 'Tier 4 (极弱相关/泛泛而谈)'

    df['Relevance_Level'] = df.apply(assign_level, axis=1)

    # 移除原本的直接删除逻辑，全部保留做去重
    if 'Title' in df.columns and 'Content' in df.columns:
        df_final = df.drop_duplicates(subset=['Title', 'Content'], keep='first')
    else:
        df_final = df.drop_duplicates(keep='first')

    # 整理列顺序，把扣分列也展示出来
    cols = df_final.columns.tolist()
    score_cols = ['Relevance_Level', 'Total_Score', 'Penalty_Score', 'Score_Density', 'Title_Score', 'Content_Score']
    new_cols = score_cols + [c for c in cols if c not in score_cols]
    df_final = df_final[new_cols]

    df_final = df_final.sort_values(by=['Total_Score', 'Score_Density'], ascending=[False, False])

    df_final.to_excel(output_file, index=False)

    print('-' * 40)
    print(f'已处理完成！去重后保留总量: {len(df_final)}')
    print('\n各评级数量分布：')
    print(df_final['Relevance_Level'].value_counts())
    print(f'\n文件已保存至: {output_file}')


if __name__ == '__main__':
    input_filename = '/Users/keying/Desktop/中菲媒体对比论文撰写/BERT/大表.xlsx'
    output_filename = '/Users/keying/Desktop/中菲媒体对比论文撰写/BERT/大表Clean.xlsx'

    advanced_semantic_proxy_filter(input_filename, output_filename)