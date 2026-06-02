"""
最终数据降噪脚本
=================
基于逻辑2（三级效验）的Tier体系 + 新增Taiwan语境过滤 + 扩展噪音惩罚

输入: 最终数据.xlsx (4,795条)
输出:
  - 最终数据_降噪后.xlsx (保留的Tier 1 + Tier 2)
  - 最终数据_被剔除.xlsx (被排除的数据)

使用: python3 noise_reduction.py
"""

import pandas as pd
import numpy as np
import re
import os

# ============================================================
# 文件路径
# ============================================================
BASE_DIR = '/Users/keying/Desktop/Claude Code/260602中菲媒体论文/全部data'
INPUT_FILE = os.path.join(BASE_DIR, '最终数据.xlsx')
OUTPUT_KEPT = os.path.join(BASE_DIR, '最终数据_降噪后.xlsx')
OUTPUT_DISCARDED = os.path.join(BASE_DIR, '最终数据_被剔除.xlsx')
OUTPUT_FULL = os.path.join(BASE_DIR, '最终数据_全量打标.xlsx')


def load_data(input_file):
    """读取数据"""
    print(f'正在读取: {input_file}')
    df = pd.read_excel(input_file)
    df['Title'] = df['Title'].fillna('').astype(str)
    df['Content'] = df['Content'].fillna('').astype(str)
    # 兼容Source列名
    src_col = [c for c in df.columns if 'Source' in c]
    if src_col:
        df['Source_Std'] = df[src_col[0]].fillna('').astype(str)
    else:
        df['Source_Std'] = ''
    print(f'读取完成: {len(df)} 条, {len(df.columns)} 列')
    return df


def is_taiwan_dominant(title, content):
    """
    判断文章是否以台湾问题为主（而非WPS/SCS）。
    注意：台海联动的战略叙事文章不会被误杀。
    """
    text = (title + ' ' + content).lower()

    # 强烈指向台湾本土政治的信号（与南海无关的话题）
    taiwan_local = [
        'taiwan independence', 'tsai ing-wen', 'lai ching-te',
        'taipei representative', 'taipei economic and cultural',
        'taiwan question', 'taiwan affairs office',
        "taiwan's presidential", 'taiwan election', 'taiwanese president',
        'taiwan separatist', 'reunification of china', 'peaceful reunification',
        'cross-strait relations', 'cross-strait ties',
        'arms sale to taiwan', 'pelosi visit taiwan', 'pelosi\'s taiwan',
        'pla eastern theatre', 'taiwan independence forces',
        'taiwan compatriots', 'taiwan society',
        'one country, two systems', 'one-china principle',
        'taiwan issue', 'taiwan region',
    ]

    # 联动信号：只有真正与WPS/SCS强相关的词才列入
    # （注意：japan/australia/asean/asia-pacific等太宽泛，Taiwan文章也会频繁出现）
    linkage_signals = [
        'south china sea', 'west philippine sea',
        'spratly', 'kalayaan', 'paracel', 'scarborough',
        'arbitral award', 'unclos', 'hague ruling',
        'freedom of navigation',
        'exclusive economic zone',
        'maritime militia',
        'reef', 'ayungin', 'second thomas',
    ]

    # 补充：出现在标题中的Taiwan信号权重更高
    title_lower = str(title).lower()
    title_taiwan = sum(title_lower.count(kw) for kw in [
        'taiwan', 'taipei', 'cross-strait', 'tsai', 'pelosi',
    ])

    text_lower = (str(content)).lower()
    taiwan_hits = sum(text_lower.count(kw) for kw in taiwan_local)
    linkage_hits = sum(text.count(kw) for kw in linkage_signals)

    # 标题明确以Taiwan为核心 → 除非有强SCS联动，否则排除
    if title_taiwan >= 2:
        if linkage_hits == 0:
            return True
        # 标题Taiwan信号极强但SCS联动很弱 → 排除
        if title_taiwan >= 3 and linkage_hits < 2:
            return True

    # Taiwan信号强但完全没有SCS/WPS信号 → 排除
    if taiwan_hits >= 3 and 'south china sea' not in text and 'west philippine sea' not in text:
        return True

    # Taiwan信号非常强，SCS信号很弱 → 排除
    if taiwan_hits >= 5 and linkage_hits < 2:
        return True

    return False


def apply_tier_filter(df):
    """
    应用三级分级 + Taiwan语境过滤 + 扩展惩罚
    基于逻辑2的词库体系，吸收了逻辑1的合理成分。
    """
    total = len(df)

    # ==========================================
    # 1. 黑名单（垃圾来源/内容）
    # ==========================================
    blacklist_sources = [r'News Bites']
    blacklist_content = [
        r'Jielong-3', r'rocket from the South China Sea',
        r'DASHBOARD:', r'\[3 News That Matters\]'
    ]
    pattern_black_src = '|'.join(blacklist_sources)
    pattern_black_content = '|'.join(blacklist_content)

    # ==========================================
    # 2. Tier 1: 核心高相关
    # ==========================================
    tier1_acronyms = [
        r'\bUNCLOS\b', r'\bEEZ\b', r'\bFONOPs?\b', r'\bADIZ\b', r'\bBRP\b',
        r'\bCCG\b', r'\bPCG\b', r'\bLRAD\b', r'\bSCS\b', r'\bWPS\b',
        r'\bEDCA\b', r'\bBCM\b', r'\bPLA[\-\s]?N\b',
        r'\bMDT\b', r'\bVFA\b', r'\bRORE\b', r'\bAMTI\b',
    ]
    pattern_t1_acr = '|'.join(tier1_acronyms)

    tier1_words = [
        r'South China Sea', r'West Philippine Sea', r'Biển Đông',
        r'Scarborough', r'Panatag', r'Bajo de Masinloc', r'Huangyan',
        r'Second Thomas', r'Ayungin', r'Ren[\'\-]?ai',
        r'Sabina', r'Escoda', r'Xianbin',
        r'Spratly', r'Kalayaan', r'Nansha',
        r'Paracel', r'Hoang Sa', r'Xisha',
        r'Whitsun', r'Julian Felipe', r'Niu[\'\-]?e',
        r'Thitu', r'Pag[\-\s]?asa', r'Zhongye',
        r'Reed Bank', r'Recto Bank', r'Liyue',
        r'Mischief', r'Panganiban', r'Meiji',
        r'Subi', r'Zamora', r'Zhubi',
        r'Fiery Cross', r'Kagitingan', r'Yongshu',
        r'Cuarteron', r'Huayang', r'Calderon',
        r'Gaven', r'Nanxun', r'Burgos',
        r'Hughes', r'Dongmen', r'Chigua',
        r'Johnson South', r'Mabini', r'Gac Ma',
        r'Vanguard Bank', r'Tu Chinh', r'Wan[\'\-]?an',
        r'Half Moon Shoal', r'Hasa[\-\s]?Hasa', r'Banyue',
        r'Iroquois', r'Rozul',
        r'Flat Island', r'Patag', r'Sandy Cay',
        r'Nanshan Island', r'Lawak',
        r'Loaita', r'Kota',
        r'West York', r'Likas',
        r'Northeast Cay', r'Parola',
        r'Southwest Cay', r'Pugad',
        r'Swallow Reef', r'Layang[\-\s]?Layang', r'Danwan',
        r'Commodore Reef', r'Rizal', r'Jialing',
        r'Woody Island', r'Yongxing', r'Phu Lam',
        r'Macclesfield', r'Zhongsha', r'Bankaw',
        r'Pratas', r'Dongsha',
        r'Natuna', r'Luconia', r'James Shoal', r'Zengmu',
        # 核心法理与行动
        r'Nine[\-\s]?dash', r'Ten[\-\s]?dash', r'9[\-\s]?dash', r'10[\-\s]?dash',
        r'Arbitral Tribunal', r'Arbitral Award', r'Hague ruling', r'2016 ruling',
        r'BRP Sierra Madre', r'Balikatan', r'Carta General',
        r'rotation and reprovisioning', r'maritime militia',
        r'China Coast Guard', r'Chinese Coast Guard', r'Philippine Coast Guard',
        r'military[\-\s]?grade laser',
    ]
    tier1_words_wb = [r'\b' + w + r'\b' for w in tier1_words]
    pattern_t1_words = '|'.join(tier1_words_wb)

    # ==========================================
    # 3. Tier 2: 宏观地缘相关
    # ==========================================
    tier2_words = [
        r'ASEAN', r'Indo[\-\s]?Pacific', r'Asia[\-\s]?Pacific',
        r'water cannon', r'laser illumination', r'radar lock', r'fire control radar',
        r'blocking maneuver', r'dangerous maneuver', r'collision', r'ramming',
        r'shadowing', r'radio challenge',
        r'gray zone', r'grey zone', r'swarming', r'encroachment',
        r'artificial island', r'island building', r'militarization', r'reclamation',
        r'PLA Navy', r'Chinese Navy', r'Philippine Navy', r'US Navy',
        r'tensions', r'standoff', r'flashpoint', r'provocative', r'provocation',
        r'Exclusive Economic Zone', r'Freedom of Navigation', r'rules[\-\s]?based order',
        r'territorial waters', r'sovereign rights', r'historical rights', r'maritime boundary',
        r'sovereignty', r'territorial claim', r'maritime claim', r'territorial integrity',
        r'diplomatic protest', r'note verbale',
        r'maritime dispute', r'territorial dispute', r'joint patrol', r'naval drill',
        r'oil and gas exploration', r'joint development',
        r'Coast Guard', r'coast guard vessel',
        r'fishing ban', r'fishing vessel', r'fishermen',
        r'Code of Conduct', r'Declaration on the Conduct',
    ]
    tier2_words_wb = [r'\b' + w + r'\b' for w in tier2_words]
    pattern_t2_words = '|'.join(tier2_words_wb)

    # ==========================================
    # 4. 扩展惩罚词（明显噪音）
    # ==========================================
    penalty_content = [
        # 气象灾害（台风经过南海但非政治议题）
        r'typhoon', r'hurricane', r'cyclone', r'PAGASA',
        # 海洋科研/资源（提到南海但非争端）
        r'deep[\-\s]?sea', r'marine species', r'coral reef research',
        r'oil platform', r'subsea tree', r'petroleum engineering',
        r'offshore oil[\-\s]?gas',
        # 太空/航天（南海偶尔作为发射坐标出现）
        r'rocket launch', r'satellite', r'manned space',
        # 常规渔业（非争端语境）
        r'fishing season', r'fishery production', r'aquaculture',
    ]
    pattern_penalty = '|'.join([r'\b' + w + r'\b' for w in penalty_content])

    # ==========================================
    # 执行判定
    # ==========================================
    print('正在执行分级打标...')

    is_blacklist = (
        df['Source_Std'].str.contains(pattern_black_src, flags=re.IGNORECASE, regex=True) |
        df['Content'].str.contains(pattern_black_content, flags=re.IGNORECASE, regex=True)
    )

    is_tier1 = (
        df['Title'].str.contains(pattern_t1_acr, regex=True) |
        df['Content'].str.contains(pattern_t1_acr, regex=True) |
        df['Title'].str.contains(pattern_t1_words, flags=re.IGNORECASE, regex=True) |
        df['Content'].str.contains(pattern_t1_words, flags=re.IGNORECASE, regex=True)
    )

    is_tier2 = (
        df['Title'].str.contains(pattern_t2_words, flags=re.IGNORECASE, regex=True) |
        df['Content'].str.contains(pattern_t2_words, flags=re.IGNORECASE, regex=True)
    )

    is_penalty = df['Content'].str.contains(pattern_penalty, flags=re.IGNORECASE, regex=True)

    # Taiwan语境检查（逐条）
    print('正在执行Taiwan语境检查...')
    is_taiwan = pd.Series([False] * len(df))
    for i, row in df.iterrows():
        if is_taiwan_dominant(row['Title'], row['Content']):
            is_taiwan[i] = True

    # ==========================================
    # 综合判定
    # ==========================================
    # 优先级：黑名单 > Penalty噪音 > Taiwan主导 > Tier 1 > Tier 2 > 其他
    conditions = [
        is_blacklist,
        is_penalty & ~is_tier1,  # 命中惩罚词且不是Tier 1核心
        is_taiwan,
    ]
    choices = [
        'Tier 4 (黑名单)',
        'Tier 4 (噪音: 气象/科研等)',
        'Tier 4 (台湾主导语境)',
    ]

    # 如果没有命中以上排除条件，再分配Tier
    temp_level = np.select(conditions, choices, default='待定')

    # 在"待定"中分配 Tier 1 / Tier 2 / Tier 3
    final_level = []
    for i in range(len(df)):
        if temp_level[i] != '待定':
            final_level.append(temp_level[i])
        elif is_tier1.iloc[i]:
            final_level.append('Tier 1 (核心高相关)')
        elif is_tier2.iloc[i]:
            final_level.append('Tier 2 (宏观地缘相关)')
        else:
            final_level.append('Tier 3 (低相关/待定)')

    df['Relevance_Level'] = final_level

    # ==========================================
    # 输出统计
    # ==========================================
    print('\n' + '=' * 50)
    print('📊 分级统计:')
    print(df['Relevance_Level'].value_counts().to_string())

    # 按来源统计各层级
    print('\n📊 按来源的Tier分布:')
    cross = pd.crosstab(df['Source_Std'], df['Relevance_Level'])
    print(cross.to_string())

    # 分别计算保留/剔除
    kept = df[df['Relevance_Level'].str.contains('Tier [12]')].copy()
    discarded = df[df['Relevance_Level'].str.contains('Tier [34]')].copy()

    print(f'\n📈 保留 (Tier 1+2): {len(kept)} 条 ({len(kept)/total*100:.1f}%)')
    print(f'📉 剔除 (Tier 3+4): {len(discarded)} 条 ({len(discarded)/total*100:.1f}%)')

    return df, kept, discarded


def main():
    df = load_data(INPUT_FILE)

    # 去重
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['Title', 'Content'], keep='first')
    after_dedup = len(df)
    if before_dedup > after_dedup:
        print(f'去重: {before_dedup} → {after_dedup} (删除 {before_dedup - after_dedup} 条)')

    # 分级打标
    df_full, df_kept, df_discarded = apply_tier_filter(df)

    # 导出
    # 1. 全量打标（供检查）
    df_full.to_excel(OUTPUT_FULL, index=False)
    print(f'\n✅ 全量打标已保存: {OUTPUT_FULL}')

    # 2. 保留数据（Tier 1 + 2），移除打标列以保持原始格式
    cols_to_drop = ['Relevance_Level', 'Source_Std']
    kept_clean = df_kept.drop(columns=[c for c in cols_to_drop if c in df_kept.columns])
    kept_clean.to_excel(OUTPUT_KEPT, index=False)
    print(f'✅ 降噪后保留: {OUTPUT_KEPT} ({len(kept_clean)} 条)')

    # 3. 被剔除数据
    discarded_clean = df_discarded.drop(columns=[c for c in cols_to_drop if c in df_discarded.columns])
    discarded_clean.to_excel(OUTPUT_DISCARDED, index=False)
    print(f'✅ 被剔除: {OUTPUT_DISCARDED} ({len(discarded_clean)} 条)')

    # 4. 按来源的保留率
    print('\n📊 各来源保留率:')
    for src in sorted(df['Source_Std'].unique()):
        total_src = len(df[df['Source_Std'] == src])
        kept_src = len(df_kept[df_kept['Source_Std'] == src])
        tier3 = len(df_full[(df_full['Source_Std'] == src) & (df_full['Relevance_Level'] == 'Tier 3 (低相关/待定)')])
        tier4 = len(df_full[(df_full['Source_Std'] == src) & df_full['Relevance_Level'].str.contains('Tier 4')])
        print(f'  {src:35s}: {total_src:5d} → 保留 {kept_src:5d} ({kept_src/total_src*100:5.1f}%)  |  Tier3:{tier3:4d}  Tier4:{tier4:4d}')


if __name__ == '__main__':
    main()
