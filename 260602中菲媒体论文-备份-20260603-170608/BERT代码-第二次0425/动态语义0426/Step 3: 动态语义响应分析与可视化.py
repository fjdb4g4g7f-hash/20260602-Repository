"""
Step 3: 动态语义响应分析与可视化
输出四张图：
  图A - 各事件节点前后中菲语义距离变化（核心结果）
  图B - 各事件节点叙事重心漂移（对抗词 vs 合作词方向）
  图C - 全量数据"安全"/"和平"词在中菲语义空间的近邻对比
  图D - 事件序列语义距离时间曲线（综合呈现）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. 设置中文字体（Mac本地环境）
# ─────────────────────────────────────────────
plt.rcParams['font.family'] = ['Arial Unicode MS', 'STHeiti', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

BASE = "/Users/keying/Desktop/中菲媒体对比论文撰写/BERT/"

# ─────────────────────────────────────────────
# 1. 读取数据与向量
# ─────────────────────────────────────────────
print("读取数据与向量...")
df_events     = pd.read_pickle(BASE + "preprocessed_events.pkl")
df_full       = pd.read_pickle(BASE + "preprocessed_full.pkl")
emb_events    = np.load(BASE + "embeddings_events.npy")
emb_full      = np.load(BASE + "embeddings_full.npy")

df_events = df_events.reset_index(drop=True)
df_full   = df_full.reset_index(drop=True)

# 将向量附加回DataFrame
df_events['embedding'] = list(emb_events)
df_full['embedding']   = list(emb_full)

# ─────────────────────────────────────────────
# 2. 事件元信息
# ─────────────────────────────────────────────
EVENTS = [
    {"id":"E1","name":"激光照射\n(2023.02)","name_short":"E1"},
    {"id":"E2","name":"水炮+马科斯\n声明(2023.08)","name_short":"E2"},
    {"id":"E3","name":"上海磋商\n(2024.01)","name_short":"E3"},
    {"id":"E4","name":"登船夺械\n(2024.06)","name_short":"E4"},
    {"id":"E5","name":"仁爱礁临时\n安排(2024.07)","name_short":"E5"},
]
EVENT_COLORS = {
    "E1": "#E07B54",  # 冲突事件 - 橙红
    "E2": "#C0392B",  # 冲突事件 - 深红
    "E3": "#2980B9",  # 缓和事件 - 蓝
    "E4": "#922B21",  # 最高烈度 - 暗红
    "E5": "#1A5276",  # 缓和事件 - 深蓝
}
CONFLICT_EVENTS = {"E1","E2","E4"}
DETENTE_EVENTS  = {"E3","E5"}

# ─────────────────────────────────────────────
# 3. 核心函数：计算组间余弦距离
# ─────────────────────────────────────────────
def group_cosine_distance(emb_a, emb_b):
    """
    计算两组向量的平均余弦距离
    距离 = 1 - 相似度，范围[0,2]，值越大叙事差异越大
    """
    if len(emb_a) == 0 or len(emb_b) == 0:
        return np.nan
    sim_matrix = cosine_similarity(emb_a, emb_b)
    return float(1 - sim_matrix.mean())


def get_group_embeddings(df, event_id, country, phase):
    subset = df[
        (df['event_id'] == event_id) &
        (df['Country']  == country)  &
        (df['phase']    == phase)
    ]
    if len(subset) == 0:
        return np.array([]).reshape(0, emb_events.shape[1])
    return np.stack(subset['embedding'].values)


# ─────────────────────────────────────────────
# 4. 图A：各事件节点前后中菲语义距离
# ─────────────────────────────────────────────
print("计算各事件节点语义距离...")

results = []
for event in EVENTS:
    eid = event['id']
    for phase in ['pre', 'post']:
        cn_emb = get_group_embeddings(df_events, eid, 'China',       phase)
        ph_emb = get_group_embeddings(df_events, eid, 'Philippine', phase)
        dist   = group_cosine_distance(cn_emb, ph_emb)
        cn_n   = len(cn_emb)
        ph_n   = len(ph_emb)
        results.append({
            'event_id':   eid,
            'event_name': event['name'],
            'phase':      phase,
            'distance':   dist,
            'cn_n':       cn_n,
            'ph_n':       ph_n
        })

res_df = pd.DataFrame(results)
print("\n语义距离计算结果:")
print(res_df[['event_id','phase','distance','cn_n','ph_n']].to_string(index=False))

# 绘图A
fig, ax = plt.subplots(figsize=(11, 5.5))

x       = np.arange(len(EVENTS))
width   = 0.32
pre_vals  = [res_df[(res_df.event_id==e['id'])&(res_df.phase=='pre') ]['distance'].values[0] for e in EVENTS]
post_vals = [res_df[(res_df.event_id==e['id'])&(res_df.phase=='post')]['distance'].values[0] for e in EVENTS]

bars_pre  = ax.bar(x - width/2, pre_vals,  width, label='事件前14天（基线期）',
                   color='#AED6F1', edgecolor='#2980B9', linewidth=0.8, zorder=3)
bars_post = ax.bar(x + width/2, post_vals, width, label='事件后14天（响应期）',
                   color='#F1948A', edgecolor='#C0392B', linewidth=0.8, zorder=3)

# 标注变化方向
for i, (pre, post) in enumerate(zip(pre_vals, post_vals)):
    if not (np.isnan(pre) or np.isnan(post)):
        delta = post - pre
        arrow = "↑" if delta > 0 else "↓"
        color = "#C0392B" if delta > 0 else "#1A5276"
        ax.text(x[i], max(pre, post) + 0.003, f"{arrow}{abs(delta):.3f}",
                ha='center', va='bottom', fontsize=8.5, color=color, fontweight='bold')

# 标注样本量
for i, event in enumerate(EVENTS):
    eid = event['id']
    pre_row  = res_df[(res_df.event_id==eid)&(res_df.phase=='pre')].iloc[0]
    post_row = res_df[(res_df.event_id==eid)&(res_df.phase=='post')].iloc[0]
    ax.text(x[i]-width/2, 0.001, f"CN:{pre_row.cn_n}\nPH:{pre_row.ph_n}",
            ha='center', va='bottom', fontsize=6.5, color='#555')
    ax.text(x[i]+width/2, 0.001, f"CN:{post_row.cn_n}\nPH:{post_row.ph_n}",
            ha='center', va='bottom', fontsize=6.5, color='#555')

# 标记事件类型
for i, event in enumerate(EVENTS):
    eid  = event['id']
    tag  = "▲冲突事件" if eid in CONFLICT_EVENTS else "●缓和事件"
    col  = "#922B21" if eid in CONFLICT_EVENTS else "#1A5276"
    ax.text(x[i], -0.008, tag, ha='center', va='top', fontsize=7.5,
            color=col, style='italic')

ax.set_xticks(x)
ax.set_xticklabels([e['name'] for e in EVENTS], fontsize=9)
ax.set_ylabel("中菲叙事语义距离（余弦距离）", fontsize=10)
ax.set_title("图A：关键事件节点前后中菲媒体叙事语义距离变化\n"
             "（距离越大 = 叙事错位越严重）", fontsize=11, pad=12)
ax.legend(fontsize=9, loc='upper left')
ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(BASE + "figA_semantic_distance.png", dpi=200, bbox_inches='tight')
print("✅ 图A已保存")
# plt.show()

# ─────────────────────────────────────────────
# 5. 图B：叙事重心漂移分析
# ─────────────────────────────────────────────
print("\n计算叙事重心漂移...")

# 加载模型（复用Step2的模型，用于编码锚词）
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# 定义对抗语义锚点和合作语义锚点（各用多个词取平均，更稳健）
CONFRONTATION_ANCHORS = [
    "military confrontation aggression threat provocation illegal violation",
    "coast guard vessel water cannon harassment dangerous",
    "sovereignty violation armed attack escalation",
]
COOPERATION_ANCHORS = [
    "peaceful dialogue cooperation stability regional development",
    "ASEAN partnership bilateral consultation agreement",
    "negotiation diplomatic channel de-escalation",
]

conf_vecs = model.encode(CONFRONTATION_ANCHORS, normalize_embeddings=True)
coop_vecs = model.encode(COOPERATION_ANCHORS,   normalize_embeddings=True)
conf_anchor = conf_vecs.mean(axis=0, keepdims=True)  # 1×768
coop_anchor = coop_vecs.mean(axis=0, keepdims=True)  # 1×768

def narrative_score(embeddings):
    """
    计算叙事方向得分：
    正值 = 偏向对抗语义
    负值 = 偏向合作语义
    """
    if len(embeddings) == 0:
        return np.nan
    sim_conf = cosine_similarity(embeddings, conf_anchor).mean()
    sim_coop = cosine_similarity(embeddings, coop_anchor).mean()
    return float(sim_conf - sim_coop)

# 计算各事件各阶段各国的叙事得分
score_results = []
for event in EVENTS:
    eid = event['id']
    for country in ['China','Philippine']:
        for phase in ['pre','post']:
            emb = get_group_embeddings(df_events, eid, country, phase)
            score = narrative_score(emb)
            score_results.append({
                'event_id': eid,
                'event_name': event['name'],
                'country': country,
                'phase': phase,
                'score': score
            })

score_df = pd.DataFrame(score_results)

# 绘图B：分面展示，每个事件一组
fig, axes = plt.subplots(1, 5, figsize=(14, 5), sharey=True)
fig.suptitle("图B：关键事件节点前后叙事重心漂移\n"
             "（正值偏对抗，负值偏合作；箭头方向=事件后漂移趋势）",
             fontsize=11, y=1.01)

country_colors = {'China': '#E74C3C', 'Philippine': '#2980B9'}
country_labels = {'China': '中国媒体', 'Philippine': '菲律宾媒体'}

for ax, event in zip(axes, EVENTS):
    eid = event['id']
    for country in ['China','Philippine']:
        pre_score  = score_df[(score_df.event_id==eid)&(score_df.country==country)&(score_df.phase=='pre') ]['score'].values[0]
        post_score = score_df[(score_df.event_id==eid)&(score_df.country==country)&(score_df.phase=='post')]['score'].values[0]
        col = country_colors[country]
        offset = -0.15 if country == 'China' else 0.15

        if not (np.isnan(pre_score) or np.isnan(post_score)):
            ax.annotate("",
                xy=(post_score, offset),
                xytext=(pre_score, offset),
                arrowprops=dict(arrowstyle="->", color=col, lw=2.0)
            )
            ax.plot([pre_score], [offset], 'o', color=col, ms=7, zorder=5)
            ax.plot([post_score],[offset], 's', color=col, ms=7, zorder=5)
            ax.text(post_score, offset + (0.07 if country=='Philippines' else -0.07),
                    f"{post_score:.3f}", ha='center', fontsize=7.5, color=col)

    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_title(event['name'], fontsize=8.5, pad=6)
    ax.set_xlim(-0.05, 0.12)
    ax.set_yticks([-0.15, 0.15])
    ax.set_yticklabels(['中国', '菲律宾'], fontsize=8.5)
    ax.spines[['top','right','left']].set_visible(False)

    bg_col = "#FADBD8" if eid in CONFLICT_EVENTS else "#D6EAF8"
    ax.set_facecolor(bg_col)

# 图例
patches = [
    mpatches.Patch(color=country_colors['China'],       label='中国媒体'),
    mpatches.Patch(color=country_colors['Philippine'], label='菲律宾媒体'),
    mpatches.Patch(color="#FADBD8", label='冲突事件'),
    mpatches.Patch(color="#D6EAF8", label='缓和事件'),
]
fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.08), frameon=False)
axes[0].set_xlabel("← 合作语义    对抗语义 →", fontsize=8)
fig.tight_layout()
plt.savefig(BASE + "figB_narrative_drift.png", dpi=200, bbox_inches='tight')
print("✅ 图B已保存")
# plt.show()

# ─────────────────────────────────────────────
# 6. 图C：全量数据"安全/和平"在中菲语义空间的近邻对比
# ─────────────────────────────────────────────
print("\n计算全量语义空间近邻...")

# 提取中菲全量向量
cn_full_emb = np.stack(df_full[df_full['Country']=='China']['embedding'].values)
ph_full_emb = np.stack(df_full[df_full['Country']=='Philippine']['embedding'].values)

# 查询词向量
QUERY_WORDS = {
    "security": "security threat danger risk protection",
    "peace":    "peace stability harmony cooperation",
}

TOP_K = 10  # 取前10个近邻词所在文章的片段

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("图C：核心词\"security\"与\"peace\"在中菲语义空间的近邻分布\n"
             "（证明同一词汇在两国媒体中承载截然不同的语义内容）",
             fontsize=11)

for ax, (qword, qtext) in zip(axes, QUERY_WORDS.items()):
    q_vec = model.encode([qtext], normalize_embeddings=True)

    # 找中菲各自最相近的TOP_K篇文章
    cn_sims = cosine_similarity(q_vec, cn_full_emb)[0]
    ph_sims = cosine_similarity(q_vec, ph_full_emb)[0]

    cn_top_idx = np.argsort(cn_sims)[-TOP_K:][::-1]
    ph_top_idx = np.argsort(ph_sims)[-TOP_K:][::-1]

    cn_top_df = df_full[df_full['Country']=='China'].reset_index(drop=True).iloc[cn_top_idx]
    ph_top_df = df_full[df_full['Country']=='Philippine'].reset_index(drop=True).iloc[ph_top_idx]

    # 从标题中提取关键词频率（简单词频统计）
    from collections import Counter
    import re

    def extract_words(texts, stopwords=None):
        stopwords = stopwords or set()
        words = []
        for t in texts:
            tokens = re.findall(r'\b[a-zA-Z]{4,}\b', str(t).lower())
            words.extend([w for w in tokens if w not in stopwords])
        return Counter(words)

    STOPWORDS = {
        'china','chinese','philippine','philippines','south','north','east','west',
        'said','says','also','that','this','with','from','have','will','been',
        'their','they','were','which','about','more','than','over','into','after',
        'when','what','news','year','time','like','just','even','some','such',
        'would','could','should','there','these','those','other','many','most',
        'through','between','during','under','against','before','within','while'
    }

    cn_words  = extract_words(cn_top_df['Title'].tolist() + cn_top_df['Content'].str[:200].tolist(), STOPWORDS)
    ph_words  = extract_words(ph_top_df['Title'].tolist() + ph_top_df['Content'].str[:200].tolist(), STOPWORDS)

    # 取各自前8个高频词
    cn_common = cn_words.most_common(8)
    ph_common = ph_words.most_common(8)

    # 水平条形图
    all_words = list(set([w for w,_ in cn_common] + [w for w,_ in ph_common]))
    cn_dict = dict(cn_common)
    ph_dict = dict(ph_common)

    y = np.arange(len(all_words))
    cn_vals = [cn_dict.get(w, 0) for w in all_words]
    ph_vals = [ph_dict.get(w, 0) for w in all_words]

    # 归一化到0-1
    max_val = max(max(cn_vals), max(ph_vals), 1)
    cn_norm = [v / max_val for v in cn_vals]
    ph_norm = [v / max_val for v in ph_vals]  # 就改这一行！

    ax.barh(y - 0.2, cn_norm, 0.35, label='中国媒体语境', color='#E74C3C', alpha=0.8)
    ax.barh(y + 0.2, ph_norm, 0.35, label='菲律宾媒体语境', color='#2980B9', alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(all_words, fontsize=9)
    ax.set_xlabel("词频（归一化）", fontsize=9)
    ax.set_title(f'"{qword}"的语义近邻词汇分布', fontsize=10, pad=8)
    ax.legend(fontsize=8.5)
    ax.spines[['top','right']].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(BASE + "figC_semantic_context.png", dpi=200, bbox_inches='tight')
print("✅ 图C已保存")
# plt.show()

# ─────────────────────────────────────────────
# 7. 图D：事件序列综合时间曲线
# ─────────────────────────────────────────────
print("\n生成综合时间曲线...")

# 按月计算全样本中菲语义距离
df_full['YearMonth'] = df_full['Date'].dt.to_period('M')
months = sorted(df_full['YearMonth'].unique())

monthly_dist = []
for ym in months:
    cn_emb_m = np.stack(
        df_full[(df_full['YearMonth']==ym)&(df_full['Country']=='China')]['embedding'].values
    ) if len(df_full[(df_full['YearMonth']==ym)&(df_full['Country']=='China')]) > 0 else np.array([]).reshape(0,768)
    ph_emb_m = np.stack(
        df_full[(df_full['YearMonth']==ym)&(df_full['Country']=='Philippine')]['embedding'].values
    ) if len(df_full[(df_full['YearMonth']==ym)&(df_full['Country']=='Philippine')]) > 0 else np.array([]).reshape(0,768)
    dist = group_cosine_distance(cn_emb_m, ph_emb_m)
    monthly_dist.append({'month': ym, 'distance': dist})

monthly_df = pd.DataFrame(monthly_dist).dropna()

fig, ax = plt.subplots(figsize=(13, 5))

ax.plot(
    [str(m) for m in monthly_df['month']],
    monthly_df['distance'],
    color='#555', linewidth=1.5, zorder=3, label='月度中菲语义距离'
)
ax.fill_between(
    range(len(monthly_df)),
    monthly_df['distance'],
    alpha=0.12, color='#555'
)

# 标记五个关键事件
event_dates_pd = {
    "E1": pd.Period("2023-02", "M"),
    "E2": pd.Period("2023-08", "M"),
    "E3": pd.Period("2024-01", "M"),
    "E4": pd.Period("2024-06", "M"),
    "E5": pd.Period("2024-07", "M"),
}
event_labels = {
    "E1":"激光照射\n(2023.02)",
    "E2":"水炮+声明\n(2023.08)",
    "E3":"上海磋商\n(2024.01)",
    "E4":"登船夺械\n(2024.06)",
    "E5":"临时安排\n(2024.07)",
}
month_list = list(monthly_df['month'])
for eid, ep in event_dates_pd.items():
    if ep in month_list:
        idx  = month_list.index(ep)
        dist = monthly_df.iloc[idx]['distance']
        col  = "#C0392B" if eid in CONFLICT_EVENTS else "#1A5276"
        marker = "v" if eid in CONFLICT_EVENTS else "^"
        ax.plot(idx, dist, marker=marker, ms=10, color=col, zorder=5)
        offset = 0.004 if eid in CONFLICT_EVENTS else -0.008
        ax.text(idx, dist + offset, event_labels[eid],
                ha='center', va='bottom' if offset>0 else 'top',
                fontsize=7.5, color=col,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor=col))

ax.set_xticks(range(0, len(monthly_df), 3))
ax.set_xticklabels(
    [str(monthly_df['month'].iloc[i]) for i in range(0, len(monthly_df), 3)],
    rotation=30, ha='right', fontsize=8
)
ax.set_ylabel("中菲叙事语义距离（余弦距离）", fontsize=10)
ax.set_title("图D：小马科斯执政期间中菲媒体叙事语义距离月度时间序列\n"
             "（▼冲突事件  ▲缓和事件）", fontsize=11, pad=10)

# 添加冲突/缓和图例
conflict_patch = mpatches.Patch(color='#C0392B', label='冲突事件节点')
detente_patch  = mpatches.Patch(color='#1A5276', label='缓和事件节点')
ax.legend(handles=[conflict_patch, detente_patch], fontsize=9, loc='upper left')

ax.yaxis.grid(True, linestyle='--', alpha=0.4)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(BASE + "figD_monthly_timeline.png", dpi=200, bbox_inches='tight')
print("✅ 图D已保存")
# plt.show()

# ─────────────────────────────────────────────
# 8. 输出核心结论摘要（供写作参考）
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("核心数值结论摘要（供论文写作参考）")
print("="*60)
for event in EVENTS:
    eid = event['id']
    pre  = res_df[(res_df.event_id==eid)&(res_df.phase=='pre') ]['distance'].values[0]
    post = res_df[(res_df.event_id==eid)&(res_df.phase=='post')]['distance'].values[0]
    delta = post - pre
    direction = "↑升高（叙事错位加剧）" if delta > 0 else "↓下降（叙事错位收窄）"
    print(f"{eid} {event['name'].replace(chr(10),' ')}")
    print(f"   事件前: {pre:.4f}  事件后: {post:.4f}  变化: {delta:+.4f}  {direction}")

print("\n✅ Step 3 全部完成，四张图已保存至:", BASE)