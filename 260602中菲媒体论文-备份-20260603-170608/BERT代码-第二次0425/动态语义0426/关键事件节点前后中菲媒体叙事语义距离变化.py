# ==============================
# 独立脚本：修复版 图5 叙事语义距离柱状图
# 解决底部文字重叠问题，不依赖原代码，可直接运行
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ----------------------
# 0. 基础配置（字体+路径，和原代码保持一致）
# ----------------------
plt.rcParams['font.family'] = ['Arial Unicode MS', 'STHeiti', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 【修改为你的本地路径，和原代码保持一致】
BASE = "/Users/keying/Desktop/中菲媒体对比论文撰写/BERT/"

# ----------------------
# 1. 读取数据（和原代码完全匹配）
# ----------------------
df_events = pd.read_pickle(BASE + "preprocessed_events.pkl")
emb_events = np.load(BASE + "embeddings_events.npy")
df_events = df_events.reset_index(drop=True)
df_events['embedding'] = list(emb_events)

# ----------------------
# 2. 事件定义（沿用你的设置，未做修改）
# ----------------------
EVENTS = [
    {"id":"E1","name":"激光照射\n(2023.02)","name_short":"E1"},
    {"id":"E2","name":"水炮+马科斯\n声明(2023.08)","name_short":"E2"},
    {"id":"E3","name":"上海磋商\n(2024.01)","name_short":"E3"},
    {"id":"E4","name":"登船夺械\n(2024.06)","name_short":"E4"},
    {"id":"E5","name":"仁爱礁临时\n安排(2024.07)","name_short":"E5"},
]
CONFLICT_EVENTS = {"E1","E2","E4"}
DETENTE_EVENTS  = {"E3","E5"}

# ----------------------
# 3. 核心函数（和原代码一致，用于计算语义距离）
# ----------------------
def group_cosine_distance(emb_a, emb_b):
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

# ----------------------
# 4. 计算各事件前后语义距离（和原代码逻辑一致）
# ----------------------
results = []
for event in EVENTS:
    eid = event['id']
    for phase in ['pre', 'post']:
        cn_emb = get_group_embeddings(df_events, eid, 'China', phase)
        ph_emb = get_group_embeddings(df_events, eid, 'Philippine', phase)
        dist = group_cosine_distance(cn_emb, ph_emb)
        cn_n = len(cn_emb)
        ph_n = len(ph_emb)
        results.append({
            'event_id':   eid,
            'event_name': event['name'],
            'phase':      phase,
            'distance':   dist,
            'cn_n':       cn_n,
            'ph_n':       ph_n
        })
res_df = pd.DataFrame(results)

# ----------------------
# 5. 绘图：修复文字重合问题，优化排版
# ----------------------
# 1. 增加图表宽度，给x轴留出更多空间
fig, ax = plt.subplots(figsize=(12, 6.5))

x = np.arange(len(EVENTS))
width = 0.32

pre_vals  = [res_df[(res_df.event_id==e['id'])&(res_df.phase=='pre') ]['distance'].values[0] for e in EVENTS]
post_vals = [res_df[(res_df.event_id==e['id'])&(res_df.phase=='post')]['distance'].values[0] for e in EVENTS]

# 绘制柱状图（和原代码配色一致）
bars_pre  = ax.bar(x - width/2, pre_vals,  width, label='事件前14天（基线期）',
                   color='#AED6F1', edgecolor='#2980B9', linewidth=0.8, zorder=3)
bars_post = ax.bar(x + width/2, post_vals, width, label='事件后14天（响应期）',
                   color='#F1948A', edgecolor='#C0392B', linewidth=0.8, zorder=3)

# 标注变化方向（和原代码一致）
for i, (pre, post) in enumerate(zip(pre_vals, post_vals)):
    if not (np.isnan(pre) or np.isnan(post)):
        delta = post - pre
        arrow = "↑" if delta > 0 else "↓"
        color = "#C0392B" if delta > 0 else "#1A5276"
        ax.text(x[i], max(pre, post) + 0.003, f"{arrow}{abs(delta):.3f}",
                ha='center', va='bottom', fontsize=8.5, color=color, fontweight='bold')

# 2. 优化样本量标注：缩小字体+调整位置，避免和x轴标签重叠
for i, event in enumerate(EVENTS):
    eid = event['id']
    pre_row  = res_df[(res_df.event_id==eid)&(res_df.phase=='pre')].iloc[0]
    post_row = res_df[(res_df.event_id==eid)&(res_df.phase=='post')].iloc[0]
    # 字体缩小，位置略微上移，避免贴到底部
    ax.text(x[i]-width/2, 0.006, f"CN:{pre_row.cn_n}\nPH:{pre_row.ph_n}",
            ha='center', va='bottom', fontsize=6, color='#555')
    ax.text(x[i]+width/2, 0.006, f"CN:{post_row.cn_n}\nPH:{post_row.ph_n}",
            ha='center', va='bottom', fontsize=6, color='#555')

# 3. 优化事件类型标记：仅用符号，减少文字占用空间
for i, event in enumerate(EVENTS):
    eid  = event['id']
    tag  = "▲" if eid in CONFLICT_EVENTS else "●"
    col  = "#922B21" if eid in CONFLICT_EVENTS else "#1A5276"
    # 标记下移一点，和x轴标签错开
    ax.text(x[i], -0.013, tag, ha='center', va='top', fontsize=9, color=col, fontweight='bold')

# 4. 关键优化：x轴标签旋转+增加底部边距，彻底解决文字重叠
ax.set_xticks(x)
# 事件名称换行改为空格，同时旋转30度，减少横向重叠
ax.set_xticklabels([e['name'].replace('\n', ' ') for e in EVENTS],
                   fontsize=8, rotation=30, ha='right')  # ha='right'让标签和刻度对齐

ax.set_ylabel("中菲叙事语义距离（余弦距离）", fontsize=10)
ax.set_title("图5：关键事件节点前后中菲媒体叙事语义距离变化\n"
             "（距离越大 = 叙事错位越严重）", fontsize=11, pad=12)
ax.legend(fontsize=9, loc='upper left')
ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top','right']].set_visible(False)

# 强制增加底部边距，避免标签被裁剪
plt.subplots_adjust(bottom=0.23)
plt.tight_layout()

# 保存图片（高清300dpi，适配论文）
plt.savefig(BASE + "fig5_semantic_distance_fixed.png", dpi=300, bbox_inches='tight')
print("✅ 修复版图5已保存：", BASE + "fig5_semantic_distance_fixed.png")