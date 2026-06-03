# ==============================
# 独立脚本：仅生成 图6 叙事重心漂移图
# 无任何额外依赖 | 从头计算 | 修复所有绘图bug
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# ----------------------
# 0. 基础配置（字体+你的本地路径）
# ----------------------
plt.rcParams['font.family'] = ['Arial Unicode MS', 'STHeiti', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 【和你原代码完全一致的路径，无需修改】
BASE = "/Users/keying/Desktop/中菲媒体对比论文撰写/BERT/"

# ----------------------
# 1. 仅读取你原本就有的文件（核心！解决报错）
# ----------------------
df_events = pd.read_pickle(BASE + "preprocessed_events.pkl")
emb_events = np.load(BASE + "embeddings_events.npy")
df_events = df_events.reset_index(drop=True)
df_events['embedding'] = list(emb_events)  # 把向量绑定到数据

# ----------------------
# 2. 事件配置（完全沿用你的设置）
# ----------------------
EVENTS = [
    {"id":"E1","name":"激光照射\n(2023.02)","name_short":"E1"},
    {"id":"E2","name":"水炮+马科斯\n声明(2023.08)","name_short":"E2"},
    {"id":"E3","name":"上海磋商\n(2024.01)","name_short":"E3"},
    {"id":"E4","name":"登船夺械\n(2024.06)","name_short":"E4"},
    {"id":"E5","name":"仁爱礁临时\n安排(2024.07)","name_short":"E5"},
]
CONFLICT_EVENTS = {"E1","E2","E4"}   # 冲突事件
DETENTE_EVENTS  = {"E3","E5"}        # 缓和事件
country_colors  = {'China': '#E74C3C', 'Philippine': '#2980B9'}  # 红蓝配色

# ----------------------
# 3. 核心函数：提取向量 + 计算叙事得分
# ----------------------
# 提取对应事件/国家/阶段的向量
def get_group_embeddings(df, event_id, country, phase):
    subset = df[(df['event_id'] == event_id) &
                (df['Country'] == country) &
                (df['phase'] == phase)]
    if len(subset) == 0:
        return np.array([]).reshape(0, emb_events.shape[1])
    return np.stack(subset['embedding'].values)

# 加载模型 + 定义对抗/合作锚点词
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
CONFRONTATION_ANCHORS = [
    "military confrontation aggression threat provocation illegal violation",
    "coast guard vessel water cannon harassment dangerous",
    "sovereignty violation armed attack escalation"
]
COOPERATION_ANCHORS = [
    "peaceful dialogue cooperation stability regional development",
    "ASEAN partnership bilateral consultation agreement",
    "negotiation diplomatic channel de-escalation"
]

# 计算锚点平均向量
conf_anchor = model.encode(CONFRONTATION_ANCHORS, normalize_embeddings=True).mean(axis=0, keepdims=True)
coop_anchor = model.encode(COOPERATION_ANCHORS, normalize_embeddings=True).mean(axis=0, keepdims=True)

# 叙事得分：正值=对抗，负值=合作
def narrative_score(embeddings):
    if len(embeddings) == 0:
        return np.nan
    sim_conf = cosine_similarity(embeddings, conf_anchor).mean()
    sim_coop = cosine_similarity(embeddings, coop_anchor).mean()
    return float(sim_conf - sim_coop)

# ----------------------
# 4. 从头计算所有得分（无依赖！）
# ----------------------
score_results = []
for event in EVENTS:
    eid = event['id']
    for country in ['China', 'Philippine']:
        for phase in ['pre', 'post']:
            emb = get_group_embeddings(df_events, eid, country, phase)
            score = narrative_score(emb)
            score_results.append({
                'event_id': eid,
                'country': country,
                'phase': phase,
                'score': score
            })
score_df = pd.DataFrame(score_results)

# ----------------------
# 5. 绘图：修复bug版 图6
# ----------------------
fig, axes = plt.subplots(1, 5, figsize=(14, 5), sharey=True)
fig.suptitle("图6：关键事件节点前后叙事重心漂移\n（正值偏对抗，负值偏合作；箭头=事件后漂移趋势）", fontsize=12, y=1.02)

for ax, event in zip(axes, EVENTS):
    eid = event['id']
    # 中国：下方-0.15 | 菲律宾：上方0.15
    for country, offset in [('China', -0.15), ('Philippine', 0.15)]:
        pre_score = score_df[(score_df.event_id==eid) &
                             (score_df.country==country) &
                             (score_df.phase=='pre')]['score'].values[0]
        post_score = score_df[(score_df.event_id==eid) &
                              (score_df.country==country) &
                              (score_df.phase=='post')]['score'].values[0]
        color = country_colors[country]

        # 绘制：圆点(前) → 方块(后) + 箭头
        if not np.isnan(pre_score) and not np.isnan(post_score):
            ax.annotate("", xy=(post_score, offset), xytext=(pre_score, offset),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
            ax.plot(pre_score, offset, 'o', color=color, ms=8, zorder=5)   # 事件前
            ax.plot(post_score, offset, 's', color=color, ms=8, zorder=5)  # 事件后
            ax.text(post_score, offset + 0.07, f"{post_score:.3f}", ha='center', fontsize=7.5, color=color)

    # 样式设置
    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_title(event['name'], fontsize=9)
    ax.set_xlim(-0.20, 0.12)  # 适配负值，不截断
    ax.set_yticks([-0.15, 0.15])
    ax.set_yticklabels(['中国', '菲律宾'], fontsize=9)
    ax.spines[['top','right','left']].set_visible(False)
    ax.set_facecolor("#FADBD8" if eid in CONFLICT_EVENTS else "#D6EAF8")

# 图例
patches = [
    mpatches.Patch(color='#E74C3C', label='中国媒体'),
    mpatches.Patch(color='#2980B9', label='菲律宾媒体'),
    mpatches.Patch(color="#FADBD8", label='冲突事件'),
    mpatches.Patch(color="#D6EAF8", label='缓和事件'),
]
fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.08))
axes[0].set_xlabel("← 合作语义        对抗语义 →", fontsize=9)

# 保存高清图片
plt.tight_layout()
plt.savefig(BASE + "fig6_修复版.png", dpi=300, bbox_inches='tight')
print("✅ 图6 已成功生成！保存路径：", BASE + "fig6_修复版.png")