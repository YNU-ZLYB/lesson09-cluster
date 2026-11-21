"""
Step 3：分群画像（检查点 3，本课高潮）

groupby('cluster') 输出每簇四特征均值表 → 还原业务含义 → 给出四个中文业务命名
→ 画"分组柱状图 + 雷达图"两张图。
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FEATS = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
LABELS = ["余额", "月均消费", "产品数", "逾期次数"]
NAMES = {
    0: "高消费活跃型",
    1: "低活跃低消费型",
    2: "高逾期违约风险型",
    3: "多产品优质稳健型",
}
COLORS = {0: "#4C78A8", 1: "#B0B7C3", 2: "#D9534F", 3: "#2E9E6B"}

df = pd.read_csv("data/bank_customers.csv")
X = StandardScaler().fit_transform(df[FEATS])
df["cluster"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)

# ---------- 1. 画像表 ----------
profile = df.groupby("cluster")[FEATS].mean().round(2)
profile.insert(0, "人数", df.groupby("cluster").size())
profile.insert(1, "占比", (df.groupby("cluster").size() / len(df)).map("{:.0%}".format))
profile.insert(2, "业务命名", [NAMES[c] for c in profile.index])
profile["违约率"] = df.groupby("cluster")["is_default"].mean().round(3)
print("各簇画像（均值）：")
print(profile.to_string())
profile.to_csv("outputs/cluster_profile.csv", encoding="utf-8-sig")

# ---------- 2. 分组柱状图（每张子图独立线性纵轴） ----------
fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
for ax, feat, label in zip(axes, FEATS, LABELS):
    vals = profile[feat].values
    bars = ax.bar([NAMES[c] for c in profile.index], vals,
                  color=[COLORS[c] for c in profile.index])
    ax.set_title(label, fontsize=12)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, max(vals) * 1.18)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}" if v > 100 else f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
fig.suptitle("四簇画像对比（每张子图独立纵轴）", fontsize=14)
fig.tight_layout()
fig.savefig("figures/profile_bar.png", dpi=150)

# ---------- 3. 雷达图（标准化后画，各维等权） ----------
scaled = pd.DataFrame(X, columns=FEATS)
scaled["cluster"] = df["cluster"]
radar = scaled.groupby("cluster").mean()

angles = np.linspace(0, 2 * np.pi, len(FEATS), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
for c in radar.index:
    vals = radar.loc[c].tolist()
    vals += vals[:1]
    ax.plot(angles, vals, "o-", linewidth=2, color=COLORS[c],
            label=f"簇{c} {NAMES[c]}（{profile.loc[c, '人数']}人）")
    ax.fill(angles, vals, alpha=0.12, color=COLORS[c])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(LABELS, fontsize=12)
ax.set_title("四簇客户画像雷达图（标准化后）", fontsize=14, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=9)
fig.tight_layout()
fig.savefig("figures/profile_radar.png", dpi=150)

print("\n已保存：figures/profile_bar.png、figures/profile_radar.png")
