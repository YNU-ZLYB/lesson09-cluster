"""
自主实践③：二维散点投影——任选两特征，按 cluster 着色

左图：余额 × 月均消费（最能体现"价值"的两个维度）
右图：产品数 × 逾期次数（最能体现"粘性/风险"的两个维度，点加抖动避免重叠）
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FEATS = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
NAMES = {0: "高消费活跃型", 1: "低活跃低消费型", 2: "高逾期违约风险型", 3: "多产品优质稳健型"}
COLORS = {0: "#4C78A8", 1: "#B0B7C3", 2: "#D9534F", 3: "#2E9E6B"}

df = pd.read_csv("data/bank_customers.csv")
X = StandardScaler().fit_transform(df[FEATS])
df["cluster"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)
centers = df.groupby("cluster")[FEATS].mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

for c in range(4):
    sub = df[df["cluster"] == c]
    axes[0].scatter(sub["balance"] / 1e4, sub["avg_monthly_spend"] / 1e4,
                    s=32, alpha=0.8, color=COLORS[c], label=f"簇{c} {NAMES[c]}")
axes[0].scatter(centers["balance"] / 1e4, centers["avg_monthly_spend"] / 1e4,
                marker="X", s=220, c="black", edgecolors="white", linewidths=1.2,
                label="簇中心", zorder=5)
axes[0].set_xlabel("余额（万元）")
axes[0].set_ylabel("月均消费（万元）")
axes[0].set_title("二维投影：余额 × 月均消费")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# 产品数、逾期次数都是小整数，直接画会严重重叠，加一点随机抖动
rng = pd.Series(range(len(df)))
jit_p = (rng % 11 - 5) / 12
jit_o = (rng % 7 - 3) / 10
for c in range(4):
    sub = df[df["cluster"] == c]
    idx = sub.index
    axes[1].scatter(sub["product_count"] + jit_p.loc[idx],
                    sub["overdue_count"] + jit_o.loc[idx],
                    s=32, alpha=0.75, color=COLORS[c], label=f"簇{c} {NAMES[c]}")
axes[1].set_xlabel("持有产品数（含轻微抖动）")
axes[1].set_ylabel("逾期次数（含轻微抖动）")
axes[1].set_title("二维投影：产品数 × 逾期次数")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig("figures/scatter_2d.png", dpi=150)

print("各簇中心：")
print(centers.round(2))
print("\n已保存：figures/scatter_2d.png")
