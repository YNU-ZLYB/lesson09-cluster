"""
Step 2：肘部法则选 k（检查点 2）

k=1..9 逐个跑 k-均值，记录组内误差 SSE（kmeans.inertia_），画 SSE-k 折线并标出肘点。
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FEATS = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
df = pd.read_csv("data/bank_customers.csv")
X = StandardScaler().fit_transform(df[FEATS])

ks = range(1, 10)
sse = []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    sse.append(km.inertia_)
    print(f"k={k}  SSE={km.inertia_:8.2f}")

# 相邻两段的下降幅度：降幅突然变小的地方就是"肘"
print("\nk 从 k-1 到 k 的 SSE 降幅：")
for k in range(2, 10):
    drop = sse[k - 2] - sse[k - 1]
    print(f"  {k - 1}->{k}: {drop:8.2f}")

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(list(ks), sse, "o-", color="#1f77b4", linewidth=2, label="SSE")
# 肘部区间：k=3~4，3->4 的降幅从 130.9 掉到 79.5（跌幅约 40%），是整条曲线斜率变化最大处
for k in (3, 4):
    ax.axvline(k, color="#d62728", linestyle="--", linewidth=1, alpha=0.6)
ax.scatter([3, 4], [sse[2], sse[3]], s=170, facecolors="none",
           edgecolors="#d62728", linewidths=2, zorder=5, label="肘部区间 k=3~4")
ax.annotate("3→4 降幅 130.9→79.5\n（跌 39%，之后只降 68/49/38）",
            xy=(4, sse[3]), xytext=(4.7, 620), fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#d62728"))
ax.annotate("最终选用 k=4\n业务上能分开\"高消费活跃\"与\"低活跃低消费\"",
            xy=(4, sse[3]), xytext=(5.0, 480), fontsize=10, color="#2E9E6B",
            arrowprops=dict(arrowstyle="->", color="#2E9E6B"))
ax.set_xlabel("簇数 k")
ax.set_ylabel("组内误差 SSE")
ax.set_title("肘部法则：SSE 随 k 的变化（标准化后 4 特征）")
ax.set_xticks(list(ks))
ax.grid(alpha=0.3)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("figures/elbow.png", dpi=150)
print("\n已保存：figures/elbow.png")
