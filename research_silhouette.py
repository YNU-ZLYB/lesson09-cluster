"""
独立研究③：轮廓系数 silhouette_score 是什么？用代码算一次

定义（一句话）：对每个样本 a=到本簇其它点的平均距离，b=到最近邻簇的平均距离，
s = (b - a) / max(a, b)，取值 [-1, 1]，越接近 1 说明"本簇越紧、离别的簇越远"。
整体轮廓系数 = 所有样本 s 的平均。

注意：轮廓系数只需要 X 和标签，不需要真实标签，所以可以拿来在没有标准答案时评估聚类好坏。
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FEATS = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
df = pd.read_csv("data/bank_customers.csv")
X = StandardScaler().fit_transform(df[FEATS])

ks, scores = range(2, 9), []
for k in ks:
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    s = silhouette_score(X, labels)
    scores.append(s)
    print(f"k={k}  轮廓系数={s:.4f}")

best_k = list(ks)[scores.index(max(scores))]
print(f"\n最高分：k={best_k}，轮廓系数 {max(scores):.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(ks), scores, "o-", color="#2E9E6B", linewidth=2)
ax.scatter([best_k], [max(scores)], s=160, facecolors="none",
           edgecolors="#D9534F", linewidths=2, zorder=5)
ax.annotate(f"最高 k={best_k}\n{max(scores):.4f}", xy=(best_k, max(scores)),
            xytext=(best_k + 0.6, max(scores) + 0.012), fontsize=11,
            arrowprops=dict(arrowstyle="->", color="#D9534F"))
ax.set_xlabel("簇数 k")
ax.set_ylabel("平均轮廓系数")
ax.set_title("轮廓系数随 k 的变化（标准化后 4 特征）")
ax.set_xticks(list(ks))
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figures/silhouette.png", dpi=150)
print("已保存：figures/silhouette.png")
