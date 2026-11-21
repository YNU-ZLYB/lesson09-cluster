"""
独立研究②：k-均值对"环形分布"数据会失败 —— 用玩具数据验证，并对比 DBSCAN

做法：sklearn 生成"同心圆环"数据（内圈 + 外圈，两类），
     分别用 KMeans(k=2) 和 DBSCAN 聚类，用 ARI 与散点图对比。

预期：k-均值只能用"直线"切空间，两个同心圆环它切不开；
     DBSCAN 按密度连通，能把两个环各自完整圈出来。
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_circles
from sklearn.metrics import adjusted_rand_score

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

X, y_true = make_circles(n_samples=600, factor=0.45, noise=0.06, random_state=0)

km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
db = DBSCAN(eps=0.15, min_samples=5).fit(X)

ari_km = adjusted_rand_score(y_true, km.labels_)
ari_db = adjusted_rand_score(y_true, db.labels_)
print(f"KMeans(k=2)  与真实标签一致性 ARI = {ari_km:.3f}   （1=完全正确）")
print(f"DBSCAN(eps=0.15, min_samples=5) ARI = {ari_db:.3f}")
print(f"DBSCAN 找到的簇数：{len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)}，"
      f"噪声点 {-1 in db.labels_} 共 {(db.labels_ == -1).sum()} 个")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="coolwarm", s=14)
axes[0].set_title("真实结构：两个同心圆环")
axes[1].scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="coolwarm", s=14)
axes[1].set_title(f"KMeans(k=2)  ARI={ari_km:.3f}\n只会用直线切，把两个环各切一半")
axes[2].scatter(X[db.labels_ == 0, 0], X[db.labels_ == 0, 1],
                c="#d62728", s=14, label="簇 0")
axes[2].scatter(X[db.labels_ == 1, 0], X[db.labels_ == 1, 1],
                c="#1f77b4", s=14, label="簇 1")
axes[2].scatter(X[db.labels_ == -1, 0], X[db.labels_ == -1, 1],
                c="black", marker="x", s=30, label="噪声")
axes[2].legend(fontsize=8, markerscale=1.6)
axes[2].set_title(f"DBSCAN  ARI={ari_db:.3f}\n按密度连通，两个环分得干净")

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
fig.tight_layout()
fig.savefig("figures/donut_dbscan.png", dpi=150)
print("\n已保存：figures/donut_dbscan.png")
