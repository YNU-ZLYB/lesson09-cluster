"""
自主实践②：簇的稳定性——random_state 换三个值，画像是否大变？

做法：固定 4 特征、k=4，random_state 分别取 0 / 42 / 2024，各跑一次。
比较：①簇人数结构 ②画像均值表 ③两两 ARI（调整兰德指数）④轮廓系数
"""

import itertools

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FEATS = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
SEEDS = [0, 42, 2024]
df = pd.read_csv("data/bank_customers.csv")
X = StandardScaler().fit_transform(df[FEATS])

labels, profiles = {}, {}
for s in SEEDS:
    km = KMeans(n_clusters=4, random_state=s, n_init=10).fit(X)
    labels[s] = km.labels_
    sil = silhouette_score(X, km.labels_)
    p = df.assign(cluster=km.labels_).groupby("cluster")[FEATS].mean().round(1)
    p.insert(0, "人数", df.assign(c=km.labels_).groupby("c").size())
    profiles[s] = p
    print(f"随机种子 {s}：人数 {list(p['人数'])}  SSE={km.inertia_:.2f}  轮廓系数={sil:.4f}")

print("\n两两一致性（ARI，1=分群完全相同）：")
for a, b in itertools.combinations(SEEDS, 2):
    print(f"  seed {a} vs seed {b}: ARI = {adjusted_rand_score(labels[a], labels[b]):.4f}")

# 用同一套业务命名规则去对齐三个种子的簇（按"违约率高低 + 消费高低"排序）
order_names = ["高消费活跃型", "低活跃低消费型", "高逾期违约风险型", "多产品优质稳健型"]

print("\n== 三个种子下的画像表（簇编号已按业务含义对齐）==")
aligned = {}
for s in SEEDS:
    lab = labels[s]
    info = df.assign(c=lab).groupby("c").agg(
        spend=("avg_monthly_spend", "mean"),
        overdue=("overdue_count", "mean"),
        product=("product_count", "mean"),
        n=("c", "size"),
    )
    # 业务规则对齐：逾期最高→风险型；剩余里产品最多→多产品稳健；再剩余里消费最高→高消费活跃
    risky = info["overdue"].idxmax()
    multi = info.drop(risky)["product"].idxmax()
    rest = [i for i in info.index if i not in (risky, multi)]
    active = info.loc[rest, "spend"].idxmax()
    quiet = [i for i in rest if i != active][0]
    mapping = {risky: "高逾期违约风险型", multi: "多产品优质稳健型",
               active: "高消费活跃型", quiet: "低活跃低消费型"}
    p = df.assign(c=lab).groupby("c")[FEATS].mean()
    p["人数"] = info["n"]
    p.index = [mapping[i] for i in p.index]
    aligned[s] = p.loc[order_names].round(1)
    print(f"\n--- seed {s} ---")
    print(aligned[s].to_string())

# 数值差异：同一业务簇在三个种子之间的均值极差
arr = pd.concat(aligned, axis=1)
spread = pd.DataFrame({
    "最大极差": [max(arr.loc[n, (s, feat)] for s in SEEDS)
                 - min(arr.loc[n, (s, feat)] for s in SEEDS)
             for n in order_names for feat in FEATS],
    "特征": [feat for n in order_names for feat in FEATS],
    "业务簇": [n for n in order_names for feat in FEATS],
})
print("\n== 同一业务簇在三种子间的最大极差 ==")
print(spread[spread["最大极差"] > 0].to_string(index=False))

# 画图：三个种子 × 四特征 的分组柱状
fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
for ax, feat, label in zip(axes, FEATS, ["余额", "月均消费", "产品数", "逾期次数"]):
    w = 0.26
    for i, s in enumerate(SEEDS):
        ax.bar([j + (i - 1) * w for j in range(4)], aligned[s][feat].values,
               width=w, label=f"seed={s}")
    ax.set_xticks(range(4))
    ax.set_xticklabels(order_names, rotation=25, fontsize=8)
    ax.set_title(label)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
axes[0].legend(fontsize=8)
fig.suptitle("三种随机种子下的画像对比（按业务含义对齐）", fontsize=14)
fig.tight_layout()
fig.savefig("figures/stability.png", dpi=150)
print("\n已保存：figures/stability.png")
