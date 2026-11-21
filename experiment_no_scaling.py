"""
🧪 故意实验：不做标准化直接聚类，对比两种结果的画像表

预期：balance（十万级）会碾压其他三列，分群退化成"按余额切四段"。
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FEATS = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
df = pd.read_csv("data/bank_customers.csv")

raw = df[FEATS].values
std = StandardScaler().fit_transform(raw)

res = {}
for name, X in [("不标准化", raw), ("标准化后", std)]:
    km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)
    p = df.assign(cluster=km.labels_).groupby("cluster")[FEATS].mean().round(1)
    p.insert(0, "人数", df.assign(c=km.labels_).groupby("c").size())
    res[name] = p
    print(f"\n===== {name} =====")
    print(p.to_string())
    print(f"SSE = {km.inertia_:.2f}")

print("\n===== 结论对比 =====")
for name, p in res.items():
    rng = p["balance"].max() - p["balance"].min()
    print(f"{name}：四簇余额极差 {rng:,.0f} 元；"
          f"逾期均值极差 {p['overdue_count'].max() - p['overdue_count'].min():.2f}；"
          f"产品数极差 {p['product_count'].max() - p['product_count'].min():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (name, p) in zip(axes, res.items()):
    x = range(len(p))
    ax.bar(x, p["balance"] / 1e4, color="#4C78A8", width=0.55, label="余额（万元）")
    ax2 = ax.twinx()
    ax2.plot(x, p["overdue_count"], "o-", color="#D9534F", label="逾期次数")
    ax2.plot(x, p["product_count"], "s--", color="#2E9E6B", label="产品数")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"簇{i}" for i in p.index])
    ax.set_ylabel("余额（万元）")
    ax2.set_ylabel("逾期次数 / 产品数")
    ax.set_title(f"{name}：余额几乎决定一切" if name == "不标准化" else f"{name}：四维都有区分度")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("figures/no_scaling.png", dpi=150)
print("\n已保存：figures/no_scaling.png")
