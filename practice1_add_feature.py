"""
自主实践①：换特征集——加入 account_age_months（开户月数）重新画像，对比业务解读变化

原特征：balance / avg_monthly_spend / product_count / overdue_count（纯行为特征）
新特征：在上面基础上加 account_age_months（客户关系长度）

仍然 k=4，random_state=42。对比两套分群的：人数结构、画像表、与旧分群的一致性（ARI）。
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
NEW = BASE + ["account_age_months"]

df = pd.read_csv("data/bank_customers.csv")


def cluster(feats):
    X = StandardScaler().fit_transform(df[feats])
    km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)
    return km.labels_, km.inertia_


old_labels, old_sse = cluster(BASE)
df["cluster_old"] = old_labels
df["cluster_new"], new_sse = cluster(NEW)

print("== 两套分群的簇人数 ==")
cmp = pd.DataFrame({
    "原4特征": pd.Series(old_labels).value_counts().sort_index(),
    "加开户月数": pd.Series(df["cluster_new"]).value_counts().sort_index(),
})
print(cmp.to_string())
print(f"\nSSE：原4特征 {old_sse:.2f} → 加特征 {new_sse:.2f}")
print(f"两套分群一致性 ARI = {adjusted_rand_score(old_labels, df['cluster_new']):.3f}"
      "（1=完全一致，0=随机）")

print("\n== 新分群画像（5 特征均值）==")
prof_new = df.groupby("cluster_new")[NEW].mean().round(2)
prof_new.insert(0, "人数", df.groupby("cluster_new").size())
prof_new["违约率"] = df.groupby("cluster_new")["is_default"].mean().round(3)
print(prof_new.to_string())
prof_new.to_csv("outputs/profile_add_age.csv", encoding="utf-8-sig")

# 新分群和"开户时长"标签的对应关系：看新簇是不是被开户月数重新切开了
print("\n== 各新簇的开户月数区间 ==")
print(df.groupby("cluster_new")["account_age_months"]
        .agg(["min", "mean", "max"]).round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, col, title in [(axes[0], "cluster_old", "原 4 特征分群"),
                       (axes[1], "cluster_new", "加开户月数后分群")]:
    for c in sorted(df[col].unique()):
        sub = df[df[col] == c]
        ax.scatter(sub["balance"] / 1e4, sub["avg_monthly_spend"] / 1e4,
                   s=28, alpha=0.75, label=f"簇{c}（{len(sub)}人）")
    ax.set_xlabel("余额（万元）")
    ax.set_ylabel("月均消费（万元）")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/compare_add_age.png", dpi=150)
print("\n已保存：figures/compare_add_age.png")
