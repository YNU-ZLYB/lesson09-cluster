"""
Step 1：对 bank_customers.csv 做 KMeans(k=4) 聚类，输出各簇人数（检查点 1）

特征选择理由：
  balance（余额）/ avg_monthly_spend（月均消费）/ product_count（持有产品数）
  / overdue_count（逾期次数）都是"客户行为与价值"特征，聚出来的群才能讲成客户画像。
  故意不用 age：年龄是人口属性，混进去会把客户切成"年轻人/老年人"，业务上没意义。

流程：读数据 → 选 4 列 → 标准化 → KMeans(k=4) → 簇编号写回 df → 统计人数
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATS = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
DATA = "data/bank_customers.csv"

df = pd.read_csv(DATA)

# 1. 标准化：balance 是十万级，overdue_count 只有 0~3，不标准化的话距离全被余额支配
X = StandardScaler().fit_transform(df[FEATS])

# 2. k-均值，random_state 固定保证结果可复现
km = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = km.fit_predict(X)

# 3. 检查点 1：各簇人数
counts = df["cluster"].value_counts().sort_index()
print("各簇人数：")
for c, n in counts.items():
    print(f"  簇 {c}：{n} 人（{n / len(df):.1%}）")
print(f"合计：{counts.sum()} 人")
print(f"SSE（inertia_）：{km.inertia_:.2f}")

# 4. 簇中心还原到原始量纲，方便看每簇大概是什么客户
centers = pd.DataFrame(
    StandardScaler().fit(df[FEATS]).inverse_transform(km.cluster_centers_),
    columns=FEATS,
)
centers.index.name = "cluster"
print("\n各簇中心（原始单位）：")
print(centers.round(2))

df.to_csv("outputs/clustered_k4.csv", index=False)
print("\n已保存：outputs/clustered_k4.csv")
