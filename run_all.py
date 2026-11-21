"""
一键运行全部脚本（按课堂顺序：Step1-3 → 自主实践 → 独立研究）

python run_all.py
"""

import runpy
import sys
import time

SCRIPTS = [
    ("Step1 聚类跑通（检查点1：四簇人数）", "step1_kmeans.py"),
    ("Step2 肘部法则（检查点2：肘部图）", "step2_elbow.py"),
    ("Step3 分群画像（检查点3：画像+四中文命名）", "step3_profile.py"),
    ("自主实践① 加 account_age_months 重画像", "practice1_add_feature.py"),
    ("自主实践② 三种随机种子稳定性", "practice2_stability.py"),
    ("自主实践③ 二维散点投影", "practice3_scatter.py"),
    ("故意实验 不标准化直接聚类", "experiment_no_scaling.py"),
    ("独立研究② 环形数据 vs DBSCAN", "research_donut_dbscan.py"),
    ("独立研究③ 轮廓系数", "research_silhouette.py"),
]

for title, script in SCRIPTS:
    print("\n" + "=" * 70)
    print(f"▶ {title}    [{script}]")
    print("=" * 70)
    t0 = time.time()
    try:
        runpy.run_path(script, run_name="__main__")
    except Exception as e:  # 单个脚本失败不阻塞后面的
        print(f"✗ {script} 运行失败：{e}", file=sys.stderr)
    print(f"（耗时 {time.time() - t0:.1f}s）")

print("\n全部脚本执行完毕，图表在 figures/，表格在 outputs/。")
