import pandas as pd
import matplotlib.pyplot as plt
# 1. 读取结果CSV
results_df = pd.read_csv("wavelength_f1_results.csv")

# 2. 筛选Top5波长（F1均值最高）
top5_waves = results_df.head(5)
print("===== F1均值Top5波长（成像差异最大） =====")
print(top5_waves[["wavelength(nm)", "f1_mean", "f1_std"]])

# 3. 若F1均值接近，选标准差更小的（稳定性优先）
# 例如：筛选F1均值>0.8的波长，再按标准差升序
stable_waves = results_df[results_df["f1_mean"] > 0.8].sort_values(by="f1_std")
print("\n===== 高F1+高稳定性波长 =====")
print(stable_waves[["wavelength(nm)", "f1_mean", "f1_std"]])

# 按波长数值排序，绘制误差棒图
results_df_sorted = results_df.sort_values(by="wavelength(nm)")

plt.figure(figsize=(12, 6))
# 误差棒：y轴=F1均值，误差=标准差
plt.errorbar(
    x=results_df_sorted["wavelength(nm)"],
    y=results_df_sorted["f1_mean"],
    yerr=results_df_sorted["f1_std"],
    fmt='o-', color='b', capsize=5
)
# 标注Top3波长
top3_waves = results_df.head(3)
for idx, row in top3_waves.iterrows():
    plt.annotate(
        f"{row['wavelength(nm)']}nm\nF1={row['f1_mean']:.4f}",
        xy=(row['wavelength(nm)'], row['f1_mean']),
        xytext=(5, 5), textcoords="offset points",
        fontsize=10, color='red'
    )

plt.xlabel("波长 (nm)")
plt.ylabel("宏平均F1-score（均值）")
plt.title("不同波长下建筑立面缺陷分类F1-score")
plt.grid(True, alpha=0.3)
plt.savefig("wavelength_f1_plot.png", dpi=300, bbox_inches='tight')
plt.show()