import os
import numpy as np
import pandas as pd
import os
import numpy as np
import pandas as pd
# 从train.py文件导入train_and_evaluate函数（关键！）
from train import train_and_evaluate
def run_all_wavelengths(dataset_root, save_csv="wavelength_f1_results.csv"):
    """
    遍历所有波长，批量训练评估，保存F1结果
    dataset_root: 数据集根目录
    save_csv: 结果保存路径
    """
    # 1. 获取所有波长目录
    wave_dirs = [os.path.join(dataset_root, d) for d in os.listdir(dataset_root)
                 if os.path.isdir(os.path.join(dataset_root, d))]
    wave_dirs.sort(key=lambda x: int(os.path.basename(x)))  # 按波长数值排序

    # 2. 保存结果的列表
    results = []

    # 3. 遍历每个波长
    for wave_dir in wave_dirs:
        wave = os.path.basename(wave_dir)
        # 训练评估该波长
        f1_list = train_and_evaluate(wave_dir)
        # 计算均值和标准差
        f1_mean = np.mean(f1_list)
        f1_std = np.std(f1_list)
        # 保存结果
        results.append({
            "wavelength(nm)": int(wave),
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "f1_all": f1_list
        })
        print(f"\n===== 波长 {wave}nm 结果 =====")
        print(f"F1均值：{f1_mean:.4f}，标准差：{f1_std:.4f}")

    # 4. 保存到CSV文件
    df = pd.DataFrame(results)
    df = df.sort_values(by="f1_mean", ascending=False)  # 按F1均值降序
    df.to_csv(save_csv, index=False, encoding="utf-8")
    print(f"\n所有波长结果已保存到：{save_csv}")
    return df

# 执行批量运行（替换为你的数据集根目录）
dataset_root = "simulated_multispectral_dataset"  # 模拟数据
# dataset_root = "real_dataset"  # 真实数据（后续替换）
results_df = run_all_wavelengths(dataset_root)