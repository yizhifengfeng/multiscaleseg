import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
import cv2
from PIL import Image
import os
import replace_images

# 1. 定义基础参数
classes = ["crack", "peeling", "seepage"]  # 3类缺陷（英文便于文件命名）
wavelengths = [675 + i*30 for i in range(10)]  # 25个波长：675-975nm，步长12nm
img_size = (224, 224)  # 统一图像尺寸（A卡友好）
dataset_root = "simulated_multispectral_dataset"

# 2. 创建文件结构：数据集根目录/波长/类别/图像.jpg
for wave in wavelengths:
    for cls in classes:
        save_dir = os.path.join(dataset_root, str(wave), cls)
        os.makedirs(save_dir, exist_ok=True)
        # 每个波长-类别生成50张模拟图像（像素值模拟多光谱特征）
        for img_id in range(50):
            # 模拟多光谱图像：单通道，像素值随波长/类别略有差异
            if cls == "crack":
                # 裂缝：特定波长下像素差异更大（模拟真实特征）
                img = np.random.normal(loc=150 + wave/100, scale=20, size=img_size).astype(np.uint8)
            else:
                img = np.random.normal(loc=100 + wave/100, scale=20, size=img_size).astype(np.uint8)
            # 保存图像
            img_pil = Image.fromarray(img)
            img_pil.save(os.path.join(save_dir, f"{img_id}.jpg"))

print(f"模拟数据集生成完成！路径：{dataset_root}")
print(f"包含 {len(wavelengths)} 个波长，{len(classes)} 类缺陷，每类50张图像")

