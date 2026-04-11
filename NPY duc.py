import numpy as np
import matplotlib.pyplot as plt


# ===================== 核心：配置中文显示 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体（SimHei）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号“-”显示为方块的问题

# 替换成你的.npy文件路径
npy_path = "D:\deeplearning\MS-CrackSeg-main\sampleset\img/all_bands_mosaic_crack46all_classic-z_r1c1col51row39overlap60.npy"  # 高光谱图像
# npy_path = "D:\deeplearning\MS-CrackSeg-main\sampleset\masknpy\all_bands_mosaic_crack46all_classic_mask_classic_r1c1col51row39overlap60.npy"  # 掩码（二值图）

# 加载.npy文件
data = np.load(npy_path)

# 打印核心信息
print("数据形状（shape）:", data.shape)  # 高光谱：[D, H, W]，掩码：[H, W]
print("数据类型（dtype）:", data.dtype)  # 通常是float32（图像）或uint8/int64（掩码）
print("数值范围：min={:.2f}, max={:.2f}".format(data.min(), data.max()))  # 看像素值范围
print("前5个光谱通道的均值：", data[:5].mean(axis=(1,2)))  # 高光谱：看各波段均值

# ===================== 1. 加载数据 =====================
# 替换成你的文件路径
img_path = "D:\deeplearning\MS-CrackSeg-main\sampleset\img/all_bands_mosaic_crack46all_classic-z_r1c1col51row39overlap60.npy"    # 高光谱图像
mask_path = "D:\deeplearning\MS-CrackSeg-main\sampleset\masknpy/all_bands_mosaic_crack46all_classic_mask_classic_r1c1col51row39overlap60.npy"  # 对应掩码

img = np.load(img_path)       # shape: [D, H, W]（D=光谱通道数，H=高度，W=宽度）
mask = np.load(mask_path)     # shape: [H, W]（二值图：1=裂缝，0=背景）

# ===================== 2. 可视化高光谱图像（选单个波段） =====================
plt.figure(figsize=(12, 5))

# 子图1：选第几个光谱通道（可改数字，比如50、100）
plt.subplot(131)
plt.imshow(img[49], cmap='gray')  # img[0] = 第1个波段，cmap='gray'灰度显示
plt.title(f"第50个光谱通道 (shape={img.shape})")
plt.axis('off')

# 子图2：选中间的光谱通道（比如第88个，D=176时）
plt.subplot(132)
plt.imshow(img[87], cmap='gray')  # Python索引从0开始，87=第88个波段
plt.title("第88个光谱通道")
plt.axis('off')

# 子图3：掩码（裂缝）
plt.subplot(133)
plt.imshow(mask, cmap='gray')  # 掩码是二值图，gray更清晰
plt.title(f"掩码（裂缝=1，背景=0）")
plt.axis('off')

plt.tight_layout()
plt.show()
