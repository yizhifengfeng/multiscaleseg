import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===================== 【用户配置区：默认完美适配你的数据集，无需修改】 =====================
ROOT_DIR = r"D:\deeplearning\MS-CrackSeg-main\sampleset"
# 高光谱数据所在子文件夹（训练+测试）
HSI_FOLDERS = ["img", "testimg"]
# 输出文件夹（自动创建）
OUTPUT_ROOT = os.path.join(ROOT_DIR, "spectral_indices")
# GaiaSky-mini 2 光谱参数：394-1001nm，176个波段
WAVELENGTH_MIN = 394  # nm
WAVELENGTH_MAX = 1001  # nm
NUM_BANDS = 176

# ===================== 【光谱波段定位（自动适配GaiaSky-mini 2，无需修改）】 =====================
# 生成176个波段的波长数组
WAVELENGTHS = np.linspace(WAVELENGTH_MIN, WAVELENGTH_MAX, NUM_BANDS)

def get_band_index(target_wave):
    """根据目标波长找到最接近的波段索引"""
    return np.argmin(np.abs(WAVELENGTHS - target_wave))

# 定位所有关键波段（自动计算，打印验证）
BAND_BLUE = get_band_index(450)    # 蓝光 ~450nm
BAND_GREEN = get_band_index(550)  # 绿光 ~550nm
BAND_RED = get_band_index(670)     # 红光 ~670nm
BAND_NIR = get_band_index(800)     # 近红外 ~800nm
BAND_SWIR = get_band_index(950)    # 短波红外 ~950nm（适配394-1001nm范围）

print("="*60)
print("GaiaSky-mini 2 关键波段定位结果（自动验证）：")
print(f"蓝光(450nm) → 索引: {BAND_BLUE}, 实际波长: {WAVELENGTHS[BAND_BLUE]:.1f}nm")
print(f"绿光(550nm) → 索引: {BAND_GREEN}, 实际波长: {WAVELENGTHS[BAND_GREEN]:.1f}nm")
print(f"红光(670nm) → 索引: {BAND_RED}, 实际波长: {WAVELENGTHS[BAND_RED]:.1f}nm")
print(f"近红外(800nm) → 索引: {BAND_NIR}, 实际波长: {WAVELENGTHS[BAND_NIR]:.1f}nm")
print(f"短波红外(950nm) → 索引: {BAND_SWIR}, 实际波长: {WAVELENGTHS[BAND_SWIR]:.1f}nm")
print("="*60)

# ===================== 【光谱指数计算（所有常用指数，无需修改）】 =====================
def calculate_all_indices(hsi):
    """
    输入高光谱数据，计算所有常用光谱指数
    输入格式: (H, W, 176) 或 (176, H, W)
    输出: 字典，key为指数名，value为指数图(H, W)
    """
    # 统一维度为 (H, W, C)
    if hsi.shape[0] == NUM_BANDS:
        hsi = hsi.transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
    H, W, C = hsi.shape
    assert C == NUM_BANDS, f"波段数不匹配！预期{NUM_BANDS}，实际{C}"

    # 提取关键波段（转为float32，避免整数溢出）
    B = hsi[:, :, BAND_BLUE].astype(np.float32)
    G = hsi[:, :, BAND_GREEN].astype(np.float32)
    R = hsi[:, :, BAND_RED].astype(np.float32)
    NIR = hsi[:, :, BAND_NIR].astype(np.float32)
    SWIR = hsi[:, :, BAND_SWIR].astype(np.float32)

    # 防除零常数
    eps = 1e-8

    # 1. NDVI (归一化植被指数)
    NDVI = (NIR - R) / (NIR + R + eps)

    # 2. GEMI (全球环境监测指数)
    GEMI = (2 * (NIR**2 - R**2) + 1.5 * NIR + 0.5 * R) / (NIR + R + 0.5 + eps)

    # 3. NBR (归一化燃烧率指数)
    NBR = (NIR - SWIR) / (NIR + SWIR + eps)

    # 4. NDWI (归一化水体指数, McFeeters 1996)
    NDWI = (G - NIR) / (G + NIR + eps)

    # 5. SI (土壤亮度指数)
    SI = np.sqrt((R**2 + G**2 + B**2) / 3)

    # 6. GNDVI (绿度归一化植被指数)
    GNDVI = (NIR - G) / (NIR + G + eps)

    # 7. NDBI (归一化建筑指数)
    NDBI = (SWIR - NIR) / (SWIR + NIR + eps)

    # 8. SAVI (土壤调整植被指数, L=0.5)
    SAVI = (NIR - R) / (NIR + R + 0.5 + eps) * 1.5

    # 9. EVI (增强型植被指数)
    EVI = 2.5 * (NIR - R) / (NIR + 6 * R - 7.5 * B + 1 + eps)

    # 10. 假彩色合成图 (False Color RGB: R=NIR, G=R, B=G)
    def norm(band):
        return (band - band.min()) / (band.max() - band.min() + eps)
    false_color = np.stack([norm(NIR), norm(R), norm(G)], axis=-1)  # (H, W, 3)

    return {
        "NDVI": NDVI,
        "GEMI": GEMI,
        "NBR": NBR,
        "NDWI": NDWI,
        "SI": SI,
        "GNDVI": GNDVI,
        "NDBI": NDBI,
        "SAVI": SAVI,
        "EVI": EVI,
        "FalseColor": false_color
    }

# ===================== 【批量处理函数（自动遍历所有文件）】 =====================
def process_hsi_folder(input_folder, output_folder):
    """处理单个高光谱文件夹，批量生成所有指数图和假彩色图"""
    os.makedirs(output_folder, exist_ok=True)
    hsi_files = [f for f in os.listdir(input_folder) if f.endswith(".npy")]
    print(f"\n📂 处理文件夹: {input_folder}，共{len(hsi_files)}个文件")

    for filename in tqdm(hsi_files, desc="生成光谱指数&假彩色图"):
        # 读取高光谱数据
        file_path = os.path.join(input_folder, filename)
        hsi = np.load(file_path)

        # 计算所有指数
        indices = calculate_all_indices(hsi)

        # 生成文件名前缀
        base_name = os.path.splitext(filename)[0]

        # 保存所有指数图
        for idx_name, idx_data in indices.items():
            if idx_name == "FalseColor":
                # 假彩色图（3通道RGB）
                save_path = os.path.join(output_folder, f"{base_name}_{idx_name}.png")
                plt.imsave(save_path, idx_data)
            else:
                # 单通道指数图，匹配专业colormap
                cmap = {
                    "NDVI": "viridis",
                    "GEMI": "viridis",
                    "NBR": "RdYlGn",
                    "NDWI": "Blues",
                    "SI": "gray",
                    "GNDVI": "viridis",
                    "NDBI": "coolwarm",
                    "SAVI": "viridis",
                    "EVI": "viridis"
                }.get(idx_name, "gray")

                save_path = os.path.join(output_folder, f"{base_name}_{idx_name}.png")
                plt.imsave(save_path, idx_data, cmap=cmap)

    print(f"✅ 文件夹{input_folder}处理完成！结果保存在: {output_folder}")

# ===================== 【主程序：运行批量处理】 =====================
if __name__ == "__main__":
    print("🚀 GaiaSky-mini 2 高光谱数据 光谱指数&假彩色图 批量生成工具")
    print(f"根目录: {ROOT_DIR}")
    print(f"输出目录: {OUTPUT_ROOT}")

    # 遍历训练/测试高光谱文件夹
    for folder in HSI_FOLDERS:
        input_folder = os.path.join(ROOT_DIR, folder)
        output_folder = os.path.join(OUTPUT_ROOT, folder)
        process_hsi_folder(input_folder, output_folder)

    print("\n🎉 所有文件处理完成！")
    print(f"📁 结果总目录: {OUTPUT_ROOT}")
    print("📊 生成内容: NDVI/GEMI/NBR/NDWI/SI/GNDVI/NDBI/SAVI/EVI + 假彩色RGB图")