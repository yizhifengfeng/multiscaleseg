
import os
import random
from PIL import Image  # 处理图片的核心工具
import numpy as np  # 辅助处理图片像素

# 第一步：定义关键路径
# 源图片路径：
source_img_dir = r"E:\研究生\深度学习\TAOBAO_4000张检测+分割\taobao_4000张检测+分割\Maks RCNN格式\images\train"
# 目标数据集路径：模拟数据集的根目录
target_dataset_dir = r"/simulated_multispectral_dataset"
# 模拟图片的尺寸
target_img_size = (224, 224)

# 第二步：获取源路径下的所有有效图片
def get_all_valid_images(img_dir):
    """
    功能：从指定文件夹获取所有jpg/png格式的图片路径
    参数：img_dir - 图片文件夹路径
    返回：有效图片路径的列表
    """
    # 支持的图片格式（避免读取非图片文件导致报错）
    valid_formats = [".jpg", ".jpeg", ".png", ".bmp"]
    img_paths = []

    # 遍历源文件夹里的所有文件
    for file_name in os.listdir(img_dir):
        # 获取文件后缀（比如.jpg），转为小写避免格式判断错误
        file_ext = os.path.splitext(file_name)[1].lower()
        # 如果是有效图片格式，记录完整路径
        if file_ext in valid_formats:
            full_path = os.path.join(img_dir, file_name)
            img_paths.append(full_path)

    # 如果没找到图片，直接终止并提示
    if len(img_paths) == 0:
        print(f"❌ 错误：在 {img_dir} 中未找到任何图片（支持格式：{valid_formats}）")
        exit()  # 退出程序
    else:
        print(f"✅ 成功找到 {len(img_paths)} 张有效图片")
    return img_paths


# 执行函数，获取源图片列表
source_img_paths = get_all_valid_images(source_img_dir)


#  第三步：遍历模拟数据集，随机替换图片
def replace_simulated_images():
    """
    核心功能：遍历模拟数据集的所有图片路径，用源图片随机替换
    """
    # 统计替换的图片数量
    replace_count = 0
    # 数据集的波长文件夹（比如675\975）
    for wave_folder in os.listdir(target_dataset_dir):
        wave_folder_path = os.path.join(target_dataset_dir, wave_folder)
        if not os.path.isdir(wave_folder_path):
            continue

        # 波长下的缺陷类别文件夹（crack/peeling等）
        for cls_folder in os.listdir(wave_folder_path):
            cls_folder_path = os.path.join(wave_folder_path, cls_folder)
            if not os.path.isdir(cls_folder_path):
                continue

            #模拟图片（比如0.jpg~49.jpg）
            for sim_img_name in os.listdir(cls_folder_path):
                sim_img_path = os.path.join(cls_folder_path, sim_img_name)
                # 模拟数据集生成jpg
                if os.path.splitext(sim_img_name)[1].lower() != ".jpg":
                    continue

                #随机选一张源图片
                random_source_img_path = random.choice(source_img_paths)  # 现在能找到random了

                try:
                    # 步骤1：打开源图片打开图片并转为RGB（避免透明通道）
                    source_img = Image.open(random_source_img_path).convert("RGB")
                    # 步骤2：调整尺寸匹配模拟图224x224）
                    source_img_resized = source_img.resize(target_img_size, Image.Resampling.LANCZOS)
                    # 步骤3：转为单通道（模拟图是单通道多光谱）
                    # 多光谱图片是单通道（灰度），所以要把彩色图转灰度
                    source_img_gray = source_img_resized.convert("L")
                    # 步骤4：替换保存
                    source_img_gray.save(sim_img_path)

                    replace_count += 1
                    # 每替换100张打印一次进度
                    if replace_count % 100 == 0:
                        print(f"🔄 已替换 {replace_count} 张图片，当前替换：{sim_img_path}")

                except Exception as e:
                    # 某张图片替换失败，跳过并提示，不终止程序
                    print(f"⚠️  替换图片 {sim_img_path} 失败：{str(e)}")
                    continue

    # 替换完成提示
    print(f"\n🎉 替换完成！总共替换了 {replace_count} 张模拟图片")


# ===================== 第四步：执行替换 =====================
if __name__ == "__main__":
    # 先检查目标数据集路径是否存在
    if not os.path.exists(target_dataset_dir):
        print(f"❌ 错误：模拟数据集路径 {target_dataset_dir} 不存在！请先运行模拟数据集生成代码")
    else:
        # 执行替换函数
        replace_simulated_images()