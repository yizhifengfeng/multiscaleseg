import os
import re
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.switch_backend("Agg")  # 无界面绘图，适配服务器/本地运行



# ===================== 注意力模块（与 visual2.py 一致） =====================
class dilateattention(nn.Module):
    def __init__(self, in_channel, depth):
        super(dilateattention, self).__init__()
        self.batch_norm = nn.BatchNorm3d(in_channel)
        self.relu = nn.ReLU()
        self.atrous_block1 = nn.Conv3d(in_channel, depth, 1, 1)
        self.atrous_block2 = nn.Conv3d(in_channel, depth, 3, 1, padding=2, dilation=2, groups=in_channel)
        self.atrous_block3 = nn.Conv3d(in_channel, depth, 3, 1, padding=4, dilation=4, groups=in_channel)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x_res = self.relu(self.batch_norm(x))
        v = self.atrous_block1(x_res)
        q = self.atrous_block2(x_res)
        k = self.atrous_block3(x_res)
        temp = (k * q) / (k.size(1) ** 0.5)
        attention_map = self.softmax(temp)
        output = v * attention_map + x
        return output


# ===================== 主模型（2060Ti 精简版，与 visual2.py 一致） =====================
class mymodel(nn.Module):
    def __init__(self, in_channel, classnum=1):
        super(mymodel, self).__init__()
        self.dim = 2
        self.conv3d1 = nn.Conv3d(in_channel, self.dim, kernel_size=(7, 7, 7), padding=3, stride=1)
        self.maxpooling1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dialatiaon1 = dilateattention(self.dim, self.dim)

        self.conv3d2 = nn.Conv3d(self.dim, self.dim, kernel_size=(7, 7, 7), padding=3, stride=1)
        self.maxpooling2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dialatiaon2 = dilateattention(self.dim, self.dim)

        self.conv3d3 = nn.Conv3d(self.dim, self.dim, kernel_size=(7, 7, 7), padding=3, stride=1)
        self.maxpooling3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dialatiaon3 = dilateattention(self.dim, self.dim)

        self.transpose1 = nn.ConvTranspose3d(self.dim, self.dim, kernel_size=2, stride=2)
        self.conv2d1 = nn.Conv3d(self.dim, self.dim, kernel_size=(7, 7, 7), padding=3, stride=1)

        self.transpose2 = nn.ConvTranspose3d(self.dim * 2, self.dim, kernel_size=2, stride=2)
        self.conv2d2 = nn.Conv3d(self.dim, self.dim, kernel_size=7, padding=3, stride=1)

        self.transpose3 = nn.ConvTranspose3d(self.dim * 2, self.dim, kernel_size=2, stride=2)
        self.conv2d3 = nn.Conv3d(self.dim, 1, kernel_size=7, padding=3, stride=1)
        self.final = nn.Conv2d(176, classnum, 3, 1, 1)

    def forward(self, x):
        x1 = self.conv3d1(x)
        x1 = self.maxpooling1(x1)
        x1 = self.dialatiaon1(x1)
        x2 = self.conv3d2(x1)
        x2 = self.maxpooling2(x2)
        x2 = self.dialatiaon2(x2)
        x3 = self.conv3d3(x2)
        x3 = self.maxpooling3(x3)
        x3 = self.dialatiaon3(x3)
        x4 = self.transpose1(x3)
        x4 = self.conv2d1(x4)
        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.transpose2(x4)
        x5 = self.conv2d2(x5)
        x6 = torch.cat([x5, x1], dim=1)
        x6 = self.transpose3(x6)
        x6 = self.conv2d3(x6)
        x6 = torch.squeeze(x6, dim=1)
        x6 = self.final(x6)
        return x6




# ===================== 核心路径配置（严格匹配你的文件夹） =====================
MODEL_PATH = r"D:\deeplearning\MS-CrackSeg-main\crackseg_2060ti.pth"
TEST_IMG_DIR = r"D:\deeplearning\MS-CrackSeg-main\sampleset\sampleset\testimg"  # 测试图像文件夹
TEST_MASK_DIR = r"D:\deeplearning\MS-CrackSeg-main\sampleset\sampleset\testmasknpy"  # 掩码文件夹
SAVE_ROOT = r"D:\deeplearning\MS-CrackSeg-main\output_results"  # 批量结果保存根目录
PRED_THRESHOLD = 0.5  # 裂缝预测阈值
SLICE_IDX = 88  # 高光谱可视化切片索引


# ===================== 工具函数 =====================
def calculate_metrics(pred, mask, eps=1e-7):
    """计算Dice和IoU评估指标"""
    pred = pred.flatten()
    mask = mask.flatten()
    tp = ((pred == 1) & (mask == 1)).sum()
    fp = ((pred == 1) & (mask == 0)).sum()
    fn = ((pred == 0) & (mask == 1)).sum()
    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    return {"Dice": float(dice), "IoU": float(iou)}


def get_mask_name(img_name):
    """根据图像文件名自动生成对应掩码文件名（严格匹配你的命名规则）"""
    # 图像名：xxx-z_xxx.npy → 掩码名：xxx_mask_classic_xxx.npy
    mask_name = re.sub(r'-z_', '_mask_classic_', img_name)
    return mask_name


# ===================== 批量测试主函数 =====================
def batch_test():
    # 1. 创建保存文件夹
    os.makedirs(SAVE_ROOT, exist_ok=True)
    # 创建子文件夹分类保存结果
    save_npy_dir = os.path.join(SAVE_ROOT, "pred_masks")
    save_viz_dir = os.path.join(SAVE_ROOT, "visualizations")
    os.makedirs(save_npy_dir, exist_ok=True)
    os.makedirs(save_viz_dir, exist_ok=True)

    # 2. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用设备: {device}")
    print(f"✅ 批量测试文件夹: {TEST_IMG_DIR}")
    print(f"✅ 结果保存路径: {SAVE_ROOT}\n")

    # 3. 加载模型
    model = mymodel(in_channel=1, classnum=1).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ 模型加载完成！\n")

    # 4. 获取文件夹内所有.npy图像文件
    img_files = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(".npy")]
    if not img_files:
        print("❌ 测试文件夹中未找到.npy文件！")
        return
    print(f"✅ 共找到 {len(img_files)} 张测试图像，开始批量预测...\n")

    # 5. 遍历所有图像，批量预测
    all_metrics = []
    with torch.no_grad():
        for idx, img_file in enumerate(img_files, 1):
            print(f"[{idx}/{len(img_files)}] 处理: {img_file}")

            # -------------------------- 1. 匹配文件路径 --------------------------
            img_path = os.path.join(TEST_IMG_DIR, img_file)
            mask_file = get_mask_name(img_file)
            mask_path = os.path.join(TEST_MASK_DIR, mask_file)

            # 检查掩码是否存在
            if not os.path.exists(mask_path):
                print(f"❌ 未找到对应掩码: {mask_file}，跳过\n")
                continue

            # -------------------------- 2. 加载数据（5维输入，修复维度报错） --------------------------
            img_raw = np.load(img_path)
            # 统一维度为 (176, H, W)
            if img_raw.shape[0] != 176:
                img = img_raw.transpose(2, 0, 1)
            else:
                img = img_raw
            # 构造模型需要的5维张量: [B=1, C=1, D=176, H, W]
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
            mask = np.load(mask_path)

            # -------------------------- 3. 模型推理 --------------------------
            output = model(img_tensor)
            pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_mask = (pred_prob > PRED_THRESHOLD).astype(np.uint8)

            # -------------------------- 4. 计算指标 --------------------------
            metrics = calculate_metrics(pred_mask, mask)
            all_metrics.append(metrics)
            print(f"   结果 → Dice: {metrics['Dice']:.4f} | IoU: {metrics['IoU']:.4f}")

            # -------------------------- 5. 批量保存结果 --------------------------
            # 保存预测掩码 (.npy)
            pred_save_path = os.path.join(save_npy_dir, f"pred_{os.path.splitext(img_file)[0]}.npy")
            np.save(pred_save_path, pred_mask)

            # 保存可视化对比图
            viz_save_path = os.path.join(save_viz_dir, f"viz_{os.path.splitext(img_file)[0]}.png")
            plt.figure(figsize=(18, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(img[SLICE_IDX], cmap="gray")
            plt.title(f"Hyperspectral Slice (Idx={SLICE_IDX})")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap="gray")
            plt.title("Ground Truth Mask")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(pred_mask, cmap="gray")
            plt.title(f"Pred Mask (Threshold={PRED_THRESHOLD})")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(viz_save_path, dpi=150)
            plt.close()
            print(f"   已保存 → 预测掩码 + 可视化图\n")

    # 6. 输出整体统计结果
    print("=" * 60)
    print("📊 批量测试完成！整体统计：")
    mean_dice = np.mean([m['Dice'] for m in all_metrics])
    mean_iou = np.mean([m['IoU'] for m in all_metrics])
    print(f"✅ 平均Dice系数: {mean_dice:.4f}")
    print(f"✅ 平均IoU系数: {mean_iou:.4f}")
    print(f"✅ 所有结果已保存至: {SAVE_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    batch_test()