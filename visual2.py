import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


# ===================== 1. 模型结构定义 (必须与训练时完全一致，保留修复后的变量名) =====================
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
        x1 = self.conv3d1(x);
        x1 = self.maxpooling1(x1);
        x1 = self.dialatiaon1(x1)
        x2 = self.conv3d2(x1);
        x2 = self.maxpooling2(x2);
        x2 = self.dialatiaon2(x2)
        x3 = self.conv3d3(x2);
        x3 = self.maxpooling3(x3);
        x3 = self.dialatiaon3(x3)
        x4 = self.transpose1(x3);
        x4 = self.conv2d1(x4);
        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.transpose2(x4);
        x5 = self.conv2d2(x5);
        x6 = torch.cat([x5, x1], dim=1)
        x6 = self.transpose3(x6);
        x6 = self.conv2d3(x6)
        x6 = torch.squeeze(x6, dim=1)
        x6 = self.final(x6)
        return x6


# ===================== 2. 数据加载 =====================
class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir, self.mask_dir = img_dir, mask_dir
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.npy')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])

    def __len__(self): return len(self.img_files)

    def __getitem__(self, idx):
        # 输入是 3D, 增加通道维 (1, Depth, H, W)
        img = np.expand_dims(np.load(os.path.join(self.img_dir, self.img_files[idx])), axis=0)
        # 掩码是 2D (H, W)
        mask = np.load(os.path.join(self.mask_dir, self.mask_files[idx]))
        return torch.from_numpy(img).float(), torch.from_numpy(mask).long(), self.img_files[idx]


# ===================== 3. 配置与路径 =====================
# 确保文件存在于当前目录下
WEIGHT_PATH = "crackseg_2060ti.pth"
# 修改为你电脑上的实际数据路径
TEST_IMG_DIR = r"D:\deeplearning\MS-CrackSeg-main\sampleset\testimg"
TEST_MASK_DIR = r"D:\deeplearning\MS-CrackSeg-main\sampleset\testmasknpy"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== 4. 核心可视化函数 (纯掩码版) =====================
def run_mask_visualization(sample_indices=[0, 1, 2], slice_idx=88, threshold=0.5):
    """
    展示纯掩码对比图：[原图切片 | 真值掩码 | 预测掩码]
    """
    # 初始化模型并加载权重
    print(f"正在加载权重: {WEIGHT_PATH} ...")
    model = mymodel(in_channel=1, classnum=1).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
    model.eval()
    print("模型加载完成，开始推理...")

    dataset = CrackDataset(TEST_IMG_DIR, TEST_MASK_DIR)
    num_samples = len(sample_indices)

    # 创建画布 (num_samples 行, 3 列)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    # 处理只查看一个样本时的布局问题
    if num_samples == 1: axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            if idx >= len(dataset):
                print(f"索引 {idx} 超出数据集大小，跳过。")
                continue

            img, mask, filename = dataset[idx]
            # 推理需要增加 Batch 维: (1, 1, Depth, H, W)
            img_input = img.unsqueeze(0).to(DEVICE)

            # 模型预测 (输出是 2D: (1, H, W))
            output = model(img_input)
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()  # 概率图 (H, W)
            # 应用阈值生成二值预测掩码 (0 或 1)
            pred_mask = (prob_map > threshold).astype(np.float32)

            # --- 准备展示数据 ---
            # 1. 输入数据的 2D 切片 (用于背景参考)
            input_slice = img[0, slice_idx].numpy()  # (H, W)
            # 2. 转换为 numpy 的真值掩码
            gt_mask = mask.numpy()  # (H, W)

            # --- 绘图 ---
            # 第一列: Original Image Slice
            axes[i, 0].imshow(input_slice, cmap='gray')
            axes[i, 0].set_title(f"Sample: {filename}\nInput (Slice {slice_idx})")
            axes[i, 0].axis('off')

            # 第二列: Ground Truth Mask
            axes[i, 1].imshow(gt_mask, cmap='gray')
            axes[i, 1].set_title("Ground Truth Mask")
            axes[i, 1].axis('off')

            # 第三列: Prediction Mask
            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title(f"Prediction Mask (Thresh={threshold})")
            axes[i, 2].axis('off')

    plt.tight_layout()
    print("可视化完成，正在显示窗口...")
    plt.show()


# ===================== 5. 损失曲线绘图函数 (保留供使用) =====================
def plot_loss(train_losses, val_losses):
    if not train_losses or not val_losses:
        print("未提供 Loss 数据，跳过绘图。")
        return
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.title('Training History', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


if __name__ == "__main__":
    # 想要看不同的数组（样本），修改这个列表即可，例如 [5, 10, 15]
    run_mask_visualization(sample_indices=[0, 3, 5], slice_idx=88, threshold=0.5)

    # 如果训练时记录了 Loss，填入下方并取消注释即可绘图
    # train_history = [...]
    # val_history = [...]
    # plot_loss(train_history, val_history)