import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


# ===================== 注意力模块 (已修复并简化) =====================
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
        # 原始输入首先通过BN和ReLU
        x_res = self.relu(self.batch_norm(x))

        v = self.atrous_block1(x_res)
        q = self.atrous_block2(x_res)
        k = self.atrous_block3(x_res)

        # 简单的元素级注意力
        temp = k * q
        # 增加缩放以提高稳定性
        temp = temp / (k.size(1) ** 0.5)

        attention_map = self.softmax(temp)
        output = v * attention_map

        # 正确的残差连接：将原始输入 x 添加到输出
        output += x
        return output


# ===================== 主模型（2060Ti 精简版） =====================
class mymodel(nn.Module):
    def __init__(self, in_channel, classnum=1):
        super(mymodel, self).__init__()
        self.dim = 2  # 2060Ti 只能用 2，用4会爆显存
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


# ===================== 数据集 =====================
class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.npy')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = np.load(img_path)
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img).float()

        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask = np.load(mask_path)
        mask = torch.from_numpy(mask).long()

        return img, mask


# ===================== 路径（你原来的不变） =====================
train_img_dir = r"D:\deeplearning\MS-CrackSeg-main\sampleset\img"
train_mask_dir = r"D:\deeplearning\MS-CrackSeg-main\sampleset\masknpy"
test_img_dir = r"D:\deeplearning\MS-CrackSeg-main\sampleset\testimg"
test_mask_dir = r"D:\deeplearning\MS-CrackSeg-main\sampleset\testmasknpy"

train_dataset = CrackDataset(train_img_dir, train_mask_dir)
test_dataset = CrackDataset(test_img_dir, test_mask_dir)

# ===================== 2060Ti 专用 DataLoader =====================
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


# ===================== 损失函数 =====================
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, target, smooth=1e-5):
        # 加上 BCE Loss, 使得训练更稳定，因为单纯的 Dice Loss 可能在前期导致梯度爆炸
        bce_loss = F.binary_cross_entropy_with_logits(inputs.view(-1), target.view(-1).float())

        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        target = target.view(-1)
        intersection = (inputs * target).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + target.sum() + smooth)
        dice_loss = 1 - dice

        # 结合 BCE 和 Dice Loss (可调节比例，比如 0.5 * BCE + 0.5 * Dice)
        return bce_loss + dice_loss


# ===================== 模型、优化器、设备 =====================
model = mymodel(in_channel=1, classnum=1)
criterion = DiceLoss()
# 降低学习率，增加 weight_decay，使用更稳定的优化器参数
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

# 学习率调度器，有助于稳定训练
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion.to(device)

# 混合精度（2060Ti 必开，提速+省显存）
scaler = torch.cuda.amp.GradScaler()
epochs = 20

print(f"使用设备: {device}")
print(f"模型通道 dim: {model.dim}")
print(f"batch_size: 1 (2060Ti 8G 最优)")

# ===================== 训练循环 =====================
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    train_bar = tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{epochs}] Train')
    for batch_idx, (imgs, masks) in enumerate(train_bar):
        imgs, masks = imgs.to(device), masks.to(device)

        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, masks)

        optimizer.zero_grad()

        # 梯度缩放
        scaler.scale(loss).backward()

        # 解码梯度，并进行梯度裁剪，防止梯度爆炸 (Max norm = 1.0)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_bar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'avg': f'{train_loss / (batch_idx + 1):.3f}'
        })

    # 测试
    model.eval()
    test_loss = 0.0
    test_bar = tqdm(test_loader, desc=f'Epoch [{epoch + 1}/{epochs}] Test')
    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(test_bar):
            imgs, masks = imgs.to(device), masks.to(device)
            # 测试时也可以开启 autocast 防止由于显存不足导致的问题
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, masks)
            test_loss += loss.item()

            test_bar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'avg': f'{test_loss / (batch_idx + 1):.3f}'
            })

    avg_train = train_loss / len(train_loader)
    avg_test = test_loss / len(test_loader)
    print(f"\nEpoch {epoch + 1} | Train Loss: {avg_train:.4f} | Test Loss: {avg_test:.4f}\n")

    # 更新学习率调度器
    scheduler.step(avg_test)

torch.save(model.state_dict(), "crackseg_2060ti.pth")
print("模型已保存：crackseg_2060ti.pth")

# ===================== 推理可视化 =====================
model.eval()
test_img, test_mask = test_dataset[0]
test_img = test_img.unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(test_img)
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(test_img[0, 0].cpu().numpy())
plt.title("Input")

plt.subplot(132)
plt.imshow(test_mask.cpu().numpy(), cmap='gray')
plt.title("GT")

plt.subplot(133)
plt.imshow(pred[0, 0].cpu().numpy(), cmap='gray')
plt.title("Pred")
plt.show()