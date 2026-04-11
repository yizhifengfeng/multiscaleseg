import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowCNN(nn.Module):
    """
    浅层CNN模型（适配3类建筑立面缺陷、单波长单通道输入、A卡低算力）
    结构：输入层 → 卷积层 → 池化层 → 全连接层 → Dropout → 输出层
    """
    def __init__(self, num_classes=3):
        super(ShallowCNN, self).__init__()
        # 1. 卷积层（16个3×3卷积核，单通道输入）
        self.conv1 = nn.Conv2d(
            in_channels=1,  # 单波长=单通道
            out_channels=16,  # 卷积核数量（A卡选16，减少计算）
            kernel_size=3,  # 小卷积核，捕捉缺陷细节
            stride=1,
            padding=1  # 保持输出尺寸和输入一致
        )
        # 2. 最大池化层（降维，减少计算量）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 3. 全连接层（将池化后的特征映射为128维向量，A卡选128）
        self.fc1 = nn.Linear(16 * 112 * 112, 128)  # 224/2=112（池化后尺寸）
        # 4. Dropout层（防止过拟合，新手可选）
        self.dropout = nn.Dropout(p=0.2)
        # 5. 输出层（5类缺陷分类）
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 前向传播
        x = self.pool(F.relu(self.conv1(x)))  # 卷积→激活→池化
        x = x.view(-1, 16 * 112 * 112)  # 展平为一维向量
        x = F.relu(self.fc1(x))  # 全连接→激活
        x = self.dropout(x)  # Dropout防止过拟合
        x = self.fc2(x)  # 输出层（未激活，后续用CrossEntropyLoss）
        return x

# 测试模型（确保无报错）
model = ShallowCNN(num_classes=3)
print("浅层CNN模型结构：")
print(model)
# 测试输入：batch_size=8（A卡友好）、单通道、224×224
test_input = torch.randn(8, 1, 224, 224)
test_output = model(test_input)
print(f"\n测试输入形状：{test_input.shape}")
print(f"测试输出形状：{test_output.shape}（8个样本，3类概率）")