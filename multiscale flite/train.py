import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import torch
# 补充f1_score导入
from sklearn.metrics import f1_score
from shallowCNN import ShallowCNN

# 1. 自定义多光谱数据集类
class MultispectralDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 读取单波长单通道图像
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('L')  # 转为灰度图（单通道）
        label = self.labels[idx]
        # 基础预处理（仅尺寸+归一化，后续可加降噪/裁剪）
        if self.transform:
            img = self.transform(img)
        return img, label

# 2. 数据加载函数（按波长加载，8:2划分训练/测试集）
def load_wavelength_data(wave_dir, img_size=(224, 224)):
    """
    wave_dir: 单个波长的目录（如 "simulated_multispectral_dataset/700"）
    return: 训练加载器、测试加载器
    """
    # 定义基础预处理
    transform = transforms.Compose([
        transforms.Resize(img_size),  # 统一尺寸
        transforms.ToTensor(),  # 转为Tensor，像素值0-1归一化
    ])

    # 收集所有图像路径和标签（3类缺陷：0-4）
    img_paths = []
    labels = []
    cls2idx = {"crack":0, "peeling":1, "seepage":2}
    for cls_name, cls_idx in cls2idx.items():
        cls_dir = os.path.join(wave_dir, cls_name)
        for img_name in os.listdir(cls_dir):
            img_paths.append(os.path.join(cls_dir, img_name))
            labels.append(cls_idx)

    # 8:2划分训练/测试集（所有波长用相同随机种子，避免偏差）
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        img_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 创建数据集和加载器（batch_size=8，A卡友好）
    train_dataset = MultispectralDataset(train_paths, train_labels, transform)
    test_dataset = MultispectralDataset(test_paths, test_labels, transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    return train_loader, test_loader


import torch.optim as optim


def train_and_evaluate(wave_dir, num_repeats=3, num_epochs=50):
    """
    单个波长的多次训练+评估
    wave_dir: 单个波长的目录
    num_repeats: 重复实验次数（参考文献，选3次）
    num_epochs: 训练轮数（A卡选50，足够收敛）
    return: 3次实验的宏平均F1-score列表
    """
    f1_list = []  # 保存5次实验的F1值
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n开始训练波长：{os.path.basename(wave_dir)}，设备：{device}")

    for repeat in range(num_repeats):
        print(f"\n===== 重复实验 {repeat+1}/{num_repeats} =====")
        # 1. 初始化模型、优化器、损失函数
        model = ShallowCNN(num_classes=3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # 参考文献的优化器
        criterion = nn.CrossEntropyLoss()  # 多分类损失函数

        # 2. 加载该波长的数据
        train_loader, test_loader = load_wavelength_data(wave_dir)

        # 3. 训练模型
        model.train()
        for epoch in tqdm(range(num_epochs), desc=f"训练轮数"):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向+反向+优化
                optimizer.zero_grad()  # 清零梯度
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数

                running_loss += loss.item()

        # 4. 评估模型（计算宏平均F1-score）
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():  # 关闭梯度，节省算力
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)  # 取概率最大的类别
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算宏平均F1（适配多分类，优先指标）
        f1 = f1_score(all_labels, all_preds, average='macro')
        f1_list.append(f1)
        print(f"重复实验 {repeat+1} F1-score（宏平均）：{f1:.4f}")

    return f1_list