import torch
import torch.nn as nn
import torch.nn.functional as F

##自注意力特征提取模块，核心是3D 空洞卷积提取多尺度特征+类自注意力机制融合+残差连接。
class dilateattention(nn.Module):
    def __init__(self, in_channel, depth):
        super(dilateattention, self).__init__()# 继承nn.Module并初始化

        # 初始化层：BatchNorm3d(3D批归一化)、ReLU(激活函数)
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        # self.mean = nn.AdaptiveAvgPool2d((1, 1))
        # k=1 s=1 no pad
        self.batch_norm = nn.BatchNorm3d(in_channel)  # 3D批归一化，防止过拟合，加速训练
        self.relu = nn.ReLU() # 非线性激活，让模型学习复杂特征
        # 3个不同空洞率的3D卷积，提取多尺度特征（对应论文d1=0, d2=2, d3=4）
        self.atrous_block1 = nn.Conv3d(in_channel, depth, 1, 1)
        self.atrous_block2 = nn.Conv3d(in_channel, depth, 3, 1, padding=2, dilation=2, groups=in_channel)
        self.atrous_block3 = nn.Conv3d(in_channel, depth, 3, 1, padding=4, dilation=4, groups=in_channel)
        #1x1卷积
        self.conv_1x1_output = nn.Conv3d(depth * 5, depth, 1, 1)
        self.softmax = nn.Softmax()

    def forward(self, x): # 前向传播：输入x是特征图，输出融合后的多尺度特征
        # 步骤1：用3个空洞卷积提取多尺度特征，对应论文的V,Q,K
        v = self.atrous_block1(x)  # V：1x1卷积特征（基础特征）
        q = self.atrous_block2(x)  # Q：空洞率2卷积特征（中尺度）
        k = self.atrous_block3(x)  # K：空洞率4卷积特征（大尺度）
        # 步骤2：类自注意力计算（论文公式4：Softmax(QK^T/√d)V）
        temp = k * q  # 简化版QK^T   特征交互
        # dk = torch.std(temp)
        output = v * self.softmax(temp) # V乘以注意力权重，实现特征加权融合
        # 步骤3：残差连接（论文MSSA的残差结构）：原始特征经BN+ReLU后和融合特征相加，防止梯度消失
        output += self.relu(self.batch_norm(x))
        return output

##MS-CrackSeg 的整体网络，包含编码器（3D 卷积 + 池化 + MSSA） +解码器（转置卷积上采样 + 特征拼接）
class mymodel(nn.Module):
    def __init__(self, in_channel,classnum=1): # 输入：in_channel(高光谱图像通道数)、classnum(分割类别数，裂缝分割为1)
        super(mymodel, self).__init__()
        self.dim = 8  # 网络特征图的基础通道数，可理解为"特征维度"，越小计算量越小
        # ===================== 编码器（Encoder）：3个Block =====================

        # 编码器Block1：3D卷积→3D最大池化→MSSA模块
        self.conv3d1 = nn.Conv3d(in_channel, self.dim, kernel_size=(7,7,7), padding=3,stride=1)
        self.maxpooling1 = nn.MaxPool3d(kernel_size=2,stride=2)
        self.dialatiaon1 = dilateattention(self.dim,self.dim)
        # 编码器Block2：和Block1结构一致，特征通道数保持8
        self.conv3d2 = nn.Conv3d(self.dim, self.dim, kernel_size=(7, 7, 7), padding=3, stride=1)
        self.maxpooling2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dialatiaon2 = dilateattention(self.dim, self.dim)
        # 编码器Block3：和Block1/2结构一致
        self.conv3d3 = nn.Conv3d(self.dim, self.dim, kernel_size=(7, 7, 7), padding=3, stride=1)
        self.maxpooling3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dialatiaon3 = dilateattention(self.dim, self.dim)

        # ===================== 解码器（Decoder）：3个转置卷积+特征拼接，对应论文2.3节 =====================

        self.transpose1 = nn.ConvTranspose3d(self.dim,self.dim,kernel_size=2,stride=2)
        self.conv2d1 = nn.Conv3d(self.dim,self.dim, kernel_size=(7,7,7),padding=(3,3,3),stride=1)
        self.transpose2 = nn.ConvTranspose3d(self.dim * 2, self.dim, kernel_size=2, stride=2)
        self.conv2d2 = nn.Conv3d(self.dim, self.dim, kernel_size=7, padding=3, stride=1)
        self.transpose3 = nn.ConvTranspose3d(self.dim * 2, self.dim, kernel_size=2, stride=2)
        self.conv2d3 = nn.Conv3d(self.dim, 1, kernel_size=7, padding=3, stride=1)

        # 最终2D卷积：将光谱维度的特征压缩为2D，输出分割结果（像素级分类）
        self.final = nn.Conv2d(152, classnum, 3, 1, 1)
        # 152是高光谱图像的光谱通道数（论文中176波段，作者做了裁剪）

#forward 前向传播
    def forward(self,x): # 输入x：高光谱图像，形状为[批次数, 光谱通道数, 高度, 宽度]
        # ===================== 编码器前向：逐层计算，得到3个尺度的特征x1/x2/x3 =====================
        # Block1：3D卷积
        x1 = self.conv3d1(x)
        x1 = self.maxpooling1(x1) # 下采样2倍
        x1 = self.dialatiaon1(x1) # MSSA融合多尺度特征
        # Block2：基于x1计算，再次下采样2倍
        x2 = self.conv3d2(x1)
        x2 = self.maxpooling2(x2)
        x2 = self.dialatiaon2(x2)
        # Block3：基于x2计算，第三次下采样2倍（最终x3是原始尺寸的1/8）
        x3 = self.conv3d3(x2)
        x3 = self.maxpooling3(x3)
        x3 = self.dialatiaon3(x3)

        # ===================== 解码器前向：上采样+特征拼接，逐步恢复尺寸 =====================
        # 解码器第一步：x3上采样2倍 → 卷积 → 和x2拼接（通道数8→16）
        x4 = self.transpose1(x3) # 上采样2倍，尺寸变为1/4
        x4 = self.conv2d1(x4) # 3D卷积融合特征
        x4 = torch.cat([x4, x2], dim=1) # 特征拼接，dim=1表示在通道维度拼接（8+8=16）
        # 解码器第二步：x4上采样2倍 → 卷积 → 和x1拼接（通道数16→8→16）
        x5 = self.transpose2(x4)  # 上采样2倍，尺寸变为1/2
        x5 = self.conv2d2(x5)  # 3D卷积降维为8
        x6 = torch.cat([x5,x1], dim=1) # 和x1拼接，通道数再次变为16
        # 解码器第三步：x6上采样2倍 → 卷积 → 输出1通道特征（尺寸恢复为原始）
        x6 = self.transpose3(x6) # 上采样2倍，尺寸恢复为原始图像
        x6 = self.conv2d3(x6) # 3D卷积，输出1通道特征图
        # ===================== 最终处理：转为2D，输出分割结果 =====================
        x6 = torch.squeeze(x6,dim=1) # 挤压掉通道维度（1→无），变为3D张量[批次, 光谱, 高, 宽]
        x6 = self.final(x6) # 2D卷积，融合光谱特征，输出[批次, 1, 高, 宽]（像素级分割结果）
        return x6 # 输出分割结果：每个像素值表示"是否是裂缝"（后续用sigmoid归一化到0-1）






