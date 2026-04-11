# ===================== 【训练代码的末尾！直接粘贴这部分！】 =====================
# 👇 这是你已经跑完20轮的训练代码，在最后面加下面的可视化代码
print("🎉 训练完成，开始可视化预测结果！")

# 切换模型为评估模式
model.eval()

# 取测试集第一张图片
with torch.no_grad():
    # 从测试集取一个样本
    test_img, test_mask = next(iter(test_loader))
    test_img = test_img.to(device)
    test_mask = test_mask.to(device)

    # 模型预测
    pred = model(test_img)
    # 🔥 修复维度不匹配：压缩所有多余维度，和掩码完全一致
    pred = torch.sigmoid(pred).squeeze()
    pred_mask = (pred > 0.5).cpu().numpy()

# 准备显示数据
img_display = test_img[0, 0, 88].cpu().numpy()  # 取高光谱第88个波段
gt_mask = test_mask[0].cpu().numpy()

# 绘图（三图对比）
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_display, cmap='gray')
plt.title('高光谱输入图像 (波段88)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gt_mask, cmap='gray')
plt.title('真实裂缝标签')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(pred_mask, cmap='gray')
plt.title('模型预测结果')
plt.axis('off')

plt.tight_layout()
plt.show()