import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.widgets import Button, TextBox
from matplotlib.font_manager import FontProperties

# ===================== 核心：加载本地中文字体 =====================
FONT_PATH = 'C:/Windows/Fonts/simhei.ttf'  # Windows黑体路径
font = FontProperties(fname=FONT_PATH, size=10)
font_small = FontProperties(fname=FONT_PATH, size=9)  # 输入框标签小号字体

# ===================== 全局配置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# ===================== 配置项 =====================
IMG_FOLDER = "sampleset/img"  # 高光谱文件夹
MASK_FOLDER = "sampleset/masknpy"  # 掩码文件夹


# ===================== 核心查看器类 =====================
class DatasetNPYViewer:
    def __init__(self, img_folder, mask_folder):
        # 1. 加载文件
        self.img_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.npy')])
        self.mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.npy')])

        if len(self.img_files) == 0 or len(self.mask_files) == 0:
            raise ValueError("图像/掩码文件夹为空！")
        self.max_file_idx = min(len(self.img_files), len(self.mask_files)) - 1
        self.current_file_idx = 0
        self.current_band_idx = 0
        self.total_bands = 0

        # 2. 创建画布（调整整体布局，预留足够空间）
        self.fig, (self.ax_img, self.ax_mask) = plt.subplots(1, 2, figsize=(16, 8))
        # 关键：调整画布边距，底部留足控件空间
        self.fig.subplots_adjust(
            left=0.05, right=0.95, top=0.9, bottom=0.2,
            wspace=0.2, hspace=0.1
        )

        # 3. 颜色条对象
        self.cbar_img = None
        self.cbar_mask = None

        # 4. 创建交互控件（精准坐标，避免重叠/错位）
        # 4.1 文件切换按钮（左半区）
        ax_prev_file = plt.axes([0.10, 0.08, 0.13, 0.06])  # [左, 下, 宽, 高]
        ax_next_file = plt.axes([0.25, 0.08, 0.13, 0.06])
        self.btn_prev_file = Button(ax_prev_file, '上一个文件', color='#f0f0f0', hovercolor='#d0d0d0')
        self.btn_next_file = Button(ax_next_file, '下一个文件', color='#f0f0f0', hovercolor='#d0d0d0')
        self.btn_prev_file.label.set_fontproperties(font)
        self.btn_next_file.label.set_fontproperties(font)

        # 4.2 波段切换按钮（中半区）
        ax_prev_band = plt.axes([0.40, 0.08, 0.13, 0.06])
        ax_next_band = plt.axes([0.55, 0.08, 0.13, 0.06])
        self.btn_prev_band = Button(ax_prev_band, '上一个波段', color='#f0f0f0', hovercolor='#d0d0d0')
        self.btn_next_band = Button(ax_next_band, '下一个波段', color='#f0f0f0', hovercolor='#d0d0d0')
        self.btn_prev_band.label.set_fontproperties(font)
        self.btn_next_band.label.set_fontproperties(font)

        # 4.3 波段输入框（右半区，精准对齐）
        ax_band_label = plt.axes([0.70, 0.08, 0.08, 0.06])  # 单独的标签轴
        ax_band_input = plt.axes([0.78, 0.08, 0.10, 0.06])  # 输入框轴
        # 绘制标签（替代TextBox自带标签，避免错位）
        ax_band_label.text(0.5, 0.5, '跳转到波段：', ha='center', va='center', fontproperties=font_small)
        ax_band_label.axis('off')  # 隐藏标签轴
        # 创建无标签的输入框
        self.txt_band = TextBox(ax_band_input, '', initial='1')
        self.txt_band.on_submit(self.jump_to_band)

        # 5. 绑定事件
        self.btn_prev_file.on_clicked(self.on_prev_file)
        self.btn_next_file.on_clicked(self.on_next_file)
        self.btn_prev_band.on_clicked(self.on_prev_band)
        self.btn_next_band.on_clicked(self.on_next_band)

        # 6. 初始化
        self.load_current_file()
        self.update_display()

    def load_current_file(self):
        """加载当前文件，更新波段数"""
        img_path = os.path.join(IMG_FOLDER, self.img_files[self.current_file_idx])
        mask_path = os.path.join(MASK_FOLDER, self.mask_files[self.current_file_idx])
        self.current_img = np.load(img_path)
        self.current_mask = np.load(mask_path)
        self.current_filename = self.img_files[self.current_file_idx]
        self.total_bands = self.current_img.shape[0]
        self.current_band_idx = 0
        self.txt_band.set_val('1')

    def update_display(self):
        """更新显示（修复所有文字/位置问题）"""
        # 清除旧颜色条
        if self.cbar_img: self.cbar_img.remove()
        if self.cbar_mask: self.cbar_mask.remove()
        self.cbar_img = None
        self.cbar_mask = None

        # 清空子图
        self.ax_img.clear()
        self.ax_mask.clear()

        # 打印信息
        print("\n" + "=" * 70)
        print(f"当前文件：{self.current_filename}")
        print(
            f"文件进度：{self.current_file_idx + 1}/{self.max_file_idx + 1} | 波段进度：{self.current_band_idx + 1}/{self.total_bands}")
        print(
            f"高光谱形状：{self.current_img.shape} | 数值范围：{self.current_img.min():.2f}~{self.current_img.max():.2f}")
        print(
            f"掩码形状：{self.current_mask.shape} | 裂缝像素占比：{(self.current_mask == 1).sum() / self.current_mask.size * 100:.2f}%")

        # 显示高光谱（标题居中，字体适配）
        img_show = self.current_img[self.current_band_idx]
        im_img = self.ax_img.imshow(img_show, cmap='gray')
        img_title = f"高光谱图像 - 第{self.current_band_idx + 1}波段（共{self.total_bands}个）\n{self.current_filename[:20]}..."
        self.ax_img.set_title(img_title, fontproperties=font, fontsize=10, pad=10)
        self.ax_img.axis('off')
        self.cbar_img = self.fig.colorbar(im_img, ax=self.ax_img, shrink=0.9, pad=0.02)

        # 显示掩码
        im_mask = self.ax_mask.imshow(self.current_mask, cmap='gray')
        self.ax_mask.set_title("掩码图像（白色=裂缝）", fontproperties=font, fontsize=10, pad=10)
        self.ax_mask.axis('off')
        self.cbar_mask = self.fig.colorbar(im_mask, ax=self.ax_mask, shrink=0.9, pad=0.02)

        # 刷新画布
        self.fig.canvas.draw_idle()

    # ===================== 事件处理 =====================
    def on_prev_file(self, event):
        self.current_file_idx = (self.current_file_idx - 1) % (self.max_file_idx + 1)
        self.load_current_file()
        self.update_display()

    def on_next_file(self, event):
        self.current_file_idx = (self.current_file_idx + 1) % (self.max_file_idx + 1)
        self.load_current_file()
        self.update_display()

    def on_prev_band(self, event):
        self.current_band_idx = (self.current_band_idx - 1) % self.total_bands
        self.txt_band.set_val(str(self.current_band_idx + 1))
        self.update_display()

    def on_next_band(self, event):
        self.current_band_idx = (self.current_band_idx + 1) % self.total_bands
        self.txt_band.set_val(str(self.current_band_idx + 1))
        self.update_display()

    def jump_to_band(self, text):
        try:
            band_num = int(text)
            if 1 <= band_num <= self.total_bands:
                self.current_band_idx = band_num - 1
                self.update_display()
            else:
                print(f"波段号需在1~{self.total_bands}之间！")
                self.txt_band.set_val(str(self.current_band_idx + 1))
        except ValueError:
            print("请输入有效数字！")
            self.txt_band.set_val(str(self.current_band_idx + 1))


# ===================== 启动 =====================
if __name__ == "__main__":
    try:
        viewer = DatasetNPYViewer(IMG_FOLDER, MASK_FOLDER)
        plt.show()
    except Exception as e:
        print(f"错误：{e}")