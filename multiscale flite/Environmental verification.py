import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
import cv2
from PIL import Image

# 验证PyTorch
print(f"PyTorch版本: {torch.__version__}")
print(f"是否支持GPU（A卡需适配ROCm，无则显示False）: {torch.cuda.is_available()}")
# 即使显示False，CPU也能运行（单波长训练仅需几分钟）