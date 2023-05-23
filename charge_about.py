from PIL import Image
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score

# 加载图像并转化为 numpy 数组
img1 = Image.open('infrared_image.jpg').convert('L')
img2 = Image.open('fused_laplace_new.jpg').convert('L')
img1_array = np.array(img1).flatten()  # 将矩阵拉平为一维数组
img2_array = np.array(img2).flatten()  # 将矩阵拉平为一维数组

# 计算相关系数
corr, _ = pearsonr(img1_array, img2_array)
print(f'相关系数: {corr}')

# 计算互信息量
mutual_info = mutual_info_score(img1_array, img2_array)
print(f'互信息量: {mutual_info}')
