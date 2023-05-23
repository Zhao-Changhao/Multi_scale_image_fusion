from PIL import Image
import numpy as np

# 加载图片
img = Image.open('fused_laplace_new.jpg').convert('L')  # 转换为灰度图像
img_array = np.array(img)

# 计算灰度均值
mean = np.mean(img_array)
print(f'灰度均值: {mean}')

# 计算灰度标准差
std_dev = np.std(img_array)
print(f'灰度标准差: {std_dev}')

# 计算熵
histogram = np.histogram(img_array, bins=256, range=(0,256))[0]
histogram = histogram / histogram.sum()  # 归一化
entropy = -np.sum(histogram*np.log2(histogram + np.finfo(float).eps))  # 为了防止对0取log，加上一个很小的值
print(f'熵: {entropy}')

# 保存结果到 txt 文件
with open('result_fused,WT.txt', 'w') as f:
    f.write('WT方法处理后图像的评价:\n')
    f.write(f'灰度均值: {mean}\n')
    f.write(f'灰度标准差: {std_dev}\n')
    f.write(f'熵: {entropy}\n')
