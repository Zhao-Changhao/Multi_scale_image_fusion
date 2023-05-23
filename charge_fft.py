import numpy as np
import cv2

# 加载图像（灰度模式）
img = cv2.imread('fused_WT.jpg',0)

# 对图像进行傅立叶变换
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# 计算图像的振幅谱
magnitude_spectrum = 20*np.log(np.abs(fshift))

# 创建一个掩模，只保留低频分量
rows, cols = img.shape
crow,ccol = rows//2 , cols//2
mask_low = np.zeros((rows,cols),np.uint8)
mask_low[crow-30:crow+30, ccol-30:ccol+30] = 1

# 创建一个掩模，只保留高频分量
mask_high = np.ones((rows,cols),np.uint8)
mask_high[crow-30:crow+30, ccol-30:ccol+30] = 0

# 计算低频和高频分量的平均值
low_freq_avg = np.mean(magnitude_spectrum[mask_low==1])
high_freq_avg = np.mean(magnitude_spectrum[mask_high==1])

print("Low frequency average: ", low_freq_avg)
print("High frequency average: ", high_freq_avg)
