import cv2
import numpy as np
import pywt

def reconstruct_all_levels(coeffs, wavelet):
    # 逐层还原图像
    reconstructed_levels = []
    for i in range(len(coeffs)):
        reconstructed_image = pywt.waverec2(coeffs[:i+1], wavelet)
        reconstructed_levels.append(reconstructed_image)
    return reconstructed_levels

# 读取红外图像和可见光图像
ir_image = cv2.imread('ir1.jpg')
visible_image = cv2.imread('vi1.jpg')

# 将红外图像和可见光图像转换为灰度图像
ir_gray = cv2.cvtColor(ir_image, cv2.COLOR_BGR2GRAY)
visible_gray = cv2.cvtColor(visible_image, cv2.COLOR_BGR2GRAY)

# 对红外图像和可见光图像进行小波变换
wavelet = 'db4'
ir_coeffs = pywt.wavedec2(ir_gray, wavelet, level=4)
visible_coeffs = pywt.wavedec2(visible_gray, wavelet, level=4)

# 还原每一层的图像
reconstructed_ir_levels = reconstruct_all_levels(ir_coeffs, wavelet)
reconstructed_visible_levels = reconstruct_all_levels(visible_coeffs, wavelet)

# 保存每一层的图像
for i, image in enumerate(reconstructed_ir_levels):
    cv2.imwrite(f'reconstructed_ir_level_{i+1}.png', image)

for i, image in enumerate(reconstructed_visible_levels):
    cv2.imwrite(f'reconstructed_visible_level_{i+1}.png', image)
