import cv2
import numpy as np
import pywt

# 读取红外图像和可见光图像
infrared_img = cv2.imread('infrared_image.jpg', cv2.IMREAD_GRAYSCALE)
visible_img = cv2.imread('visible_image.jpg', cv2.IMREAD_GRAYSCALE)

# 对两个图像应用二维离散小波变换，得到近似系数和细节系数

wavelet = 'db1'                                     # Daubechies系列中的第一个小波，也被称为Haar小波
coeffs_infrared = pywt.dwt2(infrared_img, wavelet)  # 输出是一个元组，带有4个系数，1个低频，3个高频.
coeffs_visible = pywt.dwt2(visible_img, wavelet)

# 对两组系数进行融合
# 这里，我们选择使用红外图像的近似系数，可见光图像的细节系数
cA_infrared, (cH_infrared, cV_infrared, cD_infrared) = coeffs_infrared
cA_visible, (cH_visible, cV_visible, cD_visible) = coeffs_visible
cH_fused = cH_visible
cV_fused = cV_visible
cD_fused =cD_visible
# cH_fused = (cH_infrared + cH_visible) / 2
# cV_fused = (cV_infrared + cV_visible) / 2
# cD_fused = (cD_infrared + cD_visible) / 2

# 使用融合后的小波系数进行逆小波变换，得到融合后的图像
fused_img = pywt.idwt2((cA_infrared, (cH_fused, cV_fused, cD_fused)), wavelet)

# 显示融合后的图像
cv2.imshow('Fused Image', fused_img.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
# 将融合后的图像转为uint8类型，然后保存为jpg格式
fused_img_uint8 = fused_img.astype(np.uint8)
cv2.imwrite('fused_WT.jpg', fused_img_uint8)
