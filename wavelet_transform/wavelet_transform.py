import cv2
import numpy as np
import pywt

def multiscale_fusion(ir_image, visible_image):
    # 将红外图像和可见光图像转换为灰度图像
    ir_gray = cv2.cvtColor(ir_image, cv2.COLOR_BGR2GRAY)
    visible_gray = cv2.cvtColor(visible_image, cv2.COLOR_BGR2GRAY)

    # 对红外图像和可见光图像进行小波变换
    ir_coeffs = pywt.wavedec2(ir_gray, 'db4', level=4)
    visible_coeffs = pywt.wavedec2(visible_gray, 'db4', level=4)

    # 定义加权融合策略，保留红外图像的较低频率成分，保留可见光图像的较高频率成分
    weights = [1, 0.8, 0.5, 0.3, 0.2]

    # 对各个尺度的系数进行融合
    fused_coeffs = []
    for ir_coeff, visible_coeff, weight in zip(ir_coeffs, visible_coeffs, weights):
        # 使用加权融合规则
        fused_coeff = tuple((ir * weight + visible * (1 - weight)) for ir, visible in zip(ir_coeff, visible_coeff))
        fused_coeffs.append(fused_coeff)

    # 重构融合后的图像
    fused_image = pywt.waverec2(fused_coeffs, 'db4')

    # 将图像像素值缩放到0-255之间
    fused_image = cv2.normalize(fused_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return fused_image

# 读取红外图像和可见光图像
ir_image = cv2.imread('ir1.jpg')
visible_image = cv2.imread('vi1.jpg')

# 进行多尺度融合
fused_image = multiscale_fusion(ir_image, visible_image)

# 保存融合结果
cv2.imwrite('fused_image33.jpg', fused_image)

# 显示融合结果
cv2.imshow('Fused Image', fused_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
