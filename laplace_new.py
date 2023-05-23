import cv2
import numpy as np

# 读取红外图像和可见光图像
infrared_img = cv2.imread('infrared_image.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
visible_img = cv2.imread('visible_image.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)

# 创建高斯金字塔
level = 5  # 设置金字塔的层数
gp_infrared = [infrared_img.copy()]
gp_visible = [visible_img.copy()]
for i in range(level):
    infrared_img = cv2.pyrDown(infrared_img)  # 进行下采样
    visible_img = cv2.pyrDown(visible_img)
    gp_infrared.append(infrared_img)
    gp_visible.append(visible_img)

# 创建拉普拉斯金字塔
# 拉普拉斯金字塔向上每层分辨率逐渐提高
# 拉普拉斯金字塔是通过对高斯金字塔的每个层进行上采样并从较低分辨率版本中减去相应的高分辨率版本得到的。
lp_infrared = [gp_infrared[-1]]
lp_visible = [gp_visible[-1]]
for i in range(level, 0, -1):
    infrared_lap = cv2.subtract(gp_infrared[i-1], cv2.pyrUp(gp_infrared[i]))    # 进行上采样，对于每个高斯金字塔层，使用 cv2.pyrUp 函数将其上采样到较大的尺寸
    visible_lap = cv2.subtract(gp_visible[i-1], cv2.pyrUp(gp_visible[i]))       # 并通过减去较低分辨率版本来计算拉普拉斯金字塔的每一层。
    lp_infrared.append(infrared_lap)
    lp_visible.append(visible_lap)

# 拉普拉斯金字塔融合
lp_fused = []
for i in range(level+1):
    if i == level:  # 对于高斯金字塔的最底层（红外图像的低频部分），直接使用红外图像
        fused_lap = lp_infrared[i]
    else:  # 对于其他层（可见光图像的高频部分），使用可见光图像
        fused_lap = lp_visible[i]
    lp_fused.append(fused_lap)

# 使用融合后的拉普拉斯金字塔进行重建，得到融合后的图像
fused_img = lp_fused[0]
for i in range(1, level+1):
    fused_img = cv2.pyrUp(fused_img)
    rows, cols = lp_fused[i].shape
    fused_img = cv2.add(fused_img[:rows, :cols], lp_fused[i])

# 显示融合后的图像
cv2.imshow('Fused Image', fused_img.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存融合后的图像
cv2.imwrite('fused_laplace_new.jpg', fused_img.astype(np.uint8))
