import cv2
import numpy as np

# 读取红外图像和可见光图像
infrared_img = cv2.imread('ir1.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
visible_img = cv2.imread('vi1.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)

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
lp_infrared = [gp_infrared[-1]]
lp_visible = [gp_visible[-1]]
for i in range(level, 0, -1):
    infrared_lap = cv2.subtract(gp_infrared[i-1], cv2.pyrUp(gp_infrared[i]))    # 进行上采样
    visible_lap = cv2.subtract(gp_visible[i-1], cv2.pyrUp(gp_visible[i]))
    lp_infrared.append(infrared_lap)
    lp_visible.append(visible_lap)

# 保存高斯金字塔的每一层
for i in range(level+1):
    cv2.imwrite(f'gaussian_pyramid_infrared_level{i}.jpg', gp_infrared[i].astype(np.uint8))
    cv2.imwrite(f'gaussian_pyramid_visible_level{i}.jpg', gp_visible[i].astype(np.uint8))

# 保存拉普拉斯金字塔的每一层
for i in range(level+1):
    cv2.imwrite(f'laplacian_pyramid_infrared_level{i}.jpg', lp_infrared[i].astype(np.uint8))
    cv2.imwrite(f'laplacian_pyramid_visible_level{i}.jpg', lp_visible[i].astype(np.uint8))
