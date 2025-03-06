import os
import cv2
import matplotlib.pyplot as plt


# fig = plt.figure()
# ax = fig.add_subplot(3,2,1)
# ax.set_title('Reference')
# # plt.imshow(reference[:,:,::-1])
# ax = fig.add_subplot(3,2,2)
# # target_name=os.path.basename(tar_file)  #250221
# ax.set_title('Target:')
# # plt.imshow(target[:,:,::-1])
# # plt.show()	#一旦表示するとfigがリセットされる

# ax = fig.add_subplot(3,2,3)
# ax.set_title('Target2:')
# ax = fig.add_subplot(3,2,4)
# ax.set_title('Target3:')
# ax = fig.add_subplot(3,2,5)
# ax.set_title('Target4:')
# ax = fig.add_subplot(3,2,6)

# plt.show()



# 画像を読み込む (cv2: BGR形式)
img_cv2 = cv2.imread('test_bad1.jpg')  # OpenCV用
img_plt = plt.imread('test_good1.jpg')  # matplotlib用 (RGB形式)

# OpenCVの画像をRGBに変換
# img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

# 画像を並べて表示
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# ax[0].imshow(img_cv2_rgb)
ax[0].imshow(img_cv2)

ax[0].set_title("OpenCV Image")
ax[0].axis("off")  # 軸を非表示

ax[1].imshow(img_plt)
ax[1].set_title("Matplotlib Image")
ax[1].axis("off")

plt.show()
