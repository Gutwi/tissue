import cv2
from matplotlib import pyplot as plt

# img = cv2.imread('cat.png',0)
img = cv2.imread('./my_dir/matigai/ref.jpg',0)

area = [3,15,31,63,127,255]

for i in range(6):
    plt.subplot(2,3,i+1)
    new_img =  cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,area[i],0)
    plt.imshow(new_img,'gray')
    plt.title("block size is {}".format(area[i]))
    plt.xticks([]),plt.yticks([])

plt.show()