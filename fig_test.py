import os
import cv2
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(3,2,1)
ax.set_title('Reference')
# plt.imshow(reference[:,:,::-1])
ax = fig.add_subplot(3,2,2)
# target_name=os.path.basename(tar_file)  #250221
ax.set_title('Target:')
# plt.imshow(target[:,:,::-1])
# plt.show()	#一旦表示するとfigがリセットされる

ax = fig.add_subplot(3,2,3)
ax.set_title('Target2:')
ax = fig.add_subplot(3,2,4)
ax.set_title('Target3:')
ax = fig.add_subplot(3,2,5)
ax.set_title('Target4:')
ax = fig.add_subplot(3,2,6)

plt.show()