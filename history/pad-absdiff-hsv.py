import cv2
import numpy as np


# --------------------------------------------------- #
# 画像合成                                             #
# --------------------------------------------------- #

def FitImageSize_small(img1, img2):
    # height
    if img1.shape[0] > img2.shape[0]:
        height = img2.shape[0]
        width = img1.shape[1]
        img1 = cv2.resize(img1, (width, height))
    else:
        height = img1.shape[0]
        width = img2.shape[1]
        img2 = cv2.resize(img2, (width, height))

    # width
    if img1.shape[1] > img2.shape[1]:
        height = img1.shape[0]
        width = img2.shape[1]
        img1 = cv2.resize(img1, (width, height))
    else:
        height = img2.shape[0]
        width = img1.shape[1]
        img2 = cv2.resize(img2, (width, height))
    return img1, img2


path_img ="./my_dir/matigai/ref.jpg"    #250301
img = cv2.imread(path_img)

if img is None:
    print('ファイルを読み込めません')
    import sys

    sys.exit()

cv2.imshow("img", img)

# 余白を取り除いたときに2つの画像が最も一致するような適切な余白（padding）の幅を見つける
img_src = img
padding_result = []
for padding in range(0, 50):
    # 画像の余白を削除
    # (余白無しの可能性を考慮)
    if padding:
        img = img_src[:, padding:-padding]

    # 画像を左右で分割する
    height, width, channels = img.shape[:3]
    img1 = img[:, :width // 2]
    img2 = img[:, width // 2:]

    # 画像サイズを合わせる(小さい方に)
    img1, img2 = FitImageSize_small(img1, img2)

    # 2つの画像の差分を算出
    img_diff = cv2.absdiff(img2, img1)
    img_diff_sum = np.sum(img_diff)

    padding_result.append((img_diff_sum, padding))

# 差分が最も少ないものを選ぶ
_, padding = min(padding_result, key=lambda x: x[0])

# 画像の余白を削除
if padding:
    img = img_src[:, padding:-padding]

# 画像を左右で分割する
height, width, channels = img.shape[:3]
img1 = img[:, :width // 2]
img2 = img[:, width // 2:]
cv2.imshow("img2", img2)

# もともとの対象が印刷物なので、HSVに変換してみる。
img1_hsv = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
img2_hsv = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)

# 2つの画像の差分を算出
img_diff = cv2.absdiff(img2_hsv, img1_hsv)
diff_h = img_diff[:, :, 0]
diff_s = img_diff[:, :, 1] * 3
diff_v = img_diff[:, :, 2]

# 一定未満の彩度(V)の部分は色合い(H)の差分を考慮しないようにする
H_THRESHOLD = 70
_, diff_h_mask = cv2.threshold(diff_v, H_THRESHOLD, 255, cv2.THRESH_BINARY)
diff_h = np.minimum(diff_h, diff_h_mask)

# 輝度、彩度の差分を規格化する
diff_s = cv2.normalize(diff_s, _, 255, 0, cv2.NORM_MINMAX)
diff_v = cv2.normalize(diff_v, _, 255, 0, cv2.NORM_MINMAX)

# HSVの差分の一番大きく変わったところを取得する
diff_glay = np.maximum(diff_h, diff_s, diff_v)

# 変わっている場所の候補を二値化とオープニングで取得
DIFF_THRESHOLD = 60
_, diff_bin = cv2.threshold(diff_glay, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

# 画像サイズを合わせる(小さい方に)
img2, diff_bin = FitImageSize_small(img2, diff_bin)
cv2.imshow("img_diff", diff_bin)

# 画像合成
diff_bin_rgb = cv2.cvtColor(diff_bin, cv2.COLOR_GRAY2RGB)
add = np.maximum(np.minimum(img2, diff_bin_rgb), (img2 // 3))
cv2.imshow("add", add)

cv2.waitKey(0)
cv2.destroyAllWindows()

