import cv2
from matplotlib import pyplot as plt

# 入力画像1を読み込む
image1 = cv2.imread("./my_dir/matigai/ref.jpg")

# 入力画像2を読み込む
image2 = cv2.imread("./my_dir/matigai/tar_bad_easy.jpg")

# 画像のサイズが同じであることを確認
if image1.shape != image2.shape:
    print("画像のサイズが異なります。リサイズが必要です。")
    exit

# BGRのチャンネル並びをRGBの並びに変更(matplotlibで結果を表示するため)
rgb_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB) 
rgb_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB) 

GAUB = 11
# rgb_image1 = cv2.GaussianBlur(rgb_image1, (GAUB, GAUB), 0)
# rgb_image2 = cv2.GaussianBlur(rgb_image2, (GAUB, GAUB), 0)

# 差分を計算
diff = cv2.absdiff(rgb_image1, rgb_image2)
# diff = cv2.subtract(rgb_image1, rgb_image2)
diff = cv2.GaussianBlur(diff, (GAUB, GAUB), 0)  #250219 ぼかして誤検出を減らしたい

# 応用と使い分け
# cv2.subtract関数
#   背景除去：静的な背景画像から現在のフレームを引く場合
#   画像の明るさ調整：一定値を引くことで画像全体を暗くする
#   マスク処理：特定の領域を除外する際に使用
# cv2.absdiff関数
#   動体検出：連続するフレーム間の差を検出する場合
#   エッジ検出：画像のシャープな変化を検出する際に使用
#   画像の類似性評価：2つの画像がどの程度異なるかを測定する場合。

THRE_V=64

# グレースケールに変換
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
# 二値化で異なる部分を強調、背景が白になる様にcv2.THRESH_BINARY_INVを指定
# retval, result = cv2.threshold(diff_gray, THRE_V, 255, cv2.THRESH_BINARY_INV)   #250219 何故か_INVなかった
retval, result = cv2.threshold(diff_gray, THRE_V, 255, cv2.THRESH_OTSU)   #250219

# 結果の可視化
plt.rcParams["figure.figsize"] = [18,7]                             # ウィンドウサイズを設定
title = "cv2.absdiff: codevace.com"
plt.figure(title)                                                   # ウィンドウタイトルを設定
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.03, top=0.95)   # 余白を設定
plt.subplot(131)                                                    # 1行3列の1番目の領域にプロットを設定
plt.imshow(rgb_image1)                                              # 入力画像1を表示
plt.title('image1')                                                 # 画像タイトル設定
plt.axis("off")                                                     # 軸目盛、軸ラベルを消す
plt.subplot(132)                                                    # 1行3列の2番目の領域にプロットを設定
plt.imshow(rgb_image2)                                              # 入力画像2を表示
plt.title('image2')                                                 # 画像タイトル設定
plt.axis("off")                                                     # 軸目盛、軸ラベルを消す
plt.subplot(133)                                                    # 1行3列の3番目の領域にプロットを設定
plt.imshow(result, cmap='gray')                                     # 差分の画像を表示
plt.title('subrtacted')                                             # 画像タイトル設定
plt.axis("off")                                                     # 軸目盛、軸ラベルを消す
plt.show()