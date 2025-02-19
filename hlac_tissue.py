import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from concurrent.futures import ThreadPoolExecutor

# cv2.imread('./my_dir/matigai/ref.jpg')
# plt.imshow(image)
# plt.show()

"""OpenCVでの画像読み込みはBGRチャネルの順番であり、MatplotlibはRGBの順番で表示されるため、色は少し変になります。これを正しく表示するためにはOpenCVで色空間を変更してもいいですが、下記のようにチャネルを逆読み込みするのが楽です。"""
# plt.imshow(image[:,:,::-1])
# plt.show()

# r,c = image.shape[:2]
reference = cv2.imread('./my_dir/matigai/ref.jpg')
target = cv2.imread('./my_dir/matigai/tar_good_easy.jpg')
# cv2.imwrite('./my_dir/matigai/reference.png', reference)    #250219
# cv2.imwrite('./my_dir/matigai/target.png', target)          #250219
# 可視化
fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.set_title('Reference')
plt.imshow(reference[:,:,::-1])
ax = fig.add_subplot(1,2,2)
ax.set_title('Target')
plt.imshow(target[:,:,::-1])
plt.show()

"""# OpenCVで間違い探し

OpenCVでやるならばReferenceとTargetの差分画像から検出するのが最も簡単でしょう。
差分画像は以下のように計算できます。
"""

reference_g = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) # グレイスケールに変換
target_g = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
diff = cv2.absdiff(reference_g, target_g) # 差分画像を作成
ret, diff = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # 差分画像を二値化
diff = cv2.GaussianBlur(diff, (11, 11), 0)
# ぼかして小さな誤検出を減らす default: 11
# plt.imshow(diff)
# plt.show()

# """差分画像から検出枠を描くとわかりやすいです。"""
dst = np.copy(target)
dst = np.zeros_like(target)
contours, hierarchy = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 白い領域をグルーピング
# 検出位置に枠を描く
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 127, 0), 3)
dst = cv2.addWeighted(target, 0.2, dst, 0.8, 1.0)
# plt.imshow(dst[:, :, ::-1])
# plt.show()

# # HLACで間違い探し
# では同様のことを2値HLACを使ってやってみましょう！  
# まずはHLAC用の2値マスクを用意するところですが、今回は以下にまとめました。
# """
hlac_filters =  [np.array([[False, False, False], [False,  True, False], [False, False, False]]),  np.array([[False, False, False], [False,  True,  True], [False, False, False]]),  np.array([[False, False,  True], [False,  True, False], [False, False, False]]),  np.array([[False,  True, False], [False,  True, False], [False, False, False]]),  np.array([[ True, False, False], [False,  True, False], [False, False, False]]),  np.array([[False, False, False], [ True,  True,  True], [False, False, False]]),  np.array([[False, False,  True], [False,  True, False], [ True, False, False]]),  np.array([[False,  True, False], [False,  True, False], [False,  True, False]]),  np.array([[ True, False, False], [False,  True, False], [False, False,  True]]),  np.array([[False, False,  True], [ True,  True, False], [False, False, False]]),  np.array([[False,  True, False], [False,  True, False], [ True, False, False]]),  np.array([[ True, False, False], [False,  True, False], [False,  True, False]]),  np.array([[False, False, False], [ True,  True, False], [False, False,  True]]),  np.array([[False, False, False], [False,  True,  True], [ True, False, False]]),  np.array([[False, False,  True], [False,  True, False], [False,  True, False]]),  np.array([[False,  True, False], [False,  True, False], [False, False,  True]]),  np.array([[ True, False, False], [False,  True,  True], [False, False, False]]),  np.array([[False,  True, False], [ True,  True, False], [False, False, False]]),  np.array([[ True, False, False], [False,  True, False], [ True, False, False]]),  np.array([[False, False, False], [ True,  True, False], [False,  True, False]]),  np.array([[False, False, False], [False,  True, False], [ True, False,  True]]),  np.array([[False, False, False], [False,  True,  True], [False,  True, False]]),  np.array([[False, False,  True], [False,  True, False], [False, False,  True]]),  np.array([[False,  True, False], [False,  True,  True], [False, False, False]]),  np.array([[ True, False,  True], [False,  True, False], [False, False, False]])]

# """2値画像を処理する必要があるので、カラー画像を一旦グレイスケールにした後に二値化してみましょう。"""

# reference_bin = cv2.threshold(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] == 255
# target_bin = cv2.threshold(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] == 255
# reference_bin = cv2.threshold(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU)[1] == 255
# target_bin = cv2.threshold(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU)[1] == 255

#adaptive250219
BINA_TH = 63
reference_bin =  cv2.adaptiveThreshold(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,BINA_TH,0)
target_bin =  cv2.adaptiveThreshold(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,BINA_TH,0)
# ADAPTIVE_THRESH_GAUSSIAN_C
plt.imshow(reference_bin)
plt.show()

# """HLAC特徴量の計算は2次元平面における自己相関なので、マスクをラスタースキャンするような形の積和で計算できます。  
# つまり2次元の畳み込みで計算できます。ちょっと特殊なのは、二値HLACなので出力がTrue/Falseになるよう、マスク畳み込みの後でマスクとの形状一致を評価する必要がある点に注意してください。
# """

def extract_hlac(image, hlac_filters):
    result = []
    image = np.uint8(image)
    hlac_filters = np.uint8(hlac_filters)
    for filter in hlac_filters:
        feature_map = signal.convolve2d(image, filter, mode='valid')
        count = np.sum(feature_map == np.sum(filter))
        result.append(count)
    return result

# """ではさっそくHLAC特徴量を計算してみましょう。"""
# reference_vec = extract_hlac(reference_bin, hlac_filters)
# target_vec = extract_hlac(target_bin, hlac_filters)
# plt.plot(reference_vec, label='Reference')
# plt.plot(target_vec, label='Target')
# plt.legend()
# plt.show()

# """このように、各特徴次元ごとに微妙な違いがあります。  
# 画像に対してまとめてHLAC特徴を計算したので大した違いはありませんが、小さなパッチ単位で計算すれば更に大きな違いが見えてくるはずです。
# """

# # パッチ版HLAC特徴量
def split_into_batches(image, nx, ny):
    batches = []
    for y_batches in np.array_split(image, ny, axis=0):
        for x_batches in np.array_split(y_batches, nx, axis=1):
            batches.append(x_batches)
    return batches

def extract_batchwise_hlac(image, hlac_filters, nx, ny):
    batches = split_into_batches(np.uint8(image), nx, ny)
    hlac_filters = np.uint8(hlac_filters)
    hlac_batches = []
    extracter = lambda args: np.sum(signal.convolve2d(args[0], args[1], mode='valid') == np.sum(args[1]))
    with ThreadPoolExecutor(max_workers=int(os.cpu_count() / 2)) as e:
        for batch in batches:
            result = list(e.map(extracter, zip([batch] * len(hlac_filters), hlac_filters)))
            hlac_batches.append(result)
    return np.array(hlac_batches)

# """余談ですが、concurrent.futures のThreadPoolExecutorやProcessPoolExecutorは簡単に並行・並列処理化できるのでおすすめです。  
# とりあえず縦横20個、合計400のパッチに分けて計算してみます。
# """

nx, ny = 20, 20
reference_hlac = extract_batchwise_hlac(reference_bin, hlac_filters, nx, ny)
target_hlac = extract_batchwise_hlac(target_bin, hlac_filters, nx, ny)
print(f'Reference:\n {reference_hlac}\n')
print(f'Target:\n {target_hlac}\n')

#可視化
fig = plt.figure()
ax = fig.add_subplot(1,3,1)
ax.set_title('Reference')
plt.imshow(reference_hlac, aspect='auto', cmap='gray')
ax = fig.add_subplot(1,3,2)
ax.set_title('Target')
plt.imshow(target_hlac, aspect='auto', cmap='gray')
ax = fig.add_subplot(1,3,3)
ax.set_title('Difference')
plt.imshow(target_hlac-reference_hlac, aspect='auto', cmap='gray')
fig.tight_layout()
plt.show()

# """上記の可視化は特徴ベクトルを並べて可視化したもので、1行が1パッチのHLAC特徴量に対応しています。  
# このままでは特徴量がどの程度違うのか評価できないので、ReferenceとTargetのHLAC特徴量の内積から角度を計算してみます。
# """

def vector_angle(hv1, hv2, eps = 1e-7): #default 1e-6
    hv1 = (hv1 + eps) / np.linalg.norm(hv1 + eps)
    hv2 = (hv2 + eps) / np.linalg.norm(hv2 + eps)
    return np.arccos(np.clip(np.dot(hv1, hv2), -1.0, 1.0))

# """上記の関数を用いてパッチごとの角度をプロットすると"""

hlac_angles = [vector_angle(rv, tv) for rv, tv in zip(reference_hlac, target_hlac)]
plt.plot(hlac_angles)
plt.show()

# """このようにパッチ単位の差をきれいに可視化できます。  
# 最後に、この情報を元に異常部位を可視化してみましょう！
# """

def visualize(image, hlac_angles, nx, ny, th=0.1):
    batches = split_into_batches(image, nx, ny)
    dst = np.zeros_like(image)
    hlac_angles -= np.nanmin(hlac_angles)
    hlac_angles /= np.nanmax(hlac_angles)
    py = 0
    for y in range(ny):
        px = 0
        for x in range(nx):
            batch = batches[y * nx + x]
            angle = hlac_angles[y * nx + x]
            if angle > th:
                dst = cv2.rectangle(dst, (px, py), (px + batch.shape[1], py + batch.shape[0]), (0, int(255 * angle), 0), -1)
                dst = cv2.rectangle(dst, (px, py), (px + batch.shape[1], py + batch.shape[0]), (0, 255, 0), 1)
            px += batch.shape[1]
        py += batch.shape[0]
    return cv2.addWeighted(image, 0.2, dst, 0.8, 1.0)

out = visualize(reference, hlac_angles, nx, ny)
plt.imshow(out[:, :, ::-1])
plt.show()

# """OpenCVと同じような結果が得られると思いますが、HLAC特徴量では差のある部分を緑の輝度で表現することで、特に違いが大きい部分というのも分かると思います。  
# 当然、差分画像でも差の程度は可視化できますが、それは輝度差でしかありません。  
# HLAC特徴量での差は画像内の構造差を反映するので、より人間のフィーリングに近いかもしれません。  
# 最後に差分画像とHLAC特徴量による結果をまとめて結びとします。
# """

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.set_title('Difference based result')
plt.imshow(dst[:,:,::-1])
ax = fig.add_subplot(1,2,2)
ax.set_title('HLAC based result')
plt.imshow(out[:,:,::-1])
fig.set_figheight(4)
fig.set_figwidth(8)
plt.show()