import cv2,os
import numpy as np
import pickle
from sklearn.cluster import KMeans
# from keras.api.applications import MobileNetV2
# from keras.src.utils.image_utils import img_to_array
# from keras.models import load_model
import matplotlib.pyplot as plt

# 定数設定
IMG_WIDTH, IMG_HEIGHT = 1024, 576
MODEL_PATH = "model_km3_Cdora2.pkl"  # 事前学習済みの正解モデル
NUM_CLUSTERS = 2  # k-means で分類する色の数

#ティッシュ色領域サイズ・座標
TIS_W = 710
TIS_H = 355
W_L_EG = 195
W_R_EG = W_L_EG + TIS_W
H_T_EG = 115
H_B_EG = H_T_EG + TIS_H

# 検査する画像（入力）
IMAGE_PATH = "./my_dir/tis_insp/Input/test_dra1.jpg"  # テスト画像


""" 画像の主要色を取得 """
def get_dominant_colors(image):

    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    pixels = image.reshape(-1, 3)
    labels = kmeans.fit_predict(pixels)

    #最も多く出現する色を主要色とする
    counts = np.bincount(labels) 
    main_color = kmeans.cluster_centers_[np.argmax(counts)].astype(int)

    return tuple(main_color)    #250305


def check_tissue_order(image):
    """ ティッシュの上下の色ペアを取得し、並び順をチェック """
    h, w = TIS_H, TIS_W
    third_w = w // 3
    # half_h = h // 2  # 上下分割
    half_tis_w = third_w // 2

    wid_mid_eg = W_L_EG + third_w
    wid_rit_eg = W_R_EG - third_w

    # 3つのティッシュ領域を取得
    tissues = [
        image[H_T_EG:H_B_EG, W_L_EG:wid_mid_eg],       # 左
        image[H_T_EG:H_B_EG, wid_mid_eg:wid_rit_eg],  # 中央
        image[H_T_EG:H_B_EG, wid_rit_eg:W_R_EG]      # 右
    ]

    detected_colors = []
    
    for tissue in tissues:
        top_half = tissue[:, 0:half_tis_w]  # ティッシュの上半分カラー
        bottom_half = tissue[:, half_tis_w:third_w]  # ティッシュの下半分

        # print("TOP: ",top_half)
        # print("BOTTOM: ",bottom_half)

        top_color = get_dominant_colors(top_half)         #250304
        bottom_color = get_dominant_colors(bottom_half)   #250304

        # detected_colors.append((tuple(top_color[0]), tuple(bottom_color[0])))
        detected_colors.append((top_color, bottom_color))  # 1ペアのみ記録    #250305

    return detected_colors


# 入れ子タプルをNp.int64など型表示をなくす処理
def convert_tuple(tpl):
    return tuple(convert_tuple(x) if isinstance(x, tuple) else int(x) for x in tpl)


#メイン
# 画像を読み込む
image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    print("画像が読み込めません")
    # return

target_name=os.path.basename(IMAGE_PATH)

# 画像を読み込んで判定
detected_colors = check_tissue_order(image_bgr)  #250306

# データを numpy 配列に変換
X = np.array([color for pair in detected_colors for color in pair])

# K-Means クラスタリングで学習
kmeans = KMeans(n_clusters=6, random_state=42, n_init=5)
kmeans.fit(X)

# # 学習済みモデルを保存
with open(MODEL_PATH, "wb") as f:
    pickle.dump(kmeans, f)

print("学習完了！モデルを保存しました。")


#### 結果の表示 ####
print("検出された色ペア:", convert_tuple(detected_colors) )   #250304

#正規化(plt表示用)
color = np.array(convert_tuple(detected_colors), dtype=float) / 255

#配列を1次元に
flat_color = np.array(color).reshape(-1,3)    #250304

#BGR->RGB(plt表示用)
flat_color_rgb = flat_color[:,[2,1,0]]         #250306

# 結果を描画
fig, ax = plt.subplots(2, 1, figsize=(8, 10))       #250306

ax[0].imshow([flat_color_rgb])  # 1行の画像として表示    #250304
ax[0].axis("off")  # 軸を非表示
ax[0].set_title("Correct Color")
ax[0].set_position([0.2,0.4,0.6,0.6])

img_rec=cv2.rectangle(image_bgr, (W_L_EG, H_T_EG), (W_R_EG, H_B_EG), (255, 0, 0))         #250304
img_lin1=cv2.line(image_bgr,( W_L_EG+TIS_W//3, H_T_EG),(W_L_EG+TIS_W//3,H_B_EG ),(0,255,0))
img_lin2=cv2.line(image_bgr,(W_R_EG-TIS_W//3, H_T_EG),(W_R_EG-TIS_W//3,H_B_EG),(0,0,255))
# img_tar=cv2.imshow("Inspection Result", image_bgr)

ax[1].imshow(img_rec)
ax[1].imshow(img_lin1)
ax[1].imshow(img_lin2)
ax[1].imshow(image_bgr[:, :, [2,1,0]])
ax[1].set_title("Inspection Result: "+target_name)
ax[1].axis("off")
ax[1].set_position([0.1,0,0.8,0.8])

plt.savefig("./my_dir/tis_insp/res_"+target_name+".png") #250221
plt.show()