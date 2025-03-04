import cv2
import numpy as np
import keras
from sklearn.cluster import KMeans
# from keras.api.applications import MobileNetV2
# from keras.src.utils.image_utils import img_to_array
# from keras.models import load_model
import matplotlib.pyplot as plt

# 定数設定
IMG_WIDTH, IMG_HEIGHT = 1024, 576
NUM_CLUSTERS = 3  # k-means で分類する色の数
MODEL_PATH = "tissue_orientation_model.h5"  # 事前学習済みの向き判定モデル

def get_dominant_colors(image, k=2):
    """ 画像の主要な2色を取得（白色を除外） """
    pixels = image.reshape(-1, 3)
    
    # K-meansクラスタリング
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    colors = kmeans.cluster_centers_.astype(int)

    # 白色に近いクラスタを背景とみなして除外
    brightness = np.sum(colors, axis=1)  # RGB値の合計（白に近いほど大きい）
    sorted_indices = np.argsort(brightness)  # 明るさでソート（暗い色を優先）

    return colors[sorted_indices[:2]]  # 上位2色を返す

def check_tissue_order(image):
    """ ティッシュの上下の色ペアを取得し、並び順をチェック """
    # h, w, _ = image.shape
    # third_w = w // 3
    # half_h = h // 2  # 上下分割   #250304

    # tissues = [
    #     image[:, 0:third_w],       # 左
    #     image[:, third_w:2*third_w],  # 中央
    #     image[:, 2*third_w:w]      # 右
    # ]

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
        # top_half = tissue[0:half_h, :]  # 上半分
        # bottom_half = tissue[half_h:h, :]  # 下半分  #250304

        top_half = tissue[:, 0:half_tis_w]  # ティッシュの上半分カラー
        bottom_half = tissue[:, half_tis_w:third_w]  # ティッシュの下半分

        # print("TOP: ",top_half)
        # print("BOTTOM: ",bottom_half)

        top_color = get_dominant_colors(top_half)         #250304
        bottom_color = get_dominant_colors(bottom_half)   #250304

        detected_colors.append((tuple(top_color[0]), tuple(bottom_color[0])))

    # 正しい順番（黄緑/白、水色/白、紺色/白）と比較
    correct_order = [
        ((185, 200, 120), (230, 230, 230)),  # 黄緑/白
        ((55, 170, 155), (230, 230, 230)),  # 水色/白
        ((30, 45, 105), (230, 230, 230))     # 紺色/白
    ]

    error = detected_colors != correct_order

    return error, detected_colors

# メイン処理
def inspect_tissue(image_path):
    # 画像を読み込む
    image = cv2.imread(image_path)
    if image is None:
        print("画像が読み込めません")
        return

    # 画像を読み込んで判定
    # image_resized = cv2.resize(image, (1024, 576))
    error, detected_colors = check_tissue_order(image)

    print("並び順エラーあり:", error)
    print("検出された色ペア:", convert_tuple(detected_colors) )   #250304

    color = np.array(convert_tuple(detected_colors), dtype=float) / 255
    # print(color)

    flat_color = np.array(color).reshape(-1,3)    #250304
    # print("検出された色ペア:",flat_color)

    # 画像として表示
    plt.imshow([flat_color])  # 1行の画像として表示    #250304
    plt.axis("off")  # 軸を非表示
    plt.show()


    # 結果を描画
    result_text = f"Order Error: {error}" #250303, 向き: {orientation}"
    cv2.putText(image, result_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(image, (W_L_EG, H_T_EG), (W_R_EG, H_B_EG), (255, 0, 0))         #250304
    cv2.line(image,( W_L_EG+TIS_W//3, H_T_EG),(W_L_EG+TIS_W//3,H_B_EG ),(0,255,0))
    cv2.line(image,(W_R_EG-TIS_W//3, H_T_EG),(W_R_EG-TIS_W//3,H_B_EG),(0,0,255))
    cv2.imshow("Inspection Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 入れ子タプルをNp.int64など型表示をなくす処理
def convert_tuple(tpl):
    return tuple(convert_tuple(x) if isinstance(x, tuple) else int(x) for x in tpl)

TIS_W = 710
TIS_H = 355
W_L_EG = 195
W_R_EG = W_L_EG + TIS_W
H_T_EG = 115
H_B_EG = H_T_EG + TIS_H

# 画像を検査
image_path = "test_good2.jpg"  # テスト画像
inspect_tissue(image_path)

# rgb = ((100, 50, 0), (200, 100, 50))

# # NumPy を使って 0〜1 に正規化
# color = np.array(rgb, dtype=float) / 255

# # 画像として表示
# plt.imshow([color])  # 1行の画像として表示
# plt.axis("off")  # 軸を非表示
# plt.show()