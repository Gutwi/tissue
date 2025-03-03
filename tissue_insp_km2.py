import cv2
import numpy as np
import keras
from sklearn.cluster import KMeans
# from keras.api.applications import MobileNetV2
# from keras.src.utils.image_utils import img_to_array
# from keras.models import load_model

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
    h, w, _ = image.shape
    third_w = w // 3
    half_h = h // 2  # 上下分割

    # 3つのティッシュ領域を取得
    tissues = [
        image[:, 0:third_w],       # 左
        image[:, third_w:2*third_w],  # 中央
        image[:, 2*third_w:w]      # 右
    ]

    detected_colors = []
    
    for tissue in tissues:
        top_half = tissue[0:half_h, :]  # 上半分
        bottom_half = tissue[half_h:h, :]  # 下半分

        top_color = get_dominant_colors(top_half)
        bottom_color = get_dominant_colors(bottom_half)

        detected_colors.append((tuple(top_color[0]), tuple(bottom_color[0])))
    # 正しい順番（黄緑/白、水色/白、紺色/白）と比較
    correct_order = [
        ((150, 200, 100), (255, 255, 255)),  # 黄緑/白
        ((100, 150, 200), (255, 255, 255)),  # 水色/白
        ((50, 50, 100), (255, 255, 255))     # 紺色/白
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
    print("検出された色ペア:", detected_colors)

    # 結果を描画
    result_text = f"Order Error: {error}" #250303, 向き: {orientation}"
    cv2.putText(image, result_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Inspection Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 画像を検査
image_path = "test_good1.jpg"  # テスト画像
inspect_tissue(image_path)
