import cv2
import numpy as np
import keras
from sklearn.cluster import KMeans
from keras.api.applications import MobileNetV2
from keras.src.utils.image_utils import img_to_array
from keras.models import load_model

# 定数設定
IMG_WIDTH, IMG_HEIGHT = 1024, 576
NUM_CLUSTERS = 3  # k-means で分類する色の数
MODEL_PATH = "tissue_orientation_model.h5"  # 事前学習済みの向き判定モデル

# 向き判定モデルをロード
# model = load_model(MODEL_PATH)    #250303

# 色の順番とダブりを検出する関数
def check_color_order(image):
    # 画像をリサイズ
    image_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    pixels = image_resized.reshape(-1, 3)  # (H*W, 3) に変換
    # cv2.imshow("Resized",image_resized)   #250303

    # k-means クラスタリング
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)  # クラスタごとの色

    # 色のソート（明度順やカラースペース変換で並び替え）
    sorted_colors = sorted(colors, key=lambda c: c[0] + c[1] + c[2])  # RGB の合計値で並び替え
    print("検出された色:", sorted_colors)   
    # print("検出された色:", colors)            #250303

    # 正しい色の順番と比較（例: [黄緑, 青緑, 紺]）
    correct_order = [(190, 210, 110), (150, 220, 200), (25, 40, 105)]  # 仮の正しい RGB
    error = sum(np.linalg.norm(np.array(sorted_colors) - np.array(correct_order), axis=1)) #250303  > 100 #250303
    # error = sum(np.linalg.norm(np.array(colors) - np.array(correct_order), axis=1)) #250303  > 100
    is_duplicate = len(set(map(tuple, sorted_colors))) < 3  # 3種類の色があるか

    return error, is_duplicate

# 向きを判定する関数
def check_orientation(image):
    image_resized = cv2.resize(image, (224, 224))  # MobileNetV2 の入力サイズに合わせる
    image_array = img_to_array(image_resized) / 255.0  # 正規化
    image_array = np.expand_dims(image_array, axis=0)  # バッチ次元追加

    # 推論
    predictions = model.predict(image_array)
    class_labels = ["正しい向き", "上下逆", "裏返し"]
    predicted_class = np.argmax(predictions)

    return class_labels[predicted_class]

# メイン処理
def inspect_tissue(image_path):
    # 画像を読み込む
    image = cv2.imread(image_path)
    if image is None:
        print("画像が読み込めません")
        return

    # 色の順番 & ダブりチェック
    color_error, duplicate = check_color_order(image)

    # 向きのチェック
    # orientation = check_orientation(image)    #250303 学習できるまではキャンセル

    # 結果の出力
    print("色の順番:", color_error)
    print("ダブリ:", duplicate)
    # print("向き:", orientation) #250303

    # 結果を描画
    result_text = f"Order Error: {color_error}, Duplicate: {duplicate}" #250303, 向き: {orientation}"
    cv2.putText(image, result_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Inspection Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 画像を検査
image_path = "test_bad1.jpg"  # テスト画像
inspect_tissue(image_path)
