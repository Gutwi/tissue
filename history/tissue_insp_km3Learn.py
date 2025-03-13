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
NUM_CLUSTERS = 2  # k-means で分類する色の数
MODEL_PATH = "model_km3.pkl"  # 事前学習済みの向き判定モデル
DIST_TH = 100

# 正しい順番（黄緑/白、水色/白、紺色/白）と比較  250306
correct_colors = [
    ((120, 200, 185), (230, 230, 230)),  # 黄緑/白
    ((155, 170, 55), (230, 230, 230)),  # 水色/白
    ((105, 45, 30), (230, 230, 230))     # 紺色/白
]

# データを numpy 配列に変換
X = np.array([color for pair in correct_colors for color in pair])

print(X)

# K-Means クラスタリングで学習
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
kmeans.fit(X)

# 学習済みモデルを保存
with open(MODEL_PATH, "wb") as f:
    pickle.dump(kmeans, f)

print("学習完了！モデルを保存しました。")