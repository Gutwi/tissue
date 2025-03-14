import cv2
import numpy as np
import pickle

# 正しい順番（黄緑/白、水色/白、紺色/白）と比較  250306
CORRECT_ORDER = [
    #spl
    # ((120, 200, 185), (230, 230, 230)),  # 黄緑/白
    # ((155, 170, 55), (230, 230, 230)),  # 水色/白
    # ((105, 45, 30), (230, 230, 230))     # 紺色/白
    #dora
    ((129, 171, 171), (214, 217, 226)),
    ((191, 184, 130), (220, 220, 227)),
    ((152, 157, 234), (218, 228, 235))
]

DIST_TH = 125   # OK/NG判定基準

#ティッシュ色領域サイズ・座標
TIS_W = 710
TIS_H = 355
W_L_EG = 195
W_R_EG = W_L_EG + TIS_W
H_T_EG = 115
H_B_EG = H_T_EG + TIS_H

# 学習済みモデルをロード
with open("model_km3_Cdora.pkl", "rb") as f:
    kmeans = pickle.load(f)

def get_dominant_colors(image):
    """ 画像の主要な色1つを取得 """
    pixels = image.reshape(-1, 3)
    labels = kmeans.predict(pixels)
    
    # 最も多く出現する色を主要色とする
    counts = np.bincount(labels)
    main_color = kmeans.cluster_centers_[np.argmax(counts)].astype(int)
    
    return tuple(main_color)

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
        # top_half = tissue[0:half_h, :]  # 上半分
        # bottom_half = tissue[half_h:h, :]  # 下半分  #250304

        top_half = tissue[:, 0:half_tis_w]  # ティッシュの上半分カラー
        bottom_half = tissue[:, half_tis_w:third_w]  # ティッシュの下半分

        # print("TOP: ",top_half)
        # print("BOTTOM: ",bottom_half)

        top_color = get_dominant_colors(top_half)         #250304
        bottom_color = get_dominant_colors(bottom_half)   #250304

        # detected_colors.append((tuple(top_color[0]), tuple(bottom_color[0])))
        detected_colors.append((top_color, bottom_color))  # 1ペアのみ記録    #250305


    # error = detected_colors != correct_order
    error = sum(np.linalg.norm(np.array(detected_colors) - np.array(CORRECT_ORDER), axis=1))
    error = np.linalg.norm(error)

    return error, detected_colors

# USBカメラを起動
cap = cv2.VideoCapture(0)

# カメラの解像度を設定（1024x576 に固定）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 画像をリサイズ
    frame_resized = cv2.resize(frame, (1024, 576))
    # normalized = frame_resized / 255.0

    # 並び順判定
    # error, detected_colors = check_tissue_order(normalized)
    error, detected_colors = check_tissue_order(frame_resized)

    print("エラー値： ",error)

    # 結果を画面に表示
    text = "OK" if error < DIST_TH else "NG"
    color = (0, 255, 0) if not error else (0, 0, 255)
    cv2.putText(frame_resized, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # ウィンドウに表示
    cv2.imshow("Tissue Inspection", frame_resized)
    # cv2.imshow("Tissue Inspection", frame)

    # "q"キーで終了
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 終了処理
cap.release()
cv2.destroyAllWindows()
