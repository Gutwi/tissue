#250121 ティッシュ本番用
#250123 WIDTH, HIGHTを整理、フィルタサイズやストライドも調整
#250126 バッチサイズを変更していく insp5

import cv2
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    # GlobalAveragePooling2D, BatchNormalization, Dropout
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageDraw, ImageFont #250120
import os

MODEL_NAME='tissue_model_250207-01.keras'
WIDTH=1024  #250121
HIGHT=576    #250121
NUM_CLASS=1 #250124
BT_SIZE=16  #250126 もとは32

def create_model(input_shape=(WIDTH, HIGHT, 3)):  #250124 元のモデル
    """Create a simple CNN model for binary classification"""
    model = Sequential([                                                #250121
        Conv2D(48, (3, 3), activation='relu', input_shape=input_shape),         #250124
        MaxPooling2D(2, 2),                                            
        Conv2D(64, (3, 3), activation='relu'),                                  #250124
        MaxPooling2D(2, 2),                                            
        Conv2D(80, (3, 3), activation='relu'),                          #250128
        MaxPooling2D(2, 2),                                            
        # Conv2D(256, (3, 3), activation='relu'),                          #250128
        # MaxPooling2D(2, 2),                                            
        # Conv2D(64, (3, 3), activation='relu'),                          #250126
        # MaxPooling2D(2, 2),
        Flatten(),                                                      #250122   一次元に落としている
        Dense(96, activation='relu'),                                   #250128
        Dense(1, activation='sigmoid')
    ])
    return model
    #250124 576x576ではepoc=10でもおおむねOK。やはり縦横サイズの相違がキーか。
    #250124 256x128でもepoc=10でaccuracy=0.84,val_accuracy=0.94。画像サイズとフィルタサイズ？
    #250124 512x256ではepoc=10でaccuracy=0.5x,val_accuracy=0.5x。画像サイズとフィルタサイズのミスマッチと確定
    #250204 224x224ではepoc=12でaccuracy=0.6x,val_accuracy=0.3x。KerasTunerによる15epoch探索後のベストHP値

def setup_data_generators(train_dir, validation_dir):
    """Set up data generators for training and validation"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=2,      #250205 10 -> 2
        width_shift_range=0.02, #250205 0.2 -> 0.02
        height_shift_range=0.02,#250205 0.2 -> 0.02
        # shear_range=0.2,
        zoom_range=0.9,         #250205 0.2 -> 0.9
        # horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(HIGHT,WIDTH),                         #250123
        batch_size=BT_SIZE,                         #250126
        class_mode='binary'
    )
    
    # クラスラベルの出力
    print("\nクラスラベルの対応:")   #250120
    print(train_generator.class_indices)    #250120

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(HIGHT,WIDTH),                          #250123
        batch_size=BT_SIZE,                         #250126
        class_mode='binary'
    )

    return train_generator, validation_generator


def train_model(model, train_generator, validation_generator, epochs=10):
    """Train the model"""
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size
    )
    return history

def predict_single_image(model, image_path):
    """Predict whether a single image contains a defect"""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (WIDTH,HIGHT))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)

    # return "不良品" if prediction[0] > 0.5 else "合格品"    #250120
    result = "不良品" if prediction[0] < 0.5 else "合格品"  #250120
    result_text = result + str(prediction[0])   #250120

    return result_text  #250120


def capture_and_predict(model):
    """Capture from webcam and predict in real-time"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("カメラを開けません。カメラの接続を確認してください。")
            return False

        print("カメラを起動しました。'q'キーで終了します...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("カメラからの読み取りに失敗しました")
                break
                
            # 画像の前処理
            resized = cv2.resize(frame, (WIDTH, HIGHT))
            normalized = resized / 255.0
            batch = np.expand_dims(normalized, axis=0)
            
            # 予測
            prediction = model.predict(batch)
            # result = "不良品" if prediction[0] > 0.5 else "合格品"  #250120
            result = "不良品" if prediction[0] < 0.5 else "合格品"  #250120
            color = (255,125,0) if prediction[0] < 0.5 else (125,255,0) #250120

            # 結果とraw予測値を表示
            display_text = result + str(prediction[0])   #250120
  
            # BGRからRGBに変換（PILで描画するため）   250120
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # テキストを描画
            frame_with_text = cv2_putText_jp(
                frame_rgb, 
                display_text, 
                (10, 30),  # 位置
                None,      # フォントフェイス（使用されない）
                32,       # フォントサイズ
                color  # 色（RGB） 250120
            )
            # RGBからBGRに戻す
            frame_bgr = cv2.cvtColor(frame_with_text, cv2.COLOR_RGB2BGR)
        
            cv2.imshow('Inspection', frame_bgr) #250120
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        return True
    
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return False
    
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Windowsでの表示の問題を解決するための追加待機


def train_menu():
    """Training menu function"""
    print("\n=== モデルの学習 ===")
    print("必要なフォルダ構成:")
    print("dataset/")
    print("  ├── train/")
    print("  │   ├── good/        # 合格品の訓練用画像")
    print("  │   └── defective/   # 不良品の訓練用画像")
    print("  └── validation/")
    print("      ├── good/        # 合格品の検証用画像")
    print("      └── defective/   # 不良品の検証用画像")
    
    input("\nフォルダ構成を確認したら Enter キーを押してください...")
    
    model = create_model()
    train_generator, validation_generator = setup_data_generators(
        train_dir='dataset/train',
        validation_dir='dataset/validation'
    )

    model.summary()         #250128
    
    epochs = int(input("\n学習回数(epochs)を入力してください (推奨: 10-20): "))
    history = train_model(model, train_generator, validation_generator, epochs=epochs)
    
    model.save(MODEL_NAME)  #250117

    print(f"\nモデルを保存しました: {MODEL_NAME}")     #250117


def predict_menu():
    """Prediction menu function"""
    # if not os.path.exists('tissue_inspection_model.h5'):
    if not os.path.exists(MODEL_NAME):  #250117
        print(f"エラー: モデルファイル({MODEL_NAME})が見つかりません。")    #250117
        print("先にモデルの学習を実行してください。")
        return
    
    # model = load_model('tissue_inspection_model.h5')
    model = load_model(MODEL_NAME)    #250117

    while True:
        print("\n=== 予測モード ===")
        print("1: 単一画像の予測")
        print("2: カメラからのリアルタイム予測")
        print("3: メインメニューに戻る")
        
        choice = input("選択してください (1-3): ")
        
        if choice == '1':
            image_path = input("画像のパスを入力してください: ")
            if os.path.exists(image_path):
                result = predict_single_image(model, image_path)
                print(f"\n予測結果: {result}")
            else:
                print("エラー: 指定された画像が見つかりません。")
        
        elif choice == '2':
            success = capture_and_predict(model)
            if not success:
                print("カメラの処理中にエラーが発生しました。")
            print("\nカメラモードを終了しました。")
        
        elif choice == '3':
            break
        
        else:
            print("無効な選択です。もう一度選択してください。")


#250120
def cv2_putText_jp(img, text, org, fontFace, fontScale, color):
    """日本語テキストを画像に描画する関数"""
    # PIL Image に変換
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    
    # フォントの設定（Windows標準フォントを使用）
    try:
        font = ImageFont.truetype('msgothic.ttc', fontScale)
    except:
        try:
            font = ImageFont.truetype('meiryo.ttc', fontScale)
        except:
            font = ImageFont.truetype('arial.ttf', fontScale)
    
    # テキスト描画
    draw.text(org, text, font=font, fill=color)
    
    # OpenCV画像に戻す
    return np.array(img_pil)

            
def main_menu():
    """Main menu function"""
    while True:
        print("\n=== ポケットティッシュ検品システム ===")
        print("1: モデルの学習")
        print("2: 予測の実行")
        print("3: 終了")
        
        choice = input("選択してください (1-3): ")
        
        if choice == '1':
            train_menu()
        elif choice == '2':
            predict_menu()
        elif choice == '3':
            print("システムを終了します")
            break
        else:
            print("無効な選択です。もう一度選択してください。")

if __name__ == "__main__":
    main_menu()