from keras_tuner.applications import HyperXception
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
# from tensorflow.keras.models import Model
from keras_tuner import HyperParameters

# ハイパーパラメータの設定
hypermodel = HyperXception(input_shape=(224, 224, 3), classes=1)

hp = HyperParameters()

# # モデルの構築
model = hypermodel.build(hp)

# # 最終層を変更（softmax → sigmoid）
x = model.layers[-2].output  # 最終層の1つ手前の出力を取得
new_output = Dense(1, activation='sigmoid')(x)  # 2クラス分類用に1ユニットのsigmoid層を追加
new_model = Model(inputs=model.input, outputs=new_output)

# モデルの概要を確認
new_model.summary()

# model = Sequential()
# model.add(
#     hypermodel,
#     Dense(1, activation="sigmoid")
# )

# model.summary()
