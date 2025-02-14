import os
import cv2
import keras
import numpy as np
import keras_tuner as kt
from keras_tuner.src.engine.hyperparameters import HyperParameters
from keras_tuner.applications import HyperXception
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageDraw, ImageFont

MODEL_NAME='kt_image_realtis00.keras'
WIDTH=224  #250121
HIGHT=224    #250121
NUM_CLASS=1 #250124
BT_SIZE=16  #250126 もとは32
F_STEP=16   #250130 250205:32->16
EPC=3      #250204 epochs

# hp4tune=HyperParameters()

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
    # print("\nクラスラベルの対応:")   #250120
    # print(train_generator.class_indices)    #250120

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(HIGHT,WIDTH),                  #250123
        batch_size=BT_SIZE,                         #250126
        class_mode='binary'
    )
    return train_generator, validation_generator

#250210
#モデル構築
# def build_model(hp):
#     model = Sequential()
#     model.add(
#         Conv2D(
#             filters=hp.Int('conv_1_filter', min_value=16, max_value=48, step=F_STEP),
#             kernel_size=hp.Choice('conv_1_kernel', values = [3,5]),
#             # kernel_size=3,
#             activation='relu',
#             input_shape=(WIDTH, HIGHT, 3)
#         )
#     )
#     model.add(MaxPooling2D(2, 2))    
#     model.add(
#         Conv2D(
#             filters=hp.Int('conv_2_filter', min_value=48, max_value=80, step=F_STEP),
#             kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
#             activation='relu'
#         )
#     )
#     model.add(MaxPooling2D(2, 2))
#     model.add(
#         Conv2D(
#             filters=hp.Int('conv_3_filter', min_value=80, max_value=112, step=F_STEP),
#             # kernel_size=hp.Choice('conv_3_kernel', values = [3]),
#             kernel_size=3,
#             activation='relu'
#         )
#     )
#     model.add(MaxPooling2D(2, 2))
#     model.add(Flatten())
#     model.add(
#         Dense(
#             units=hp.Int("units", min_value=32, max_value=112, step=F_STEP),
#             activation="relu"
#         )
#     )
#     # model.add(Dense(NUM_CLASS, activation="softmax"))  #250130 10->num_classes
#     model.add(Dense(NUM_CLASS, activation="sigmoid"))  #250130 10->num_classes
#     model.compile(
#         optimizer="adam",
#         loss="binary_crossentropy",    #250204 pre:categorical_crossentropy
#         metrics=["accuracy"],
#     )
#     model.summary()
#     hp4tune=hp
#     return model

#250205:ImgGen最適化前
# conv_1_filter: 64,conv_1_kernel: 3,conv_2_filter: 64,conv_2_kernel: 5,
# conv_3_filter: 64,conv_3_kernel: 5,units: 96,Score: 0.7083333134651184

#250205:ImgGen最適化後
#conv_1_filter: 32,conv_1_kernel: 3,conv_2_filter: 32,conv_2_kernel: 3
#conv_3_filter: 64,conv_3_kernel: 5,units: 128,Score: 0.625

#250206:
#conv_1_filter: 48,conv_1_kernel: 3,conv_2_filter: 64,conv_2_kernel: 3
#conv_3_filter: 80,conv_3_kernel: 3,units: 96,Score: 0.875

train_generator, validation_generator = setup_data_generators(
    train_dir='dataset/train',
    validation_dir='dataset/validation'
    )

# hypermodel = HyperXception(input_shape=(HIGHT,WIDTH, 3), classes=1)    #250207
# hp4tune = HyperParameters()
# # hp4tune.Fixed('learning_rate', value=1e-2)
# modelX = hypermodel.build(hp4tune)

callback = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=3)    #250210

# # 最終層を変更（softmax → sigmoid）
# x = modelX.layers[-2].output  # 最終層の1つ手前の出力を取得
# new_output = Dense(1, activation='sigmoid')(x)  # 2クラス分類用に1ユニットのsigmoid層を追加
# new_model = Model(inputs=modelX.input, outputs=new_output)

# new_model.summary()

tuner = kt.RandomSearch(  #250207
# tuner = kt.Hyperband(
    # hypermodel,
    HyperXception(input_shape=(HIGHT,WIDTH, 3), classes=1),    #250210
    # new_model,    250210
    # hyperparameters=hp4tune,
    loss="binary_crossentropy", #250210
    objective='val_accuracy', 
    max_trials=3,    #250207
    directory='my_dir',
    project_name='cnn_tuning_XcepC',
    # tune_new_entries=False
    # ,
    overwrite=True
    )

tuner.search(train_generator, epochs=EPC, validation_data=validation_generator
,callbacks=[callback] #250210
)    #250207
best_hp = tuner.get_best_hyperparameters()[0]   #250130

tuner.results_summary(5)
# best_model = tuner.get_best_models()[0]    #250130
# # # print(best_model.summary())

b_model = build_model(best_hp)            #250130
# b_model.fit(train_generator,epochs=EPC)                #250130
