import os
import keras
import keras_tuner as kt
from keras_tuner.src.engine.hyperparameters import HyperParameters
from keras_tuner.src.applications import HyperXception,HyperResNet  #250214
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout,Input
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageDraw, ImageFont

MODEL_NAME='kt_image_realtis00.keras'
WIDTH=224  #250121
HIGHT=224    #250121
NUM_CLASS=1 #250124
BT_SIZE=16  #250126 もとは32
F_STEP=16   #250130 250205:32->16
EPC=1      #250204 epochs

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


train_generator, validation_generator = setup_data_generators(
    train_dir='dataset/train',
    validation_dir='dataset/validation'
    )

#Teigizumi model 
# input_tensor = Input(shape=(HIGHT,WIDTH, 3))  #250216
# vgg16_model = VGG16(
#     include_top=False, #全結合層を除外
#     weights='imagenet', 
#     input_tensor=input_tensor
#     )
hypermodel = HyperXception(
# hypermodel = HyperResNet(
    include_top=False,
    input_shape=(HIGHT,WIDTH, 3),
    classes=NUM_CLASS,
    # input_tensor=input_tensor #250216
    )    #250214
hp4tune = HyperParameters()
# hp4tune.Fixed('learning_rate', value=1e-2)
modelX = hypermodel.build(hp4tune)

# 最終層を変更（softmax → sigmoid）   #250213
#   Functional APIからSequentialへ
own_model = Sequential()    #250216
own_model.add(modelX)       #250216
# own_model.add(hypermodel)
# x = modelX.output
own_model.add(Flatten() )   #250216
# x = Flatten()(x)  #250216
# x = Dense(128,activation="relu")(x)   #250216
own_model.add(Dense(128,activation="relu") )
# # x = Dropout(0.5)(x)
# prediction = Dense(1, activation="sigmoid")(x)    #250216
own_model.add(Dense(1, activation="sigmoid"))   #250216
# own_model = Model(inputs=modelX.input,outputs=prediction) #250216
own_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) #250216

own_model.summary()

# tuner = kt.RandomSearch(  #250207
tuner = kt.Hyperband(
    own_model,
    # hypermodel,
    # HyperXception(input_shape=(HIGHT,WIDTH, 3), classes=1),    #250210
    # hyperparameters=hp4tune,
    loss="binary_crossentropy", #250210
    objective='val_accuracy', 
    # max_trials=3,    #250207
    directory='my_dir',
    project_name='cnn_tuning_Xcep',
    # tune_new_entries=False
    # ,
    overwrite=True
    )

callback = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=3)    #250210

# tuner.search(train_generator, epochs=EPC, validation_data=validation_generator
# ,callbacks=[callback] #250210
# )    #250207
# best_hp = tuner.get_best_hyperparameters()[0]   #250130

# tuner.results_summary(5)
# # best_model = tuner.get_best_models()[0]    #250130
# # # # print(best_model.summary())

# b_model = build_model(best_hp)            #250130
# # b_model.fit(train_generator,epochs=EPC)                #250130
