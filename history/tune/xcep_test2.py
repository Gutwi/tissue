import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import xception

import os
import keras
import keras_tuner as kt
from keras_tuner.src.applications import HyperXception  #250214
from keras.models import Sequential,Model


def build_model(hp):
    hp_model = kt.applications.HyperXception(
        input_shape=(128, 128, 3),  # 適切なサイズを設定
        classes=10,
        activation="softmax",
        tuner=hp
    )
    return hp_model

tuner = kt.Hyperband(
    build_model,
    objective="val_accuracy",
    max_epochs=10,
    directory="my_dir",
    project_name="xception_tuning"
)
