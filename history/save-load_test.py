from keras.api.applications import ResNet50
from keras.src.losses import SparseCategoricalCrossentropy
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential,Model,load_model

base_model = ResNet50(include_top=False, weights='imagenet')
global_average_layer = GlobalAveragePooling2D()
output_layer = Dense(1)

model = Sequential([
    base_model,
    global_average_layer,
    output_layer
])

model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Save the model
model.save('model-v1.keras')

# Load the model with explicit input shape and compile
input_shape = (224, 224, 3)  # Example input shape for ResNet50
n_model = load_model('model-v1.keras', compile=False, custom_objects={'input_shape': input_shape})
n_model.compile(optimizer='adam',
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

# Print model summary
n_model.summary()