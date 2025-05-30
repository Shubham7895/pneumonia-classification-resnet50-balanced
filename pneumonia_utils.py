import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
import numpy as np

def preprocess(images):
    images = np.expand_dims(images, axis=-1)
    images_resized = tf.image.resize(images, (224, 224))
    images_rgb = tf.image.grayscale_to_rgb(images_resized)
    return images_rgb / 255.0

def build_model():
    input_layer = Input(shape=(224, 224, 3))
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_layer)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    return Model(inputs=input_layer, outputs=output)
