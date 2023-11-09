import tensorflow as tf
from tensorflow import keras
from keras import layers


def get_model_conv(upscale_factor=3, channels=1):
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(channels*(upscale_factor**2), kernel_size=3, strides=upscale_factor, padding='same')(x)
    outputs = layers.Activation('sigmoid')(x)
    model = keras.Model(inputs, outputs)

    return model


def get_model_conv_seq(upscale_factor=3, channels=1):
    model = keras.Sequential([
        layers.Conv2D(64, 3, padding='same', input_shape=(None, None, channels)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2DTranspose(channels*(upscale_factor**2), kernel_size=3, strides=upscale_factor, padding='same'),
        layers.Activation('sigmoid')
    ])

    return model
