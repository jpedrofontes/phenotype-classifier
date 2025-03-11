import tensorflow as tf
import visualkeras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv3D,
    MaxPool3D,
    BatchNormalization,
    GlobalAveragePooling3D,
    Dense,
    Dropout,
)


class CNN3D(Model):
    """
    A 3D Convolutional Neural Network (CNN) model for volumetric data.

    Attributes:
        model (Model): The Keras Model instance representing the 3D CNN.
    """

    def __init__(self, depth=64, width=128, height=128):
        super(CNN3D, self).__init__()
        inputs = Input((depth, width, height, 1))
        x = inputs

        for i in [512, 256, 128, 64]:
            x = Conv3D(filters=i, kernel_size=3, activation="relu")(x)
            x = MaxPool3D(pool_size=2)(x)
            x = BatchNormalization()(x)

        x = GlobalAveragePooling3D()(x)
        x = Dense(units=64, activation="relu")(x)
        x = Dropout(0.3)(x)
        outputs = Dense(units=1, activation="sigmoid")(x)
        self.model = Model(inputs, outputs, name="3d_cnn")

    def call(self, inputs):
        return self.model(inputs)


if __name__ == "__main__":
    cnn3d = CNN3D()
    visualkeras.layered_view(
        cnn3d,
        legend=True,
        draw_volume=False,
        spacing=30,
        to_file="cnn3d.png",
    )
