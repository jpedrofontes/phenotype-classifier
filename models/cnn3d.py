import tensorflow as tf
import visualkeras


class CNN3D:
    def __init__(self, depth=64, width=128, height=128):
        """Build a 3D convolutional neural network model."""
        inputs = tf.keras.Input((depth, width, height, 1))
        x = inputs
        
        for i in [512, 256, 128, 64]:
            x = tf.keras.layers.Conv3D(filters=i, kernel_size=3, activation="relu")(x)
            x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.GlobalAveragePooling3D()(x)
        x = tf.keras.layers.Dense(units=64, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)
        self.model = tf.keras.Model(inputs, outputs, name="3d_cnn")

    def __get_model__(self):
        return self.model


if __name__ == "__main__":
    model = CNN3D()
    visualkeras.layered_view(
        model.__get_model__(),
        legend=True,
        draw_volume=False,
        spacing=30,
        to_file="cnn3d.png",
    )
