import keras
from keras import layers


class CNN3D:        
    def __init__(self, depth=64, width=128, height=128):
        """Build a 3D convolutional neural network model."""
        # Stack layers
        inputs = keras.Input((width, height, depth, 1))

        # Stack convolutional and pooling layers
        x = inputs
        for i in range(4):
            x = layers.Conv3D(filters=64 * (2 ** i),
                            kernel_size=3, activation="relu")(x)
            x = layers.MaxPool3D(pool_size=2)(x)
            x = layers.BatchNormalization()(x)

        # Add a global average pooling layer
        x = layers.GlobalAveragePooling3D()(x)

        # Add a dense layer with dropout
        x = layers.Dense(units=512, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        # Define the output tensor
        outputs = layers.Dense(units=1, activation="sigmoid")(x)

        # Define the model
        self.model = keras.Model(inputs, outputs, name="3dcnn")
        
    def __get_model__(self):
        return self.model
