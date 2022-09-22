import tensorflow as tf

class CustomModel:
    def __init__(self, sizes, input_shape, num_classes):
        self._sizes = sizes
        self._input_shape = input_shape
        self._num_classes = num_classes
        
    def make_model(self):
        inputs = tf.keras.Input(shape=self._input_shape)

        # Entry block
        x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
        x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        for size in self._sizes:
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = tf.keras.layers.Conv2D(size, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = tf.keras.layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        x = tf.keras.layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        if self._num_classes == 2:
            activation = "sigmoid"
            units = 1
        else:
            activation = "softmax"
            units = self._num_classes
        
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(units, activation=activation)(x)
        return tf.keras.Model(inputs, outputs)
