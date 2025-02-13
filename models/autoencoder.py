import tensorflow as tf
import visualkeras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Layer,
    Activation,
    Conv3D,
    BatchNormalization,
    ReLU,
    MaxPool3D,
    Dropout,
    Flatten,
    Dense,
    Reshape,
    Conv3DTranspose,
    UpSampling3D,
)


class ResidualBlock(Layer):
    def __init__(self, filters, kernel_size=3, transpose=False, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.transpose = transpose
        if self.transpose:
            self.conv1 = Conv3DTranspose(
                filters, kernel_size, padding="same"
            )
            self.conv2 = Conv3DTranspose(
                filters, kernel_size, padding="same"
            )
        else:
            self.conv1 = Conv3D(filters, kernel_size, padding="same")
            self.conv2 = Conv3D(filters, kernel_size, padding="same")
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        self.bn2 = BatchNormalization()

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu(x + inputs)

    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "transpose": self.transpose,
            }
        )
        return config


class AttentionBlock(Layer):
    def __init__(self, filters, transpose=False, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.transpose = transpose
        if self.transpose:
            self.conv = Conv3DTranspose(
                filters, kernel_size=1, padding="same"
            )
        else:
            self.conv = Conv3D(filters, kernel_size=1, padding="same")
        self.bn = BatchNormalization()
        self.sigmoid = Activation("sigmoid")

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        return inputs * self.sigmoid(x)

    def get_config(self):
        config = super(AttentionBlock, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "transpose": self.transpose,
            }
        )
        return config


class AutoEncoder3D(Model):
    """
    A 3D convolutional autoencoder model.

    Attributes:
        autoencoder (Model): The complete autoencoder model.
        encoder (Model): The encoder part of the autoencoder model.

    Methods:
        get_encoder(): Returns the encoder model.

    Args:
        depth (int): The depth of the input volume. Default is 64.
        width (int): The width of the input volume. Default is 128.
        height (int): The height of the input volume. Default is 128.
        filters (list): A list of integers representing the number of filters
                        for each convolutional layer in the encoder. The decoder
                        uses the reverse of this list. Default is [64, 128, 256, 512].
        latent_space_size (int): The size of the latent space. Default is 256.
        dropout_rate (float): The dropout rate for dropout layers. Default is 0.0.
    """

    def __init__(
        self,
        depth=64,
        width=128,
        height=128,
        filters=[64, 128, 256, 512],
        latent_space_size=256,
        dropout_rate=0.0,
    ):
        super(AutoEncoder3D, self).__init__()
        self.depth = depth
        self.width = width
        self.height = height
        self.filters = filters
        self.latent_space_size = latent_space_size
        self.dropout_rate = dropout_rate

        encoder_filters = filters
        decoder_filters = filters[::-1]

        # Encoder
        inputs = Input((depth, width, height, 1))
        x = inputs

        for i in encoder_filters:
            x = Conv3D(
                filters=i, kernel_size=3, activation=None, padding="same"
            )(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = MaxPool3D(pool_size=2, padding="same")(x)
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)

        # Latent space
        shape_before_flattening = x.shape[1:]
        x = Flatten()(x)
        latent_space = Dense(
            units=latent_space_size, activation="relu"
        )(x)

        # Decoder
        x = Dense(
            units=tf.reduce_prod(shape_before_flattening), activation="relu"
        )(latent_space)
        x = Reshape(shape_before_flattening)(x)

        for i in decoder_filters:
            x = Conv3DTranspose(
                filters=i, kernel_size=3, activation=None, padding="same"
            )(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = UpSampling3D(size=2)(x)
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)

        x = Conv3DTranspose(
            filters=1, kernel_size=3, activation="sigmoid", padding="same"
        )(x)
        self.autoencoder = Model(inputs, x, name="3d_autoencoder")
        self.encoder = Model(inputs, latent_space, name="3d_encoder")

    def call(self, inputs, training=False):
        return self.autoencoder(inputs, training=training)

    def get_encoder(self) -> Model:
        """
        Builds and returns the encoder model.

        Returns:
            Model: The encoder model.
        """
        return self.encoder


if __name__ == "__main__":
    autoencoder = AutoEncoder3D()
    model = autoencoder.__get_model__()
    visualkeras.layered_view(
        model[0],
        legend=True,
        draw_volume=False,
        spacing=30,
        to_file="autoencoder3d.png",
    )
