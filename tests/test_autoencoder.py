import pytest
import numpy as np
import tensorflow as tf
from models.autoencoder import ResidualBlock, AttentionBlock, AutoEncoder3D

def test_residual_block_output_shape():
    block = ResidualBlock(filters=64)
    input_tensor = tf.random.normal([1, 32, 32, 32, 64])
    output_tensor = block(input_tensor)
    assert input_tensor.shape == output_tensor.shape

def test_attention_block_output_shape():
    block = AttentionBlock(filters=64)
    input_tensor = tf.random.normal([1, 32, 32, 32, 64])
    output_tensor = block(input_tensor)
    assert input_tensor.shape == output_tensor.shape

@pytest.fixture
def autoencoder():
    return AutoEncoder3D()

@pytest.fixture
def encoder(autoencoder):
    return autoencoder.get_encoder()

def test_autoencoder_output_shape(autoencoder):
    input_tensor = tf.random.normal([1, 64, 128, 128, 1])
    output_tensor = autoencoder(input_tensor)
    assert input_tensor.shape == output_tensor.shape

def test_encoder_output_shape(encoder):
    input_tensor = tf.random.normal([1, 64, 128, 128, 1])
    output_tensor = encoder(input_tensor)
    assert output_tensor.shape[-1] == 256
