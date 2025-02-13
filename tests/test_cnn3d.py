import pytest
import tensorflow as tf
from models.cnn3d import CNN3D

@pytest.fixture
def model():
    return CNN3D().__get_model__()

def test_model_structure(model):
    assert len(model.layers) == 17, "Model should have 17 layers"

def test_input_shape(model):
    assert model.input_shape == (None, 64, 128, 128, 1), "Input shape should be (None, 64, 128, 128, 1)"

def test_output_shape(model):
    assert model.output_shape == (None, 1), "Output shape should be (None, 1)"

def test_model_compilation(model):
    try:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    except Exception as e:
        pytest.fail(f"Model compilation failed with error: {e}")
