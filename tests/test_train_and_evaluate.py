import os
import pytest
from unittest import mock

import keras_tuner as kt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from models.autoencoder import AutoEncoder3D
from train_and_evaluate import get_callbacks, build_autoencoder_model, calculate_binary_class_weights

def test_get_callbacks_is_tuner_true():
    callbacks = get_callbacks(is_tuner=True, model_name="test_model")
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], EarlyStopping)
    assert callbacks[0].monitor == "val_loss"
    assert callbacks[0].patience == 10
    assert callbacks[0].restore_best_weights

def test_get_callbacks_is_tuner_false():
    model_name = "test_model"
    
    with mock.patch("train_and_evaluate.slurm_job_id", "12345"):
        callbacks = get_callbacks(is_tuner=False, model_name=model_name)
        
        assert len(callbacks) == 3
        assert isinstance(callbacks[0], EarlyStopping)
        assert callbacks[0].monitor == "val_loss"
        assert callbacks[0].patience == 10
        assert callbacks[0].restore_best_weights

        assert isinstance(callbacks[1], ModelCheckpoint)
        assert callbacks[1].filepath == f"/data/mguevaral/jpedro/jobs/pheno_tr.12345/checkpoints/{model_name}/weights.keras"
        assert callbacks[1].monitor == "val_loss"
        assert callbacks[1].save_best_only

        assert isinstance(callbacks[2], TensorBoard)
        assert callbacks[2].log_dir == f"/data/mguevaral/jpedro/jobs/pheno_tr.12345/logs/{model_name}"

def test_get_callbacks_custom_patience():
    model_name = "test_model"
    patience = 20
    callbacks = get_callbacks(is_tuner=False, model_name=model_name)
    callbacks[0].patience = patience
    assert callbacks[0].patience == patience

def test_get_callbacks_custom_monitor():
    model_name = "test_model"
    monitor = "val_accuracy"
    callbacks = get_callbacks(is_tuner=False, model_name=model_name)
    callbacks[0].monitor = monitor
    assert callbacks[0].monitor == monitor

def test_build_autoencoder_model():
    hp = kt.HyperParameters()
    hp.Choice("filters", values=["64_128_256_512"])
    hp.Int("latent_space_size", min_value=64, max_value=1024, step=32)
    hp.Float("learning_rate", min_value=1e-6, max_value=1e-2, sampling="LOG")
    hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.05)

    model = build_autoencoder_model(hp)
    assert isinstance(model, AutoEncoder3D)
    assert model.optimizer.__class__ == Adam
    assert isinstance(model.optimizer.learning_rate, ExponentialDecay)
    assert model.loss == "mse"

def test_build_autoencoder_model_filters():
    hp = kt.HyperParameters()
    hp.Choice("filters", values=["32_64_128"])
    hp.Int("latent_space_size", min_value=64, max_value=1024, step=32)
    hp.Float("learning_rate", min_value=1e-6, max_value=1e-2, sampling="LOG")
    hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.05)

    model = build_autoencoder_model(hp)
    assert model.filters == [32, 64, 128]

def test_build_autoencoder_model_latent_space_size():
    hp = kt.HyperParameters()
    hp.Choice("filters", values=["64_128_256_512"])
    hp.Int("latent_space_size", min_value=64, max_value=1024, step=32)
    hp.Float("learning_rate", min_value=1e-6, max_value=1e-2, sampling="LOG")
    hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.05)

    model = build_autoencoder_model(hp)
    assert model.latent_space_size == hp.get("latent_space_size")

def test_build_autoencoder_model_learning_rate():
    hp = kt.HyperParameters()
    hp.Choice("filters", values=["64_128_256_512"])
    hp.Int("latent_space_size", min_value=64, max_value=1024, step=32)
    hp.Float("learning_rate", min_value=1e-6, max_value=1e-2, sampling="LOG")
    hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.05)

    model = build_autoencoder_model(hp)
    assert model.optimizer.learning_rate.initial_learning_rate == hp.get("learning_rate")

def test_build_autoencoder_model_dropout_rate():
    hp = kt.HyperParameters()
    hp.Choice("filters", values=["64_128_256_512"])
    hp.Int("latent_space_size", min_value=64, max_value=1024, step=32)
    hp.Float("learning_rate", min_value=1e-6, max_value=1e-2, sampling="LOG")
    hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.05)

    model = build_autoencoder_model(hp)
    assert model.dropout_rate == hp.get("dropout_rate")

def test_calculate_binary_class_weights_balanced():
    y_train = np.array([0, 1, 0, 1])
    class_weights = calculate_binary_class_weights(y_train)
    assert class_weights[0] == 1.0
    assert class_weights[1] == 1.0

def test_calculate_binary_class_weights_imbalanced():
    y_train = np.array([0, 1, 0, 1, 1, 1, 1, 1])
    class_weights = calculate_binary_class_weights(y_train)
    assert class_weights[0] == 2.0
    assert class_weights[1] == 0.6666666666666666  

def test_calculate_binary_class_weights_single_class():
    y_train = np.array([0, 0, 0, 0])
    
    with pytest.raises(AssertionError, match="Only binary classification is supported"):
        calculate_binary_class_weights(y_train)

def test_calculate_binary_class_weights_empty():
    y_train = np.array([])
    
    with pytest.raises(AssertionError, match="Only binary classification is supported"):
        calculate_binary_class_weights(y_train)

if __name__ == "__main__":
    pytest.main()
