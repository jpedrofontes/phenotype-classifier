import argparse
import os
import random

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import ADASYN
from resnet3d import Resnet3DBuilder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from datasets.csv_dataset import CSVDataGenerator
from datasets.dataset_3d import Dataset_3D
from datasets.generator_3d import DataGenerator
from models.autoencoder import AutoEncoder3D
from models.cnn3d import CNN3D


phenotypes = {0: "Luminal_A", 1: "Luminal_B", 2: "HER2_Enriched", 3: "Triple_Negative"}

# Set random seeds for reproducibility
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
tf.random.set_seed(random_seed)


def setup_gpu():
    """
    Configures TensorFlow to use available GPU(s) with memory growth enabled.

    This function lists all physical GPU devices available to TensorFlow and
    sets the memory growth option to True for each GPU. This allows TensorFlow
    to allocate memory on the GPU as needed, rather than reserving all memory
    at once. If an error occurs during this process, it is caught and printed.

    Raises:
        RuntimeError: If setting memory growth fails.
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e, flush=True)


def load_dataset(input_size, phenotype=None, autoencoder=False):
    """
    Loads the dataset and returns the train and test data generators.

    Args:
        input_size (tuple): The dimensions to which the input data should be resized.
        phenotype (str, optional): The specific phenotype to filter the data by. Defaults to None.
        autoencoder (bool, optional): Whether the data is for an autoencoder model. Defaults to False.

    Returns:
        tuple: A tuple containing the train data generator and the test data generator.
    """
    dataset = Dataset_3D("/data/mguevaral/crop_bbox/", crop_size=input_size)
    train_generator = DataGenerator(
        "/data/mguevaral/crop_bbox/",
        dataset=dataset,
        stage="train",
        dim=input_size,
        batch_size=batch_size,
        positive_class=phenotype,
        autoencoder=autoencoder,
    )
    test_generator = DataGenerator(
        "/data/mguevaral/crop_bbox/",
        dataset=dataset,
        stage="test",
        dim=input_size,
        batch_size=batch_size,
        positive_class=phenotype,
        autoencoder=autoencoder,
    )

    return train_generator, test_generator


def get_callbacks(is_tuner=True, model_name=""):
    """
    Generates a list of Keras callbacks for training a model.

    Args:
        model_name (str): The name of the model, used for saving checkpoints and logs.

    Returns:
        list: A list of Keras callbacks including EarlyStopping, ModelCheckpoint, and TensorBoard.
            - EarlyStopping: Monitors validation loss and stops training when it stops improving.
            - ModelCheckpoint: Saves the model weights to a specified filepath when the validation loss improves.
            - TensorBoard: Logs training metrics for visualization in TensorBoard.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
    ]

    if not is_tuner:
        callbacks.append(
            [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath="/home/mguevaral/jpedro/phenotype-classifier/checkpoints/"
                    + model_name
                    + "/weights.keras",
                    monitor="val_loss",
                    save_best_only=True,
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir="/home/mguevaral/jpedro/phenotype-classifier/logs/"
                    + model_name
                ),
            ]
        )

    return callbacks


def build_autoencoder_model(hp: kt.HyperParameters) -> AutoEncoder3D:
    """
    Builds and compiles a 3D autoencoder model based on the given hyperparameters.

    Args:
        hp (HyperParameters): Hyperparameters object containing the search space for model tuning.

    Returns:
        tuple: A tuple containing the compiled autoencoder model and the encoder model.
    """
    input_size = (64, 128, 128)

    # Filter choices for the convolutional layers
    filters_map = {
        "64_128_256_512": [64, 128, 256, 512],
        "32_64_128_256": [32, 64, 128, 256],
        "16_32_64_128": [16, 32, 64, 128],
        "8_16_32_64": [8, 16, 32, 64],
        "128_256_512_1024": [128, 256, 512, 1024],
    }
    filters_choice = hp.Choice(
        "filters",
        values=list(filters_map.keys()),
    )
    filters = filters_map[filters_choice]

    # Hyperparameters for the autoencoder model
    latent_space_size = hp.Int(
        "latent_space_size", min_value=64, max_value=1024, step=32
    )
    learning_rate = hp.Float(
        "learning_rate", min_value=1e-6, max_value=1e-2, sampling="LOG"
    )
    dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.05)

    autoencoder = AutoEncoder3D(
        depth=input_size[0],
        width=input_size[1],
        height=input_size[2],
        filters=filters,
        latent_space_size=latent_space_size,
        dropout_rate=dropout_rate,
    )

    autoencoder.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True
            )
        ),
    )

    return autoencoder


def train_autoencoder(input_size, num_epochs, notrain, tune):
    """
    Trains an autoencoder model on the given dataset.

    Parameters:
    input_size (tuple): The dimensions of the input data (depth, width, height).
    num_epochs (int): The number of epochs to train the model.
    notrain (bool): If True, the model will not be trained.
    tune (bool): If True, hyperparameter tuning will be performed using Keras Tuner.

    Returns:
        None

    The function performs the following steps:
    1. Loads the dataset using the specified input size.
    2. If tuning is enabled, performs hyperparameter tuning using Keras Tuner's Hyperband.
    3. Builds the autoencoder model.
    4. If notrain is False, trains the model using the training dataset.
    5. Predicts latent space values for the training dataset and saves them to a CSV file.
    """
    train_generator, test_generator = load_dataset(input_size, autoencoder=True)

    if tune:
        tuner = kt.Hyperband(
            build_autoencoder_model,
            objective="val_loss",
            max_epochs=50,
            factor=3,
            directory="/home/mguevaral/jpedro/phenotype-classifier/hyperband/"
            + os.environ.get("SLURM_JOB_ID"),
            project_name="autoencoder_tuning",
        )

        tuner.search(
            train_generator,
            epochs=num_epochs,
            verbose=2,
            validation_data=test_generator,
            callbacks=get_callbacks(is_tuner=True),
        )

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"Best hyperparameters: {best_hps.values}", flush=True)

        model = tuner.hypermodel.build(best_hps)
        encoder = model.get_encoder()
    else:
        autoencoder = AutoEncoder3D(
            depth=input_size[0],
            width=input_size[1],
            height=input_size[2],
            filters=[64, 128, 256, 512],
            latent_space_size=256,
        )
        model_name = "AutoEncoder3D." + os.environ.get("SLURM_JOB_ID")
        autoencoder.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                    0.00001, decay_steps=10000, decay_rate=0.9, staircase=True
                ),
            ),
        )

        if not notrain:
            autoencoder.fit(
                train_generator,
                epochs=num_epochs,
                verbose=2,
                validation_data=test_generator,
                callbacks=get_callbacks(is_tuner=False, model_name=model_name),
            )

    # Predict latent space values for the training dataset
    encoder = autoencoder.get_encoder()
    latent_values = encoder.predict(train_generator)
    all_y_values = []

    for i in range(len(train_generator)):
        _, y_batch, _ = train_generator[i]
        all_y_values.extend(y_batch)

    df_latent_values = pd.DataFrame(latent_values)
    df_latent_values["y"] = all_y_values
    df_latent_values.to_csv(
        "/home/mguevaral/jpedro/phenotype-classifier/datasets/latent_space_values.csv",
        index=False,
    )
    print(
        "Latent space values have been successfully saved to latent_space_values.csv",
        flush=True,
    )


def train_model(input_size, batch_size, num_epochs, phenotype, notrain, model_type):
    """
    Trains a 3D CNN or ResNet model on the given phenotype dataset.
    Args:
        input_size (tuple): The dimensions of the input data (depth, width, height).
        batch_size (int): The number of samples per batch.
        num_epochs (int): The number of epochs to train the model.
        phenotype (str): The phenotype to classify.
        notrain (bool): If True, the model will not be trained.
        model_type (str): The type of model to train ('cnn' or 'resnet').
    Returns:
        None
    """
    train_generator, test_generator = load_dataset(input_size, phenotype)

    if model_type == "cnn":
        model = CNN3D(
            depth=input_size[0], width=input_size[1], height=input_size[2]
        ).__get_model__()
        model_name = (
            "CNN_3D." + os.environ.get("SLURM_JOB_ID") + "." + phenotypes[phenotype]
        )
    elif model_type == "resnet":
        model = Resnet3DBuilder.build_resnet_50(
            (input_size[0], input_size[1], input_size[2], 1), 1
        )
        model_name = (
            "Resnet50_3D."
            + os.environ.get("SLURM_JOB_ID")
            + "."
            + phenotypes[phenotype]
        )

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                0.00001, decay_steps=10000, decay_rate=0.9, staircase=True
            ),
        ),
        metrics=[
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    class_weight = {
        0: train_generator.weight_for_0,
        1: train_generator.weight_for_1,
    }

    if not notrain:
        model.fit(
            train_generator,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_data=test_generator,
            verbose=2,
            callbacks=get_callbacks(model_name),
            class_weight=class_weight,
        )


def calculate_binary_class_weights(y_train):
    """
    Calculate class weights for binary classification.

    This function computes the weights for each class in a binary classification
    problem to handle class imbalance. The weights are calculated based on the
    inverse frequency of each class in the training data.

    Parameters:
    y_train (array-like): Array of shape (n_samples,) containing the class labels
                          for the training data. Must contain exactly two unique
                          classes.

    Returns:
    dict: A dictionary where keys are the class labels and values are the
          corresponding class weights.

    Raises:
    AssertionError: If the number of unique classes in y_train is not equal to 2.
    """
    class_weights = {}
    total_samples = len(y_train)
    unique_classes = np.unique(y_train)

    assert unique_classes.shape[0] == 2, "Only binary classification is supported"

    for cls in unique_classes:
        n_x = np.sum(y_train == cls)
        class_weights[cls] = (1 / n_x) * (total_samples / 2)

    return class_weights


def train_svm(phenotype):
    """
    Train a Support Vector Machine (SVM) classifier for a given phenotype.

    This function reads latent space values from a CSV file, normalizes the features,
    handles imbalanced data using ADASYN, splits the dataset into training and testing sets,
    performs hyperparameter tuning using Grid Search, and evaluates the best model.

    Args:
        phenotype (str): The positive class label for the phenotype to be classified.

    Returns:
        None

    Prints:
        Best parameters from Grid Search.
        Accuracy, Precision, Recall, AUC, and F1 Score of the classifier on the test set.
        Confusion Matrix of the classifier on the test set.
    """
    csv_file_path = (
        "/home/mguevaral/jpedro/phenotype-classifier/datasets/latent_space_values.csv"
    )
    csv_generator = CSVDataGenerator(
        csv_file_path,
        batch_size=32,
        shuffle=True,
        positive_class=phenotype,
    )

    X_all, y_all = [], []

    for i in range(len(csv_generator)):
        X_batch, y_batch = csv_generator[i]
        X_all.extend(X_batch)
        y_all.extend(y_batch)

    X_all, y_all = np.array(X_all), np.array(y_all)

    # Normalize the features
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    # Handle imbalanced data using ADASYN
    adasyn = ADASYN(random_state=random_seed)
    X_all, y_all = adasyn.fit_resample(X_all, y_all)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=random_seed
    )

    # Hyperparameter tuning using Grid Search
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf"],
    }
    grid_search = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    # Best parameters from Grid Search
    print(f"Best parameters: {grid_search.best_params_}", flush=True)

    # Use the best estimator for predictions and evaluations
    classifier = grid_search.best_estimator_

    y_pred = classifier.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}", flush=True)
    print(f"Precision: {precision_score(y_test, y_pred):.4f}", flush=True)
    print(f"Recall: {recall_score(y_test, y_pred):.4f}", flush=True)
    print(f"AUC: {roc_auc_score(y_test, y_pred):.4f}", flush=True)
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}", flush=True)

    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_df = pd.DataFrame(
        conf_matrix, index=np.unique(y_test), columns=np.unique(y_test)
    )
    print("Confusion Matrix:", flush=True)
    print(conf_matrix_df, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-p", "--phenotype", type=int, help="id of phenotype to identify"
    )
    parser.add_argument("-nt", "--notrain", action="store_true")
    parser.add_argument("-ae", "--autoencoder", action="store_true")
    parser.add_argument("-svm", "--svm", action="store_true")
    parser.add_argument("-r", "--resnet", action="store_true")
    parser.add_argument("-t", "--tune", action="store_true")
    args = parser.parse_args()

    np.random.seed(123)
    input_size = (64, 128, 128)
    batch_size = 2
    num_epochs = 1000

    setup_gpu()

    if args.autoencoder:
        train_autoencoder(input_size, num_epochs, args.notrain, args.tune)
    elif args.svm:
        train_svm(args.phenotype)
    elif args.resnet:
        train_model(
            input_size, batch_size, num_epochs, args.phenotype, args.notrain, "resnet"
        )
    else:
        train_model(
            input_size, batch_size, num_epochs, args.phenotype, args.notrain, "cnn"
        )
