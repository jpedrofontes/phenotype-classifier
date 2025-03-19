import argparse
import os
import random

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import ADASYN
from resnet3d import Resnet3DBuilder
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.metrics import (
    AUC,
    FalseNegatives,
    FalsePositives,
    Precision,
    Recall,
    TrueNegatives,
    TruePositives,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from datasets import CSVDataGenerator, DukeDataGenerator, DukeDataset
from models import CNN3D, AutoEncoder3D
from settings import Settings 

settings = Settings()

# Set random seeds for reproducibility
np.random.seed(settings.RANDOM_SEED)
random.seed(settings.RANDOM_SEED)
tf.random.set_seed(settings.RANDOM_SEED)


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

    Parameters:
        input_size (tuple): The dimensions to which the input data should be resized.
        phenotype (str, optional): The specific phenotype to filter the data by. Defaults to None.
        autoencoder (bool, optional): Whether the data is for an autoencoder model. Defaults to False.

    Returns:
        tuple: A tuple containing the train data generator and the test data generator.
    """
    dataset = DukeDataset(settings.DATASET_DIR, crop_size=input_size)
    train_generator = DukeDataGenerator(
        settings.DATASET_DIR,
        dataset=dataset,
        stage="train",
        dim=input_size,
        batch_size=settings.BATCH_SIZE,
        positive_class=phenotype,
        autoencoder=autoencoder,
    )
    test_generator = DukeDataGenerator(
        settings.DATASET_DIR,
        dataset=dataset,
        stage="test",
        dim=input_size,
        batch_size=settings.BATCH_SIZE,
        positive_class=phenotype,
        autoencoder=autoencoder,
    )

    return train_generator, test_generator


def create_directories(model_name=None):
    """
    Creates directories for saving model checkpoints, logs, and ROC curve images.

    Parameters:
        model_name (str, optional): The name of the model, used for creating specific directories.

    Returns:
        tuple: A tuple containing the checkpoint directory, log directory, and img directory paths.
    """
    if model_name:
        base_dir = os.path.join(settings.BASE_DATA_DIR, "jobs", settings.JOB_ID, model_name)
    else:
        base_dir = os.path.join(settings.BASE_DATA_DIR, "jobs", settings.JOB_ID)

    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    log_dir = os.path.join(base_dir, "logs")
    img_dir = os.path.join(base_dir, "img")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    return checkpoint_dir, log_dir, img_dir


def get_callbacks(is_tuner=False, model_name=None, checkpoint_dir=None, log_dir=None):
    """
    Generates a list of Keras callbacks for model training and evaluation.

    Parameters:
        is_tuner (bool): Flag to indicate if the function is being used for hyperparameter tuning.
                        If True, only EarlyStopping callback is added. Default is True.
        model_name (str, optional): Name of the model. If provided, the monitor metric will be "val_auc" and mode will be "max".
                                    If not provided, the monitor metric will be "val_loss" and mode will be "min".
        checkpoint_dir (str, optional): Directory where model checkpoints will be saved. Required if is_tuner is False.
        log_dir (str, optional): Directory where TensorBoard logs will be saved. Required if is_tuner is False.

    Returns:
        list: A list of Keras callbacks including EarlyStopping, and optionally ModelCheckpoint and TensorBoard.
    """
    if model_name:
        monitor_metric = "val_auc"
        mode = "max"
    else:
        monitor_metric = "val_loss"
        mode = "min"

    callbacks = [
        EarlyStopping(
            monitor=monitor_metric, patience=50, restore_best_weights=True, mode=mode
        ),
    ]

    if not is_tuner:
        callbacks.extend(
            [
                ModelCheckpoint(
                    filepath=os.path.join(checkpoint_dir, "weights.h5"),
                    monitor=monitor_metric,
                    save_best_only=True,
                    save_weights_only=True,
                    mode=mode,
                ),
                TensorBoard(log_dir=log_dir),
            ]
        )

    return callbacks


def plot_roc_curve(y_true, y_pred, img_dir):
    """
    Plots the Receiver Operating Characteristic (ROC) curve and saves it as an image.

    Parameters:
        y_true (array-like): True binary labels.
        y_pred (array-like): Target scores, can either be probability estimates of the positive class, confidence values, or binary decisions.
        img_dir (str): Directory where the ROC curve image will be saved.

    Returns:
        None
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="black", lw=2, label="No Skill", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(img_dir, "roc_curve.png"))
    plt.close()


def calculate_and_print_metrics(y_true, y_pred):
    """
    Calculate and print various classification metrics for binary classification.

    The function also calculates class weights and uses them to compute a weighted AUC.

    Parameters:
        y_true (array-like): True binary labels.
        y_pred (array-like): Predicted probabilities or scores.

    Prints:
        Accuracy, Precision, Recall, AUC, F1 Score, and Weighted AUC of the classifier on the test set.
        Confusion Matrix of the classifier on the test set.
    """
    # Calculate base metrics
    print(f"Accuracy: {accuracy_score(y_true, y_pred > 0.5):.4f}", flush=True)
    print(f"Precision: {precision_score(y_true, y_pred > 0.5):.4f}", flush=True)
    print(f"Recall: {recall_score(y_true, y_pred > 0.5):.4f}", flush=True)
    print(f"AUC: {roc_auc_score(y_true, y_pred):.4f}", flush=True)
    print(f"F1 Score: {f1_score(y_true, y_pred > 0.5):.4f}", flush=True)

    # Calculate class weights
    class_weights = calculate_binary_class_weights(y_true)
    sample_weights = np.array([class_weights[cls] for cls in y_true])

    # Calculate weighted AUC
    weighted_auc = roc_auc_score(y_true, y_pred, sample_weight=sample_weights)
    print(f"Weighted AUC: {weighted_auc:.4f}", flush=True)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred > 0.5)
    conf_matrix_df = pd.DataFrame(
        conf_matrix, index=np.unique(y_true), columns=np.unique(y_true)
    )

    print("Confusion Matrix:", flush=True)
    print(conf_matrix_df, flush=True)


def calculate_binary_class_weights(labels):
    """
    Calculate class weights for binary classification.

    This function computes the weights for each class in a binary classification
    problem to handle class imbalance. The weights are calculated based on the
    inverse frequency of each class in the training data.

    Parameters:
        labels (array-like): Array of shape (n_samples,) containing the class labels
                             for the training data. Must contain exactly two unique
                             classes.

    Returns:
        dict: A dictionary where keys are the class labels and values are the
              corresponding class weights.

    Raises:
        AssertionError: If the number of unique classes in labels is not equal to 2.
    """
    class_weights = {}
    total_samples = len(labels)
    unique_classes = np.unique(labels)

    assert unique_classes.shape[0] == 2, "Only binary classification is supported"

    for cls in unique_classes:
        n_x = np.sum(labels == cls)
        class_weights[cls] = (1 / n_x) * (total_samples / 2)

    return class_weights


def build_autoencoder_model(hp: kt.HyperParameters) -> AutoEncoder3D:
    """
    Builds and compiles a 3D autoencoder model based on the given hyperparameters.

    Parameters:
        hp (HyperParameters): Hyperparameters object containing the search space for model tuning.

    Returns:
        AutoEncoder3D: The autoencoder model.
    """
    input_size = (64, 128, 128)

    # Filter choices for the convolutional layers
    filters_map = {
        "64_128_256_512": [64, 128, 256, 512],
        "32_64_128_256": [32, 64, 128, 256],
        "16_32_64_128": [16, 32, 64, 128],
        "128_256_512_1024": [128, 256, 512, 1024],
        "64_128_256": [64, 128, 256],
        "32_64_128": [32, 64, 128],
        "128_256_512": [128, 256, 512],
        "64_128": [64, 128],
        "32_64": [32, 64],
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
        optimizer=Adam(
            learning_rate=ExponentialDecay(
                learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True
            ),
        ),
    )

    return autoencoder


def train_autoencoder(
    input_size, batch_size, num_epochs, notrain, tune, checkpoint_dir, log_dir
):
    """
    Trains an autoencoder model on the given dataset.

    Parameters:
        input_size (tuple): The dimensions of the input data (depth, width, height).
        batch_size (int): The number of samples per batch.
        num_epochs (int): The number of epochs to train the model.
        notrain (bool): If True, the model will not be trained.
        tune (bool): If True, hyperparameter tuning will be performed using Keras Tuner.
        checkpoint_dir (str): Directory to save model checkpoints.
        log_dir (str): Directory to save TensorBoard logs.

    Returns:
        None

    The function performs the following steps:
        1. Loads the dataset using the specified input size.
        2. If tuning is enabled, performs hyperparameter tuning using Keras Tuner's RandomSearch.
        3. Builds the autoencoder model.
        4. If notrain is False, trains the model using the training dataset.
        5. Predicts latent space values for the training dataset and saves them to a CSV file.
    """
    train_generator, test_generator = load_dataset(input_size, autoencoder=True)

    if tune:
        tuner = kt.RandomSearch(
            build_autoencoder_model,
            objective=kt.Objective("val_loss", direction="min"),
            max_trials=150,
            directory=f"{settings.BASE_DATA_DIR}/keras-tuner/",
            project_name=settings.JOB_ID,
        )
        callbacks = get_callbacks(is_tuner=True)
        tuner.search(
            train_generator,
            validation_data=test_generator,
            batch_size=batch_size,
            epochs=num_epochs,
            verbose=2,
            callbacks=callbacks,
        )

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"Best hyperparameters: {best_hps.values}", flush=True)

        model: AutoEncoder3D = tuner.hypermodel.build(best_hps)
        encoder = model.get_encoder()
    else:
        autoencoder = AutoEncoder3D(
            depth=input_size[0],
            width=input_size[1],
            height=input_size[2],
            filters=[64, 128, 256, 512],
            latent_space_size=256,
        )
        model_name = f"AutoEncoder3D.{settings.JOB_ID}"
        autoencoder.compile(
            loss="mse",
            optimizer=Adam(
                learning_rate=ExponentialDecay(
                    0.00001, decay_steps=10000, decay_rate=0.9, staircase=True
                ),
            ),
        )

        if not notrain:
            callbacks = get_callbacks(
                is_tuner=False, checkpoint_dir=checkpoint_dir, log_dir=log_dir
            )
            autoencoder.fit(
                train_generator,
                batch_size=batch_size,
                epochs=num_epochs,
                verbose=2,
                validation_data=test_generator,
                callbacks=callbacks,
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
        os.path.join(settings.BASE_DATA_DIR, "latent_space_values.csv"),
        index=False,
    )
    print(
        "Latent space values have been successfully saved to latent_space_values.csv",
        flush=True,
    )


def train_model(
    input_size,
    batch_size,
    num_epochs,
    phenotype,
    notrain,
    model_type,
    checkpoint_dir,
    log_dir,
    img_dir,
):
    """
    Trains a 3D CNN or ResNet model on the given phenotype dataset.

    Parameters:
        input_size (tuple): The dimensions of the input data (depth, width, height).
        batch_size (int): The number of samples per batch.
        num_epochs (int): The number of epochs to train the model.
        phenotype (str): The phenotype to classify.
        notrain (bool): If True, the model will not be trained.
        model_type (str): The type of model to train ('cnn' or 'resnet').
        checkpoint_dir (str): Directory to save model checkpoints.
        log_dir (str): Directory to save TensorBoard logs.
        img_dir (str): Directory to save ROC curve images.

    Returns:
        None

    Prints:
        Accuracy, Precision, Recall, AUC, F1 Score, and Weighted AUC of the classifier on the test set.
        Confusion Matrix of the classifier on the test set.
    """
    train_generator, test_generator = load_dataset(input_size, phenotype)

    if model_type == "cnn":
        model = CNN3D(depth=input_size[0], width=input_size[1], height=input_size[2])
    elif model_type == "resnet":
        model = Resnet3DBuilder.build_resnet_50(
            (input_size[0], input_size[1], input_size[2], 1), 1
        )

    model_name = f"{settings.PHENOTYPES[phenotype]}"
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(
            learning_rate=ExponentialDecay(
                0.00001, decay_steps=10000, decay_rate=0.9, staircase=True
            ),
        ),
        metrics=[
            FalseNegatives(name="fn"),
            FalsePositives(name="fp"),
            TrueNegatives(name="tn"),
            TruePositives(name="tp"),
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc"),
        ],
    )
    class_weight = {
        0: train_generator.weight_for_0,
        1: train_generator.weight_for_1,
    }

    if notrain:
        # We'll assume that the model has already been trained
        model.load_weights(os.path.join(checkpoint_dir, "weights.h5"))
    else:
        # Train the model
        callbacks = get_callbacks(
            model_name=model_name, checkpoint_dir=checkpoint_dir, log_dir=log_dir
        )
        model.fit(
            train_generator,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_data=test_generator,
            verbose=2,
            callbacks=callbacks,
            class_weight=class_weight,
        )

    # Build y_true and y_pred arrays for calculating metrics
    y_true = []
    y_pred = []

    for i in range(len(test_generator)):
        X_batch, y_batch = test_generator[i]
        y_true.extend(y_batch)
        y_pred.extend(model.predict(X_batch))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred).ravel()

    # Calculate metrics and plot ROC curve
    calculate_and_print_metrics(y_true, y_pred)
    plot_roc_curve(y_true, y_pred, img_dir)


def train_svm(phenotype, csv_file_path, img_dir):
    """
    Train a Support Vector Machine (SVM) classifier for a given phenotype.

    This function reads latent space values from a CSV file, normalizes the features,
    handles imbalanced data using ADASYN, splits the dataset into training and testing sets,
    performs hyperparameter tuning using Grid Search, and evaluates the best model.

    Parameters:
        phenotype (str): The positive class label for the phenotype to be classified.

    Returns:
        None

    Prints:
        Best parameters from Grid Search.
        Accuracy, Precision, Recall, AUC, F1 Score, and Weighted AUC of the classifier on the test set.
        Confusion Matrix of the classifier on the test set.
    """
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
    adasyn = ADASYN(random_state=settings.RANDOM_SEED)
    X_all, y_all = adasyn.fit_resample(X_all, y_all)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=settings.RANDOM_SEED
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

    # Calculate metrics and plot ROC curve
    calculate_and_print_metrics(y_test, y_pred)
    plot_roc_curve(y_test, y_pred, img_dir)


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

    setup_gpu()

    if args.autoencoder:
        checkpoint_dir, log_dir, img_dir = create_directories()
        train_autoencoder(
            settings.INPUT_SIZE,
            settings.BATCH_SIZE,
            settings.NUM_EPOCHS,
            args.notrain,
            args.tune,
            checkpoint_dir,
            log_dir,
        )
    elif args.svm:
        _, _, img_dir = create_directories(f"{settings.PHENOTYPES[args.phenotype]}")
        train_svm(args.phenotype, os.path.join(settings.BASE_DATA_DIR, "latent_space_values.csv"), img_dir)
    elif args.resnet:
        checkpoint_dir, log_dir, img_dir = create_directories(
            f"{settings.PHENOTYPES[args.phenotype]}"
        )
        train_model(
            settings.INPUT_SIZE,
            settings.BATCH_SIZE,
            settings.NUM_EPOCHS,
            args.phenotype,
            args.notrain,
            "resnet",
            checkpoint_dir,
            log_dir,
            img_dir,
        )
    else:
        checkpoint_dir, log_dir, img_dir = create_directories(
            f"{settings.PHENOTYPES[args.phenotype]}"
        )
        train_model(
            settings.INPUT_SIZE,
            settings.BATCH_SIZE,
            settings.NUM_EPOCHS,
            args.phenotype,
            args.notrain,
            "cnn",
            checkpoint_dir,
            log_dir,
            img_dir,
        )
