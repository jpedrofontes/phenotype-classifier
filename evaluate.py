import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import ndimage
import visualkeras

from models.cnn3d import CNN3D

# Define the input size of the volume of images
input_size = (64, 128, 128)


def load_model(weights_filepath):
    'Define the function to load the model weights'
    # Create an instance of the CNN3D class to build the model
    model = CNN3D(input_size[0], input_size[1], input_size[2]).__get_model__()
    # Load the weights from the specified filepath
    model.load_weights(weights_filepath)
    initial_learning_rate = 0.0001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    metrics = [
        tf.keras.metrics.FalseNegatives(name="fn"),
        tf.keras.metrics.FalsePositives(name="fp"),
        tf.keras.metrics.TrueNegatives(name="tn"),
        tf.keras.metrics.TruePositives(name="tp"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name='auc'),
    ]
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=metrics,
    )
    return model


def make_prediction(model, image_filepath):
    'Define the function to make a prediction using the loaded model'
    # Read the volume of images from the specified filepath
    images = process_scan(image_filepath)
    # Convert the images to a numpy array
    images = tf.cast(images, dtype=tf.float32)
    # images = tf.expand_dims(images, axis=-1)
    # Use the model to make a prediction on the input images
    prediction = model.predict(images)
    return prediction


def process_scan(image_folder):
    """Read and resize volume"""
    # Read scan
    volume = read_from_folder(image_folder)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(
        volume, desired_depth=input_size[0], desired_width=input_size[1], desired_height=input_size[2])
    return volume


def read_from_folder(image_folder):
    'Define the function to read a volume of images from a folder'
    volume = []
    for img_path in os.listdir(image_folder):
        # Read the image
        img = Image.open(os.path.join(image_folder, img_path))  # .convert('RGB')
        img = img.resize((input_size[0], input_size[1]))
        img = np.array(img)
        # Add to volume
        volume.append(img)
    volume = np.array(volume)
    return volume


def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(volume, desired_width=128, desired_height=128, desired_depth=64):
    """Resize across z-axis"""
    # Get current depth
    current_depth = volume.shape[0]
    current_width = volume.shape[1]
    current_height = volume.shape[2]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Resize across z-axis
    volume = ndimage.zoom(
        volume, (depth_factor, width_factor, height_factor), order=1)
    return volume


if __name__ == "__main__":
    # Get the model name and weights filepath from the command line arguments
    model_name = sys.argv[1]
    weights_filepath = os.path.join(
        "/home/mguevaral/jpedro/phenotype-classifier/checkpoints", model_name, "weights.h5")
    # Load the model from the specified filepath
    model = load_model(weights_filepath)
    print(model.summary())
    # visualkeras.layered_view(model, to_file='output.png', legend=True)
    # Get the image filepath from the command line arguments
    image_filepath = sys.argv[2]
    # Make a prediction on the image using the loaded model
    prediction = make_prediction(model, image_filepath)
    print(f"Prediction: {prediction}")
    # Get the names of the metrics
    metric_names = [name.replace("val_", "") for name in model.metrics_names]
    # Print the final metrics in a readable format
    print("Model:", model_name)
    for i in range(len(prediction)):
        print(f"{metric_names[i]}: {prediction[i]}")
