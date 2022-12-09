import argparse
import os

import numpy as np
import tensorflow as tf

from load_dataset.generator_3d import DataGenerator
from load_dataset.dataset_3d import Dataset_3D
from models.cnn3d import CNN3D

phenotypes = {
    0: "Luminal-like",
    1: "ER/PR pos, HER2 pos",
    2: "ER & PR neg, HER2 pos",
    3: "Triple Negative"
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-p', '--phenotype', type=int,
                        help='id of phenotype to identify')
    args = parser.parse_args()

    np.random.seed(123)
    input_size = (64, 128, 128)
    batch_size = 4
    num_epochs = 100

    dataset = Dataset_3D('/data/mguevaral/crop_bbox/', crop_size=input_size)
    train_generator = DataGenerator(
        '/data/mguevaral/crop_bbox/', dataset=dataset, stage='train', dim=input_size, batch_size=batch_size, positive_class=args.phenotype)
    test_generator = DataGenerator(
        '/data/mguevaral/crop_bbox/', dataset=dataset, stage='test', dim=input_size, batch_size=batch_size, positive_class=args.phenotype)

    model = CNN3D(input_size[0], input_size[1], input_size[2]).__get_model__()
    model_name = "CNN3D." + os.environ.get("SLURM_JOB_ID") + "." + phenotypes[args.phenotype] 

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=50),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="/home/mguevaral/jpedro/phenotype-classifier/checkpoints/" + model_name + "/weights.h5", save_best_only=True),
        tf.keras.callbacks.TensorBoard(
            log_dir="/home/mguevaral/jpedro/phenotype-classifier/logs/" + model_name)
    ]    
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
    # Get the names of the metrics
    metric_names = [name.replace("val_", "") for name in model.metrics_names]
    # Train the model using the train_on_batch() method inside a function() method, passing in the callbacks
    with tf.keras.backend.function(model.inputs, model.outputs, callbacks=callbacks):
        # Train the model for a specified number of epochs
        for epoch in range(num_epochs):
            # Train the model using the train_on_batch() method in a loop
            for batch_data, batch_labels in train_generator:
                # Train the model on the current batch of data
                loss = model.train_on_batch(batch_data, batch_labels)
            # Evaluate the model on the validation and training datasets
            train_results = model.evaluate(train_generator, metrics=metrics)
            val_results = model.evaluate(test_generator, metrics=metrics)
            # Concatenate the epoch number and the values of the metrics
            metrics_string = f"Epoch {epoch + 1}/{num_epochs}: "
            for i in range(len(val_results)):
                # Add the training metrics to the string
                metrics_string += f"{metric_names[i]} = {train_results[i]}, "
                # Add the validation metrics to the string
                metrics_string += f"val_{metric_names[i]} = {val_results[i]}, "
            # Print the metrics in a single line
            print(metrics_string)
    # Load the best model
    model.load_weights(
        "/home/mguevaral/jpedro/phenotype-classifier/checkpoints/" + model_name + "/weights.h5")
    # Evaluate the model on the test data
    score = model.evaluate(test_generator, verbose=2)
    # Print the final metrics in a readable format
    print("Model:", model_name)
    for i in range(len(score)):
        print(f"{metric_names[i]}: {score[i]}")
