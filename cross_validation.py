import argparse
import os

import numpy as np
import tensorflow as tf
from resnet3d import Resnet3DBuilder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from load_dataset.dataset_3d import Dataset_3D
from load_dataset.generator_3d import DataGenerator
from models.cnn3d import CNN3D

phenotypes = {0: "Luminal_A", 1: "Luminal_B",
              2: "HER2_Enriched", 3: "Triple_Negative"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-p", "--phenotype", type=int, help="id of phenotype to identify"
    )
    parser.add_argument(
        "-k", "--k-fold", type=int, help="number of folds to use", default=5
    )
    parser.add_argument("-t", "--notrain", action="store_true")
    args = parser.parse_args()

    np.random.seed(123)
    input_size = (64, 128, 128)
    batch_size = 4
    num_epochs = 100
    n_splits = 5

    # Create a list of all phenotypes for StratifiedKFold
    dataset = Dataset_3D("/data/mguevaral/crop_bbox/",
                            crop_size=input_size)
    phenotypes_list = [dataset.volumes[v]["phenotype"] for v in dataset.volumes]
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

    fold_num = 1

    for train_indices, val_indices in kf.split(np.zeros(len(phenotypes_list)), phenotypes_list):
        print(f"Training on fold {fold_num}/{n_splits}")        

        # Create data generators for the current fold
        train_generator = DataGenerator(
            "/data/mguevaral/crop_bbox/",
            dataset=dataset,
            indices=train_indices,
            dim=input_size,
            batch_size=batch_size,
            positive_class=args.phenotype,
        )
        val_generator = DataGenerator(
            "/data/mguevaral/crop_bbox/",
            dataset=dataset,
            indices=val_indices,
            dim=input_size,
            batch_size=batch_size,
            positive_class=args.phenotype,
        )

        model = CNN3D(input_size[0], input_size[1], input_size[2]).__get_model__()
        # model = Resnet3DBuilder.build_resnet_50(
        #     (input_size[0], input_size[1], input_size[2], 1), 1)
        model_name = (
            "CNN_3D." +
            os.environ.get("SLURM_JOB_ID") + "." +
            phenotypes[args.phenotype] + f".fold_{fold_num}"
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=50),
            tf.keras.callbacks.ModelCheckpoint(
                filepath="/home/mguevaral/jpedro/phenotype-classifier/checkpoints/"
                + model_name
                + "/weights.h5",
                save_best_only=True,
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir="/home/mguevaral/jpedro/phenotype-classifier/logs/" + model_name
            ),
        ]
        initial_learning_rate = 1e-4
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
            tf.keras.metrics.AUC(name="auc"),
        ]
        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            metrics=metrics,
        )
        # Build the class weights structure
        class_weight = {0: train_generator.weight_for_0,
                        1: train_generator.weight_for_1}
        # Train the model
        if not args.notrain:
            model.fit(
                train_generator,
                batch_size=batch_size,
                epochs=num_epochs,
                validation_data=val_generator,
                verbose=2,
                callbacks=callbacks,
                class_weight=class_weight,
            )
        # Load the best model
        model.load_weights(
            "/home/mguevaral/jpedro/phenotype-classifier/checkpoints/"
            + model_name
            + "/weights.h5"
        )
        
        # Evaluate the model for the current fold
        score = model.evaluate(val_generator, verbose=2)
        print(f"Fold {fold_num} - Test loss:", score[0])
        print(f"Fold {fold_num} - Test accuracy:", score[1])
        model.save(
            "/home/mguevaral/jpedro/phenotype-classifier/checkpoints/" + model_name
        )

        # Predict
        y_prediction = model.predict(val_generator)

        # Create confusion matrix and normalizes it over predicted (columns)
        cf_matrix = confusion_matrix(
            dataset.y_test, y_prediction.argmax(axis=-1), normalize='pred')
        print(cf_matrix)
