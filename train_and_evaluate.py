import argparse
import os

import tensorflow as tf

from load_dataset.generator_3d import DataGenerator
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

    input_size = (64, 128, 128)
    batch_size = 4
    epochs = 100

    train_generator = DataGenerator(
        '/data/mguevaral/crop_bbox/', stage='train', dim=input_size, batch_size=batch_size, positive_class=args.phenotype)
    test_generator = DataGenerator(
        '/data/mguevaral/crop_bbox/', stage='test', dim=input_size, batch_size=batch_size, positive_class=args.phenotype)

    model = CNN3D(input_size[0], input_size[1], input_size[2]).__get_model__()
    model_name = "CNN3D." + os.environ.get("SLURM_JOB_ID") + "." + phenotypes[args.phenotype] 

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=25),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="/home/mguevaral/jpedro/phenotype-classifier/checkpoints/" + model_name + "/weights.h5", save_best_only=True),
        tf.keras.callbacks.TensorBoard(
            log_dir="/home/mguevaral/jpedro/phenotype-classifier/logs/" + model_name),
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
    ]
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=metrics,
    )

    model.fit(train_generator,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=test_generator,
              verbose=2,
              callbacks=callbacks)

    model.load_weights(
        "/home/mguevaral/jpedro/phenotype-classifier/checkpoints/" + model_name + "/weights.h5")
    score = model.evaluate(test_generator, verbose=2)
    print("Model:", model_name)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    model.save(
        "/home/mguevaral/jpedro/phenotype-classifier/checkpoints/" + model_name)
