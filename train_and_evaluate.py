import argparse
import logging
import os
import sys

import numpy as np
import tensorflow as tf

from load_dataset.custom_preprocessed448 import CustomPreProcessed448
from models.custom_model import CustomModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--arch', help='model architecture')
    parser.add_argument('--sizes', type=int, nargs='+',
                        help='sizes of the internal layers (only available if model type not specified)')
    parser.add_argument('--verbose', help='verbose mode', action='store_true')
    args = parser.parse_args()

    num_classes = 4
    img_size = (224, 224)
    input_shape = (224, 224, 3)
    verbose = 1 if args.verbose else 2
    batch_size = 64
    epochs = 500
    lr = 1e-6
    decay = lr / epochs

    if args.arch is None:
        if args.sizes is None:
            logging.fatal("no internal layers are specified")
            sys.exit()
        model = CustomModel(args.sizes, input_shape, num_classes).make_model()
        model_name = "CustomModel_" + \
            '_'.join(map(str, args.sizes)) + "." + \
            os.environ.get("SLURM_JOB_ID")
    elif args.arch == "resnet50":
        base_model = tf.keras.applications.ResNet50V2(
            include_top=False, input_shape=input_shape)
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=x)
        model_name = "ResNet50V2" + "." + os.environ.get("SLURM_JOB_ID")
    elif args.arch == "resnet101":
        base_model = tf.keras.applications.ResNet101V2(
            include_top=False, input_shape=input_shape)
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=x)
        model_name = "ResNet101V2" + "." + os.environ.get("SLURM_JOB_ID")
    elif args.arch == "resnet152":
        base_model = tf.keras.applications.ResNet152V2(
            include_top=False, input_shape=input_shape)
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=x)
        model_name = "ResNet152V2" + "." + os.environ.get("SLURM_JOB_ID")
    elif args.arch == "inception":
        base_model = tf.keras.applications.InceptionV3(
            include_top=False, input_shape=input_shape)
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=x)
        model_name = "InceptionV3" + "." + os.environ.get("SLURM_JOB_ID")
    elif args.arch == "mobilenet":
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False, input_shape=input_shape)
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=x)
        model_name = "MobileNetV2" + "." + os.environ.get("SLURM_JOB_ID")
    else:
        logging.fatal("wrong architecture name")
        sys.exit()

    dataset = CustomPreProcessed448(
        "/data/mguevaral/crop_bbox", img_size, num_classes=num_classes)
    train_generator = dataset.get_dataset_generator(batch_size=batch_size)
    test_generator = dataset.get_dataset_generator(training=False)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=50),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="/home/mguevaral/jpedro/phenotype-classifier/checkpoints/" + model_name + "/weights.h5", save_best_only=True),
        tf.keras.callbacks.TensorBoard(
            log_dir="/home/mguevaral/jpedro/phenotype-classifier/logs/" + model_name),
        tf.keras.callbacks.CSVLogger(
            "/home/mguevaral/jpedro/phenotype-classifier/logs/" + model_name + "/log.csv", separator=",", append=False),
    ]

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer, metrics=["accuracy"], )

    model.fit(train_generator,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=test_generator,
              verbose=verbose,
              callbacks=callbacks)

    model.load_weights(
        "/home/mguevaral/jpedro/phenotype-classifier/checkpoints/" + model_name + "/weights.h5")
    score = model.evaluate(test_generator, verbose=verbose)
    print("Model:", model_name)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    model.save(
        "/home/mguevaral/jpedro/phenotype-classifier/checkpoints/" + model_name)
