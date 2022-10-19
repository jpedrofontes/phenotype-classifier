import argparse
import logging
import os
import sys

import numpy as np
import tensorflow as tf

from load_dataset.custom_preprocessed448 import CustomPreProcessed448
from models.custom_model import CustomModel


def scheduler(epoch, lr):
    return lr * 1 / (1 + decay * epoch)


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
    lr = 0.01
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

    x_train, x_test, y_train, y_test = CustomPreProcessed448(
        "/data/mguevaral/crop_bbox", img_size, verbose=verbose).read_dataset()
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    x_train, x_test = np.array(x_train), np.array(x_test)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=50),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="/home/mguevaral/jpedro/phenotype-classifier/checkpoints/" + model_name + "/weights.h5", save_best_only=True),
        tf.keras.callbacks.TensorBoard(
            log_dir="/home/mguevaral/jpedro/phenotype-classifier/logs/" + model_name),
        tf.keras.callbacks.LearningRateScheduler(scheduler),
        tf.keras.callbacks.CSVLogger(
            "/home/mguevaral/jpedro/phenotype-classifier/logs/" + model_name + "/log.csv", separator=",", append=False),
    ]

    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(
        learning_rate=lr), metrics=["accuracy"], )

    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs, validation_split=0.1, verbose=verbose, callbacks=callbacks)

    model.load_weights(
        "/home/mguevaral/jpedro/phenotype-classifier/checkpoints/" + model_name + "/weights.h5")
    score = model.evaluate(x_test, y_test, verbose=verbose)
    print("Model:", model_name)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    model.save(
        "/home/mguevaral/jpedro/phenotype-classifier/checkpoints/" + model_name)
