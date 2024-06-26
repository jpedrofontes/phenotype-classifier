import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from load_dataset.slice_dataset_2d import Slice_Dataset_2D
from models.custom_model import CustomModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--arch', help='model architecture')
    parser.add_argument('--sizes', type=int, nargs='+',
                        help='sizes of the internal layers (only available if model type not specified)')
    parser.add_argument('--verbose', help='verbose mode', action='store_true')
    args = parser.parse_args()

    num_classes = 2
    img_size = (224, 224)
    input_shape = (224, 224, 3)
    verbose = 1 if args.verbose else 2
    batch_size = 128
    epochs = 500
    lr = 1e-2

    if args.arch == "resnet152":
        inputs = tf.keras.layers.Input(shape=input_shape)
        # x = img_augmentation(inputs)
        base_model = tf.keras.applications.ResNet152V2(
            include_top=False, weights="imagenet", classes=num_classes)(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model)
        x = tf.keras.layers.BatchNormalization()(x)
        top_dropout_rate = 0.2
        x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = tf.keras.layers.Dense(
            units=1, activation="sigmoid", name="pred")(x)
        model = tf.keras.Model(inputs=inputs,
                               outputs=outputs, name="ResNet152V2")
        model_name = "ResNet152V2" + "." + os.environ.get("SLURM_JOB_ID")

    elif args.arch == "efficientnet":
        inputs = tf.keras.layers.Input(shape=input_shape)
        # x = img_augmentation(inputs)
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", classes=num_classes)(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model)
        x = tf.keras.layers.BatchNormalization()(x)
        top_dropout_rate = 0.2
        x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = tf.keras.layers.Dense(
            units=1, activation="sigmoid", name="pred")(x)
        model = tf.keras.Model(inputs=inputs,
                               outputs=outputs, name="EfficientNet")
        model_name = "EfficientNet" + "." + os.environ.get("SLURM_JOB_ID")

    else:
        if args.sizes is None:
            logging.fatal("no internal layers are specified")
            sys.exit()
        model = CustomModel(args.sizes, input_shape, num_classes).make_model()
        model_name = "CustomModel_" + \
            '_'.join(map(str, args.sizes)) + "." + \
            os.environ.get("SLURM_JOB_ID")

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
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=metrics,
    )
    print(model.summary())

    dataset = Slice_Dataset_2D(
        "/data/mguevaral/crop_bbox", img_size, num_classes=num_classes)
    train_generator, class_weights = dataset.get_dataset_generator(
        batch_size=batch_size)
    test_generator, _ = dataset.get_dataset_generator(training=False)

    model.load_weights(
        "/home/mguevaral/jpedro/phenotype-classifier/old_2d/checkpoints/" + model_name + "/weights.h5")
    score = model.evaluate(test_generator, verbose=verbose)
    print("Model:", model_name)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # Predict
    y_prediction = model.predict(dataset.x_test)

    # Create confusion matrix and normalizes it over predicted (columns)
    cf_matrix = confusion_matrix(
        dataset.y_test, y_prediction.argmax(axis=-1), normalize='pred')
    print(cf_matrix)
    ax = sn.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Confusion Matrix with labels')
    ax.set_ylabel('Predicted Values')
    ax.set_xlabel('Actual Values')

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Other', 'Luminal A'])
    ax.yaxis.set_ticklabels(['Other', 'Luminal A'])

    # Display the visualization of the Confusion Matrix.
    plt.savefig(
        '/home/mguevaral/jpedro/phenotype-classifier/old_2d/logs/' + model_name + '/cf_matrix.png')
