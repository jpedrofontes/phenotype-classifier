import tensorflow as tf

from load_dataset.custom_preprocessed448 import CustomPreProcessed448
from models.custom_model import CustomModel

# This function keeps the initial learning rate for the first ten epochs
# # and decreases it exponentially after that.
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

if __name__ == "__main__":
    num_classes = 4
    img_size = (224, 224)
    input_shape = (224, 224, 3)

    x_train, x_test, y_train, y_test = CustomPreProcessed448(
        "/data/mguevaral/crop_bbox", img_size).read_dataset()
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    model = CustomModel([128, 256, 512], input_shape, num_classes).make_model()
    batch_size = 128
    epochs = 500
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="/home/mguevaral/jpedro/phenotype-classifier/checkpoints/CustomModel_128_256_512/model.{epoch: 02d}-{val_loss: .2f}.h5"),
        tf.keras.callbacks.TensorBoard(
            log_dir="/home/mguevaral/jpedro/phenotype-classifier/logs/CustomModel_128_256_512"),
        tf.keras.callbacks.LearningRateScheduler(scheduler),
        tf.keras.callbacks.CSVLogger(
            "/home/mguevaral/jpedro/phenotype-classifier/logs/CustomModel_128_256_512/log.csv", separator=",", append=False),
    ]

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"], callbacks=callbacks)

    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs, validation_split=0.1)
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
