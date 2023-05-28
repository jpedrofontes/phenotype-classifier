import os

import numpy as np
import keras
from PIL import Image


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, images, labels, path, batch_size=32, dim=(224, 224), n_channels=3,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.path = path
        self.dim = dim
        self.batch_size = batch_size
        self.images = images
        self.labels = labels
        self.list_IDs = [x for x in range(len(images))]
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, 224, 224, 3))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Read the image and resize it
            img = Image.open(os.path.join(
                self.path, self.images[i])).convert('RGB')
            img = img.resize((224, 224))
            img = np.array(img)

            # Preprocess the image
            img = img/255

            # Store sample
            X[i, ] = img

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
