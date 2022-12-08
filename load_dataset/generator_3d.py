import sys

import keras
import numpy as np

from load_dataset.dataset_3d import Dataset_3D


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, path, stage="train", batch_size=32, dim=(64, 128, 128), positive_class=0, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.path = path
        self.dataset = Dataset_3D(path, crop_size=dim)
        self.positive_class = positive_class
        self.stage = stage
        self.volumes = list(self.dataset.volumes.keys())[0:int(0.9*len(self.dataset.volumes))] if stage == "train" else list(
            self.dataset.volumes.keys())[int(0.9*len(self.dataset.volumes)):len(self.dataset.volumes)]
        self.list_IDs = [x for x in range(len(self.volumes))]
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
        'Generates data containing batch_size samples'
        # X : (n_samples, depth, width, height, n_channels)
        # y : 0/1 representing true or false of bellonging to a specific phenotype
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, ] = self.dataset.process_scan(self.volumes[ID])
            # Store class
            phenotype = self.dataset.volumes[self.volumes[ID]]["phenotype"]
            y[i] = 1 if phenotype == self.positive_class else 0
        return X, y


if __name__ == "__main__":
    generator = DataGenerator(sys.argv[1], positive_class=0)
    print(
        f"\nNumber of batches (batch size {generator.batch_size}): {generator.__len__()}")
    X, y = generator.__getitem__(0)
    print(f"Batch shape: {X.shape}")