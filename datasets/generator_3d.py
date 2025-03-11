import sys

import tensorflow as tf
import numpy as np

from datasets.dataset_3d import Dataset_3D


class DataGenerator(tf.keras.utils.Sequence):
    """
    DataGenerator is a custom data generator class that inherits from tf.keras.utils.Sequence.
    It is used to generate batches of 3D data for training, validation, or testing purposes.

    Attributes:
        dim (tuple): Dimensions of the 3D data (depth, width, height).
        batch_size (int): Number of samples per batch.
        path (str): Path to the dataset.
        dataset (Dataset_3D): Instance of the Dataset_3D class.
        positive_class (int): Class label considered as positive.
        shuffle (bool): Whether to shuffle the data after each epoch.
        autoencoder (bool): Whether the generator is used for an autoencoder.
        transformations (dict): Transformations to be applied to the data.
        volumes (list): List of volume keys for the current stage (train/validation/test).
        list_IDs (list): List of IDs corresponding to the volumes.
        indexes (np.ndarray): Array of indexes for shuffling.
        counts (list): List containing counts of negative and positive samples.
        weight_for_0 (float): Class weight for the negative class.
        weight_for_1 (float): Class weight for the positive class.

    Methods:
        __init__(self, path, dataset=None, indices=None, stage="train", batch_size=32, dim=(64, 128, 128), positive_class=0, shuffle=True, autoencoder=False):
            Initializes the DataGenerator instance.

        set_transformations(self, transformations):
            Sets the transformations to be applied to the data.

        __len__(self):
            Returns the number of batches per epoch.

        __getitem__(self, index):
            Generates one batch of data.

        on_epoch_end(self):
            Updates indexes after each epoch.

        __data_generation(self, list_IDs_temp):
            Generates data containing batch_size samples.

        __class_weights(self):
            Calculates class weights based on the current fold's volumes.
    """

    def __init__(
        self,
        path,
        dataset=None,
        indices=None,
        stage="train",
        batch_size=32,
        dim=(64, 128, 128),
        positive_class=0,
        shuffle=True,
        autoencoder=False,
    ):
        self.dim = dim
        self.batch_size = batch_size
        self.path = path

        if dataset is None or not isinstance(dataset, Dataset_3D):
            self.dataset = Dataset_3D(path, crop_size=dim)
        else:
            self.dataset = dataset

        self.positive_class = positive_class

        if indices is None:
            self.volumes = (
                list(self.dataset.volumes.keys())[
                    0 : int(0.9 * len(self.dataset.volumes))
                ]
                if stage == "train"
                else list(self.dataset.volumes.keys())[
                    int(0.9 * len(self.dataset.volumes)) : len(self.dataset.volumes)
                ]
            )
        else:
            all_volumes = list(self.dataset.volumes.keys())
            self.volumes = [all_volumes[i] for i in indices]

        self.list_IDs = [x for x in range(len(self.volumes))]
        self.shuffle = shuffle
        self.autoencoder = autoencoder
        self.transformations = None  # Initialize transformations as None
        self.__class_weights()
        self.on_epoch_end()

    def set_transformations(self, transformations):
        self.transformations = transformations

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data using the __data_generation() method
        X, y = self.__data_generation(list_IDs_temp)
        # Return the data
        if self.autoencoder:
            return X, y, X
        else:
            return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"
        # X : (n_samples, depth, width, height, n_channels)
        # y : 0/1 representing true or false of bellonging to a specific phenotype
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            volume_info = self.dataset.volumes[self.volumes[ID]]
            transformations = volume_info.get("transformations", None)
            X[i,] = self.dataset.process_scan(self.volumes[ID], transformations)
            # Store class
            phenotype = self.dataset.volumes[self.volumes[ID]]["phenotype"]

            if self.positive_class is not None:
                y[i] = 1 if phenotype == self.positive_class else 0
            else:
                y[i] = phenotype

        assert len(X) > 0, "Training data (X) is empty in generator"
        assert len(y) > 0, "Training labels (y) are empty in generator"

        return X, y

    def __class_weights(self):
        self.counts = [0, 0]

        # Calculate based on the current fold's volumes (self.volumes) instead of the entire dataset
        self.counts[0] = len(
            [
                x
                for x in self.volumes
                if self.dataset.volumes[x]["phenotype"] != self.positive_class
            ]
        )
        self.counts[1] = len(
            [
                x
                for x in self.volumes
                if self.dataset.volumes[x]["phenotype"] == self.positive_class
            ]
        )

        total = self.counts[0] + self.counts[1]
        self.weight_for_0 = (total / 2) / self.counts[0] if self.counts[0] != 0 else 0
        self.weight_for_1 = (total / 2) / self.counts[1] if self.counts[1] != 0 else 0


if __name__ == "__main__":
    generator = DataGenerator(sys.argv[1], positive_class=int(sys.argv[2]))
    print(
        f"\nNumber of batches (batch size {generator.batch_size}): {generator.__len__()}"
    )
    X, y = generator.__getitem__(0)
    print(f"Batch shape: {X.shape}")
    print(f"Positive cases: {generator.counts[1]} => Weight: {generator.weight_for_1}")
    print(f"Negative cases: {generator.counts[0]} => Weight: {generator.weight_for_0}")
