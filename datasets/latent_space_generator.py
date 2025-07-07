import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf


class LatentSpaceGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        csv_file,
        batch_size=32,
        shuffle=True,
        positive_class=0,
    ):
        self.data = pd.read_csv(csv_file)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data))
        self.positive_class = positive_class
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        batch_data = self.data.iloc[batch_indexes]

        X = batch_data.iloc[:, :-1].values
        phenotypes = batch_data.iloc[:, -1].values
        y = np.array(
            [1 if phenotype == self.positive_class else 0 for phenotype in phenotypes]
        )

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
