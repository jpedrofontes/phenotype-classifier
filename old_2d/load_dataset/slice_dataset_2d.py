import glob
import os
import random
import re
import sys

import numpy as np
import pandas as pd

from generator import DataGenerator

phenotypes = {
    0: "Luminal_A",
    1: "Luminal_B",
    2: "HER2_Enriched",
    3: "Triple_Negative"
}


class Slice_Dataset_2D:
    def __init__(self, path, crop_size=(448, 448), num_classes=1):
        self.path = path
        self._crop_size = crop_size
        self._num_classes = num_classes

        cases = dict()
        x_train, x_test, y_train, y_test = [], [], [], []

        # Read class csv file
        classes_csv = pd.read_csv(os.path.join(self.path, 'classes.csv'))

        # Iterate over all the images
        os.chdir(self.path)
        print("Reading dataset...")
        for file in glob.glob("*.jpg"):
            # Build image path
            img = os.path.join(self.path, file)

            # Get the case number of the image
            match = re.search("Breast_MRI_[0-9]+", file)
            case = file[match.start():match.end()]
            if case is None:
                continue

            # Match the case number with the phenotype class
            phenotype = classes_csv.loc[classes_csv["patient_id"]
                                        == case]["mol_subtype"]
            if phenotype is None:
                continue
            else:
                try:
                    cases[case]["images"].append(img)
                    cases[case]["phenotype"] = 1 if phenotype.tolist()[0] == 0 else 0
                except:
                    cases[case] = dict()
                    cases[case]["number"] = case
                    cases[case]["images"] = []
                    cases[case]["images"].append(img)
                    cases[case]["phenotype"] = 1 if phenotype.tolist()[0] == 0 else 0

        # Separate in train and test by case number
        print("Split training and test cases by case number...")
        train_size = int(0.9 * len(cases))
        train_cases, test_cases = [], []

        for i, case in enumerate(cases.values()):
            if i < train_size:
                train_cases.append(case)
            else:
                test_cases.append(case)

        for case in train_cases:
            for img in case["images"]:
                x_train.append(img)
                y_train.append(case["phenotype"])

        for case in test_cases:
            for img in case["images"]:
                x_test.append(img)
                y_test.append(case["phenotype"])

        print("Train class balance:", [
              [x, ":", y_train.count(x)] for x in set(y_train)])
        print("Test class balance: ", [
              [x, ":", y_test.count(x)] for x in set(y_test)])

        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        self.y_train = self.y_train.reshape((self.y_train.shape[0], 1))
        print("Train dataset shape:", self.x_train.shape, self.y_train.shape)
        self.x_test = np.array(x_test)
        self.y_test = np.array(y_test)
        self.y_test = self.y_test.reshape((self.y_test.shape[0], 1))
        print("Train dataset shape:", self.x_test.shape, self.y_test.shape)

        counts = np.bincount(self.y_train[:, 0])
        self.class_weight = dict()
        for i in range(num_classes):
            self.class_weight[i] = (1.0 / counts[i]) * \
                (self.y_train.shape[0] / 2)
        print(self.class_weight)

        print("Dataset ready!")

    def get_dataset_generator(self, training=True, batch_size=32):
        if training:
            return DataGenerator(self.x_train, self.y_train, path=self.path, batch_size=batch_size, dim=self._crop_size, n_classes=self._num_classes, shuffle=True), self.class_weight
        else:
            return DataGenerator(self.x_test, self.y_test, path=self.path, batch_size=batch_size, dim=self._crop_size, n_classes=self._num_classes, shuffle=False), None


if __name__ == "__main__":
    img_size = (224, 224)
    num_classes = 2
    dataset = Slice_Dataset_2D(
        sys.argv[1], img_size, num_classes=num_classes)
