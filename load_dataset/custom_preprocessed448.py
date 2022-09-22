import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm


class CustomPreProcessed448:
    def __init__(self, path, crop_size=(448, 448)):
        self._path = path
        self._crop_size = crop_size

    def read_dataset(self):
        imgs, classes = [], []

        # Read class csv file
        classes_csv = pd.read_csv(os.path.join(self._path, 'classes.csv'))

        # Iterate over all the images
        os.chdir(self._path)
        for file in tqdm(glob.glob("*.jpg")):
            # Read the image and resize it
            img = Image.open(os.path.join(self._path, file))
            img = img.resize(self._crop_size)
            img = np.array(img)

            # Get the case number of the image
            match = re.search("Breast_MRI_[0-9]+", file)
            case = file[match.start():match.end()]
            if case is None:
                continue

            # Match the case number with the phenotype class
            phenotype = classes_csv.loc[classes_csv["patient_id"] == case]["patient_id"]
            if phenotype is None:
                continue
            else:
                imgs.append(img)
                classes.append(phenotype)

        # Separate in train and test split
        # TODO: check keras preprocessing functions
        print("Spliting in train and test splits...")
        return train_test_split(imgs, classes, test_size=0.1, random_state=123, shuffle=True)


if __name__ == "__main__":
    reader = CustomPreProcessed448(sys.argv[1], sys.argv[2])
    reader.read_dataset()
