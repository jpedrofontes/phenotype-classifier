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
    def __init__(self, path, crop_size=(448, 448), verbose=1):
        self._path = path
        self._crop_size = crop_size
        self._verbose = verbose

    def read_dataset(self):
        cases = dict()
        x_train, x_test, y_train, y_test = [], [], [], []

        # Read class csv file
        classes_csv = pd.read_csv(os.path.join(self._path, 'classes.csv'))

        # Iterate over all the images
        os.chdir(self._path)
        print("Reading dataset...")
        for file in glob.glob("*.jpg"):
            # Read the image and resize it
            img = Image.open(os.path.join(self._path, file)).convert('RGB')
            img = img.resize(self._crop_size)
            img = np.array(img)

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
                    cases[case]["phenotype"] = phenotype
                except:
                    cases[case] = dict()
                    cases[case]["images"] = []
                    cases[case]["images"].append(img)
                    cases[case]["phenotype"] = phenotype

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
                
        return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    reader = CustomPreProcessed448(sys.argv[1], sys.argv[2])
    reader.read_dataset()
