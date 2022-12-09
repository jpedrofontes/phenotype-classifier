import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

phenotypes = {
    0: "Luminal-like",
    1: "ER/PR pos, HER2 pos",
    2: "ER & PR neg, HER2 pos",
    3: "Triple Negative"
}


class Dataset_3D:
    def __init__(self, base_path, crop_size=(64, 128, 128)):
        self.base_path = base_path
        self.crop_size = crop_size
        self.volumes = dict()
        # Read class csv file
        classes_csv = pd.read_csv(os.path.join(self.base_path, 'classes.csv'))
        # Build dataset info
        os.chdir(base_path)
        print("Reading dataset...")
        for file in glob.glob("*.jpg"):
            # Get the case number of the image
            match_case = re.search("Breast_MRI_[0-9]+", file)
            case = file[match_case.start():match_case.end()]
            if case is None:
                print("will continue case none")
                continue
            # Get the series of the image
            match_series = re.search("_series#.+#_", file)
            series = file[match_series.start():match_series.end()]
            if series is None:
                print("will continue series none")
                continue
            # Save volume info
            volume_name = '{}.{}'.format(case, series)
            # Match the case number with the phenotype class
            phenotype = classes_csv.loc[classes_csv["patient_id"]
                                        == case]["mol_subtype"]
            if phenotype is None:
                continue
            else:
                phenotype = phenotype.tolist()[0]
                try:
                    self.volumes[volume_name]["case"] = case
                    self.volumes[volume_name]["slices"].append(
                        os.path.join(base_path, file))
                    self.volumes[volume_name]["slices"].sort()
                    self.volumes[volume_name]["phenotype"] = phenotype
                except:
                    self.volumes[volume_name] = dict()
                    self.volumes[volume_name]["case"] = case
                    self.volumes[volume_name]["slices"] = [
                        os.path.join(base_path, file)]
                    self.volumes[volume_name]["phenotype"] = phenotype

    def read_volume(self, volume_name):
        volume = []
        for img_path in self.volumes[volume_name]["slices"]:
            # Read the image
            img = Image.open(img_path)  # .convert('RGB')
            img = img.resize((self.crop_size[0], self.crop_size[1]))
            img = np.array(img)
            # Add to volume
            volume.append(img)
        volume = np.array(volume)
        return volume

    def normalize(self, volume):
        """Normalize the volume"""
        min = -1000
        max = 400
        volume[volume < min] = min
        volume[volume > max] = max
        volume = (volume - min) / (max - min)
        volume = volume.astype("float32")
        return volume

    def resize_volume(self, volume, desired_width=128, desired_height=128, desired_depth=64):
        """Resize across z-axis"""
        # Get current depth
        current_depth = volume.shape[0]
        current_width = volume.shape[1]
        current_height = volume.shape[2]
        # Compute depth factor
        depth = current_depth / desired_depth
        width = current_width / desired_width
        height = current_height / desired_height
        depth_factor = 1 / depth
        width_factor = 1 / width
        height_factor = 1 / height
        # Resize across z-axis
        volume = ndimage.zoom(
            volume, (depth_factor, width_factor, height_factor), order=1)
        return volume

    def process_scan(self, key):
        """Read and resize volume"""
        # Read scan
        volume = self.read_volume(key)
        # Normalize
        volume = self.normalize(volume)
        # Resize width, height and depth
        volume = self.resize_volume(
            volume, desired_depth=self.crop_size[0], desired_width=self.crop_size[1], desired_height=self.crop_size[2])
        return volume
