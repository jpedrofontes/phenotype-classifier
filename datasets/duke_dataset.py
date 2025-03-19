import glob
import os
import re

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage


class DukeDataset:
    """
    A class to represent a 3D medical imaging dataset.

    Attributes:
        base_path : str
            The base directory path where the dataset is stored.
        crop_size : tuple
            The dimensions to which each volume will be resized (depth, width, height).
        volumes : dict
            A dictionary to store information about each volume in the dataset.
        transformations : list
            A list of transformations to be applied to the volumes.
        default_transformations : list
            A list of default transformations (currently commented out).

    Methods:
        __init__(self, base_path, crop_size=(64, 128, 128), transformations=None):
            Initializes the Dataset_3D object with the given base path, crop size, and transformations.

        read_volume(self, volume_name):
            Reads and resizes a volume given its name.

        normalize(self, volume):
            Normalizes the volume to a specified range.

        resize_volume(self, volume, desired_width=128, desired_height=128, desired_depth=64):
            Resizes the volume to the desired dimensions.

        apply_transformations(self, volume, transformations):
            Applies a list of transformations to the volume.

        process_scan(self, key, transformations=None):
            Reads, normalizes, resizes, and applies transformations to a volume.

        rotate_90(self, volume):
            Rotates the volume by 90 degrees.

        rotate_180(self, volume):
            Rotates the volume by 180 degrees.

        rotate_270(self, volume):
            Rotates the volume by 270 degrees.
    """
    def __init__(self, base_path, crop_size=(64, 128, 128), transformations=None):
        self.base_path = base_path
        self.crop_size = crop_size
        self.volumes = dict()
        self.transformations = transformations

        # Define transformations
        self.default_transformations = [
            # self.rotate_90,
            # self.rotate_180,
            # self.rotate_270,
        ]

        # Read class csv file
        classes_csv = pd.read_csv(os.path.join(self.base_path, "classes.csv"))

        # Build dataset info
        os.chdir(base_path)
        print("Reading dataset...")

        for file in glob.glob("*.jpg"):
            # Get the case number of the image
            match_case = re.search("Breast_MRI_[0-9]+", file)
            case = file[match_case.start() : match_case.end()]

            if case is None:
                print(f"Skipping file {file}: case not found")
                continue

            # Get the series of the image
            match_series = re.search("_series#.+#_", file)
            series = file[match_series.start() : match_series.end()]

            if series is None:
                print(f"Skipping file {file}: series not found")
                continue

            # Save volume info
            volume_name = f"{case}.{series}"

            # Match the case number with the phenotype class
            phenotype = classes_csv.loc[classes_csv["patient_id"] == case][
                "mol_subtype"
            ]

            if phenotype.empty:
                print(f"Skipping file {file}: phenotype not found")
                continue

            phenotype = phenotype.iloc[0]

            if volume_name not in self.volumes:
                self.volumes[volume_name] = {
                    "case": case,
                    "slices": [],
                    "phenotype": phenotype,
                    "transformations": [],
                }

            self.volumes[volume_name]["slices"].append(os.path.join(base_path, file))
            self.volumes[volume_name]["slices"].sort()

        # Add entries for each transformation
        for volume_name in list(self.volumes.keys()):
            for transform in self.default_transformations:
                transformed_volume_name = f"{volume_name}_{transform.__name__}"
                self.volumes[transformed_volume_name] = {
                    "case": self.volumes[volume_name]["case"],
                    "slices": self.volumes[volume_name]["slices"],
                    "phenotype": self.volumes[volume_name]["phenotype"],
                    "transformations": [transform],
                }

    def read_volume(self, volume_name):
        """Read and resize volume"""
        volume = []

        for img_path in self.volumes[volume_name]["slices"]:
            # Read the image
            img = Image.open(img_path)
            img = img.resize((self.crop_size[1], self.crop_size[2]))
            img = np.array(img)

            # Add to volume
            volume.append(img)

        volume = np.array(volume)

        return volume

    def normalize(self, volume):
        """Normalize the volume"""
        min_val = -1000
        max_val = 400
        volume = np.clip(volume, min_val, max_val)
        volume = (volume - min_val) / (max_val - min_val)
        volume = volume.astype("float32")

        return volume

    def resize_volume(
        self, volume, desired_width=128, desired_height=128, desired_depth=64
    ):
        """Resize across z-axis"""
        # Get current depth, width, and height
        current_depth, current_width, current_height = volume.shape

        # Compute depth, width, and height factors
        depth_factor = desired_depth / current_depth
        width_factor = desired_width / current_width
        height_factor = desired_height / current_height

        # Resize across z-axis
        volume = ndimage.zoom(
            volume, (depth_factor, width_factor, height_factor), order=1
        )

        return volume

    def apply_transformations(self, volume, transformations):
        """Apply transformations to the volume"""
        if transformations is not None:
            for transform in transformations:
                volume = transform(volume)
        return volume

    def process_scan(self, key, transformations=None):
        """Read, normalize, resize, and apply transformations to volume"""
        # Read scan
        volume = self.read_volume(key)

        # Normalize
        volume = self.normalize(volume)

        # Resize width, height, and depth
        volume = self.resize_volume(
            volume,
            desired_depth=self.crop_size[0],
            desired_width=self.crop_size[1],
            desired_height=self.crop_size[2],
        )

        # Apply transformations
        volume = self.apply_transformations(volume, transformations)

        return volume

    def rotate_90(self, volume):
        """Rotate the volume by 90 degrees"""
        return ndimage.rotate(volume, 90, axes=(1, 2), reshape=False)

    def rotate_180(self, volume):
        """Rotate the volume by 180 degrees"""
        return ndimage.rotate(volume, 180, axes=(1, 2), reshape=False)

    def rotate_270(self, volume):
        """Rotate the volume by 270 degrees"""
        return ndimage.rotate(volume, 270, axes=(1, 2), reshape=False)
