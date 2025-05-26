import os
import pandas as pd
import numpy as np
from PIL import Image

class Dataset():
    """
    A base class for datasets with shared functionality.

    Attributes:
        base_path : str
            The base directory path where the dataset is stored.
        phenotype_map : dict
            A mapping of class IDs to phenotype names.
        crop_size : tuple
            The dimensions to which each image/volume will be resized.
        annotations : dict
            A dictionary to store information about each image/volume and its metadata.

    Methods:
        __init__(self, base_path, phenotype_map, crop_size=(128, 128)):
            Initializes the BaseDataset object with the given base path, phenotype map, and crop size.

        read_image(self, image_path):
            Reads and resizes an image.

        normalize_image(self, img):
            Normalizes an image to the range [0, 1].

        process_image(self, image_path):
            Reads, resizes, and normalizes an image.
    """
    
    def __init__(self, base_path, phenotype_map, crop_size=(128, 128)):
        self.base_path = base_path
        self.phenotype_map = phenotype_map
        self.crop_size = crop_size
        self.annotations = {}

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
