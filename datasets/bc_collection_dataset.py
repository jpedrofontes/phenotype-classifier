import os

import cv2
import numpy as np
import pandas as pd

from dataset import Dataset


class BCCollectionDataset(Dataset):
    """
    A class to represent a dataset with bounding box annotations and phenotype mapping.

    Attributes:
        base_path : str
            The base directory path where the dataset is stored.
        phenotype_map : dict
            A mapping of class IDs to phenotype names.
        crop_size : tuple
            The dimensions to which each image will be resized (width, height).
        annotations : dict
            A dictionary to store information about each image and its bounding boxes.

    Methods:
        __init__(self, base_path, phenotype_map, crop_size=(128, 128)):
            Initializes the CustomDataset object with the given base path, phenotype map, and crop size.

        read_image(self, basename):
            Reads and resizes an image given its basename.

        get_bounding_boxes(self, basename):
            Retrieves bounding box information for a given image.

        process_image(self, basename):
            Reads, resizes, and normalizes an image along with its bounding boxes.
    """

    def __init__(self, base_path, phenotype_map, crop_size=(128, 128)):
        self.base_path = base_path
        self.phenotype_map = phenotype_map
        self.crop_size = crop_size
        self.volumes = {}

        # Read the CSV file
        annotations_csv = pd.read_csv(os.path.join(self.base_path, "annotations.csv"))

        # Build dataset info
        for _, row in annotations_csv.iterrows():
            basename = row["basename"] 
            # match basename with (P-.*-[0-9]{6}-[0-9]{3})-.*\.jpg to get the volume key
            class_id = row["class_id"]
            slice_bbox = {
                "xmin": row["xmin"],
                "ymin": row["ymin"],
                "xmax": row["xmax"],
                "ymax": row["ymax"],
                "path": os.path.join(self.base_path, basename),
            }

            if basename not in self.volumes:
                self.volumes[basename] = {
                    "case": "-".join(basename.split("-")[:2]),  
                    "slices": [],
                    "phenotype": class_id
                }
                
            self.volumes[basename]["slices"].append(slice_bbox)

    def read_volume(self, volume_name):
        if volume_name not in self.volumes:
            raise ValueError(f"Volume {volume_name} not found in annotations.")

        volume_slices = self.volumes[volume_name]["slices"]
        volume = []

        for slice_info in volume_slices:
            # Read the image
            image = cv2.imread(slice_info["path"], cv2.IMREAD_UNCHANGED)
            
            if image is None:
                raise FileNotFoundError(f"Image {slice_info['path']} not found.")

            # Crop the image using bounding box coordinates
            xmin, ymin, xmax, ymax = (
                slice_info["xmin"],
                slice_info["ymin"],
                slice_info["xmax"],
                slice_info["ymax"],
            )
            cropped_image = image[ymin:ymax, xmin:xmax]

            # Resize the cropped image to the desired crop size
            resized_image = cv2.resize(
                cropped_image, self.crop_size, interpolation=cv2.INTER_LINEAR
            )

            # Add to volume
            volume.append(resized_image)

        return np.stack(volume, axis=0)
