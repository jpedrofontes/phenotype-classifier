import os
import re

import cv2
import numpy as np
import pandas as pd

from .dataset import Dataset


class BCCollectionDataset(Dataset):
    """
    A class to represent a dataset with bounding box annotations and phenotype mapping.

    Attributes:
        base_path : str
            The base directory path where the dataset is stored.
        annotations_csv_path : str
            The path to the CSV file containing annotations for the dataset.
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

    def __init__(
        self,
        base_path,
        annotations_csv_path,
        crop_size=(128, 128),
        transformations=None,
    ):
        self.base_path = base_path
        self.crop_size = crop_size
        self.volumes = dict()
        self.trasnformations = transformations or []
        
        # Define transformations
        self.default_transformations = [
            # self.rotate_90,
            # self.rotate_180,
            # self.rotate_270,
        ]

        # Read the CSV file
        annotations_csv = pd.read_csv(annotations_csv_path)

        # Build dataset info
        for _, row in annotations_csv.iterrows():
            match = re.match(r"(P-.*-.+-.+-.+)-.*\.jpg", row["Basename"])
            
            if match:
                basename = match.group(1)
            else:
                continue
            
            class_id = row["SubtypeID"]
            slice_bbox = {
                "xmin": int(row["Xmin"]),
                "ymin": int(row["Ymin"]),
                "xmax": int(row["Xmax"]),
                "ymax": int(row["Ymax"]),
                "path": os.path.join(self.base_path, row["Basename"]),
            }

            if basename not in self.volumes:
                self.volumes[basename] = {
                    "case": basename,
                    "slices": [],
                    "phenotype": class_id,
                    "transformations": [],
                }

            self.volumes[basename]["slices"].append(slice_bbox)
            self.volumes[basename]["slices"].sort(key=lambda x: x["path"])

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
                cropped_image, self.crop_size[1:], interpolation=cv2.INTER_LINEAR
            )

            # Add to volume
            volume.append(resized_image)

        return np.stack(volume, axis=0)
