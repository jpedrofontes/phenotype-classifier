import argparse
import os

import numpy as np
import tensorflow as tf
from resnet3d import Resnet3DBuilder
from sklearn.metrics import confusion_matrix

from load_dataset.dataset_3d import Dataset_3D
from load_dataset.generator_3d import DataGenerator
from models.cnn3d import CNN3D

print(tf.__version__)