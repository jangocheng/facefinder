# coding=utf-8
"""Load images dataset"""
from __future__ import division
import os
import numpy as np
import config
from skimage import io
from skimage.transform import resize, rotate
from skimage.color import rgb2gray
from helper.path import list_images
from sklearn.model_selection import train_test_split


# Load the dataset for face/non face classification
def load_dataset():
    """
    Load the faces and nonfaces dataset
    Returns:
     Null
    """

    image_numpy_data = []
    image_vector_data = []

    for image_path in list_images(config.FACES_FOLDER_DIR):
        try:
            img = resize(io.imread(image_path), config.IMAGE_DIMENSION)
            image_numpy_data.append(img.flatten().astype(np.float32))
            image_vector_data.append(config.FACES_VECTOR)

        except IOError:
            print("Image type not found {}".format(image_path))

    for image_path in list_images(config.NON_FACES_FOLDER_DIR):
        try:
            img = resize(io.imread(image_path), config.IMAGE_DIMENSION)
            image_numpy_data.append(img.flatten().astype(np.float32))
            image_vector_data.append(config.NON_FACES_VECTOR)

        except IOError:
            print("Image type not found {}".format(image_path))

    image_numpy_data = np.asarray(image_numpy_data)
    image_vector_data = np.asarray(image_vector_data)

    x_train, x_test, y_train, y_test = train_test_split(image_numpy_data, image_vector_data, test_size=0.7,
                                                        random_state=4)
    return x_train, x_test, y_train, y_test
