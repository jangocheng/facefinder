# coding=utf-8
"""Load images dataset"""
from __future__ import division
import os
import config
from skimage import io
from skimage.transform import resize, rotate

from sklearn.model_selection import train_test_split


# Load the dataset for face/non face classification
def load_dataset():
    """
    Load the faces and nonfaces dataset
    Returns:
     Null
    """

    # Load faces (faces samples)
    image_numpy_data = []
    image_vector_data = []
    for folder in os.listdir(config.FACES_FOLDER_DIR):

        if os.path.isdir(os.path.join(config.FACES_FOLDER_DIR, folder)):

            for image in os.listdir(os.path.join(config.FACES_FOLDER_DIR, folder)):
                image_file = os.path.join(config.FACES_FOLDER_DIR, folder, image)

                if os.path.isfile(image_file):
                    img = resize(io.imread(image_file), config.IMAGE_DIMENSION)
                    image_numpy_data.append(img)
                    image_vector_data.append(config.FACES_VECTOR)

    # Load NONfaces (NONfaces samples)
    for folder in os.listdir(config.NON_FACES_FOLDER_DIR):

        if os.path.isdir(os.path.join(config.NON_FACES_FOLDER_DIR, folder)):

            for image in os.listdir(os.path.join(config.NON_FACES_FOLDER_DIR, folder)):
                image_file = os.path.join(config.NON_FACES_FOLDER_DIR, folder, image)

                if os.path.isfile(image_file):
                    img = resize(io.imread(image_file), config.IMAGE_DIMENSION)
                    image_numpy_data.append(img)
                    image_vector_data.append(config.NON_FACES_VECTOR)

    x_train, x_test, y_train, y_test = train_test_split(image_numpy_data, image_vector_data, test_size=0.4,
                                                        random_state=4)
    return x_train, x_test, y_train, y_test
