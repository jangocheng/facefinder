# coding=utf-8
"""Config file"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACES_FOLDER_DIR = "{}/data/faces".format(BASE_DIR)
NON_FACES_FOLDER_DIR = "{}/data/nonfaces".format(BASE_DIR)

IMAGE_DIMENSION = (32, 32)

FACES_VECTOR = [1, 0]
NON_FACES_VECTOR = [0, 1]

TOTAL_IMAGE_COUNT = 36000

MODEL_FOLDER = "{}/data/model".format(BASE_DIR)

MODEL_FILE = "{}/modelfile".format(MODEL_FOLDER)
