# coding=utf-8
"""helper file for path function"""
import os


def list_images(base_path, contains=None):
    """
    return the set of files that are valid
    Args:
        base_path: 
        contains: 

    Returns:

    """

    return list_files(base_path, valid_ext=(".jpg", ".jpeg", ".png", ".bmp", ".ppm"), contains=contains)


def list_files(base_path, valid_ext=(".jpg", ".jpeg", ".png", ".bmp"), contains=None):
    """
    loop over the directory structure
    Args:
        base_path: 
        valid_ext: 
        contains: 

    Returns:

    """

    for (root_dir, dir_names, file_names) in os.walk(base_path):
        # loop over the file_names in the current directory
        for filename in file_names:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(valid_ext):
                # construct the path to the image and yield it
                image_path = os.path.join(root_dir, filename).replace(" ", "\\ ")
                yield image_path
