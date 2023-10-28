import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from data import dataset_constants
import data


def depth_to_disparity(depth_image):
    disparity_image = dataset_constants.FOCAL_LENGTH_X_BASELINE['indoor_flying'] / (depth_image + 1e-15)
    disparity_image[disparity_image > dataset_constants.DISPARITY_MAXIMUM] = dataset_constants.DISPARITY_MAXIMUM
    return disparity_image


def disparity_to_depth(disparity_image):
    unknown_disparity = disparity_image == float('inf')
    depth_image = \
        dataset_constants.FOCAL_LENGTH_X_BASELINE['indoor_flying'] / (
                disparity_image + 1e-7)
    depth_image[unknown_disparity] = float('inf')
    return depth_image


def save_matrix(filename,
                matrix,
                minimum_value=None,
                maximum_value=None,
                colormap='magma'
                ):

    figure = plt.figure()
    noninf_mask = matrix != float('inf')
    if minimum_value is None:
        minimum_value = np.quantile(matrix[noninf_mask], 0.001)
    if maximum_value is None:
        maximum_value = np.quantile(matrix[noninf_mask], 0.999)

    # set inf to 0
    matrix_numpy = matrix.numpy()
    matrix_numpy[~noninf_mask] = 0
    plot = plt.imshow(
        matrix_numpy, colormap, vmin=minimum_value, vmax=maximum_value)

    plot.axes.get_xaxis().set_visible(False)
    plot.axes.get_yaxis().set_visible(False)
    figure.savefig(filename, bbox_inches='tight', dpi=200)
    plt.close()
