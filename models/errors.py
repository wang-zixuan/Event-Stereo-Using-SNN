import torch


def compute_absolute_error(estimated_disparity,
                           ground_truth_disparity,
                           use_mean=True):
    absolute_difference = (estimated_disparity - ground_truth_disparity).abs()
    locations_without_ground_truth = torch.isnan(ground_truth_disparity)
    pixelwise_absolute_error = absolute_difference.clone()
    pixelwise_absolute_error[locations_without_ground_truth] = 0
    absolute_differece_with_ground_truth = absolute_difference[~locations_without_ground_truth]
    
    if absolute_differece_with_ground_truth.numel() == 0:
        average_absolute_error = 0.0
    else:
        if use_mean:
            average_absolute_error = absolute_differece_with_ground_truth.mean().item()
        else:
            average_absolute_error = absolute_differece_with_ground_truth.median().item()
    return pixelwise_absolute_error, average_absolute_error


def compute_n_pixels_error(estimated_disparity, ground_truth_disparity, n=3.0):
    locations_without_ground_truth = torch.isnan(ground_truth_disparity)
    more_than_n_pixels_absolute_difference = (
        estimated_disparity - ground_truth_disparity).abs().gt(n).float()
    pixelwise_n_pixels_error = more_than_n_pixels_absolute_difference.clone()
    pixelwise_n_pixels_error[locations_without_ground_truth] = 0.0
    more_than_n_pixels_absolute_difference_with_ground_truth = \
        more_than_n_pixels_absolute_difference[~locations_without_ground_truth]
    if more_than_n_pixels_absolute_difference_with_ground_truth.numel() == 0:
        percentage_of_pixels_with_error = 0.0
    else:
        percentage_of_pixels_with_error = \
            more_than_n_pixels_absolute_difference_with_ground_truth.mean(
                ).item() * 100
    return pixelwise_n_pixels_error, percentage_of_pixels_with_error
