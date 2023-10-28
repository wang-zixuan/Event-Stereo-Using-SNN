import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
import os
import numpy as np
from models.stereo_matching import StereoMatching
from copy import deepcopy
from data.transformers import init_transformers
import PIL
import matplotlib.pyplot as plt
import math


def transformer(left, right):
    data_transformers = init_transformers(number_of_splits=1)
    data = {
        'left': {
            'event_sequence': left
        },
        'right': {
            'event_sequence': right
        }
    }

    for self_transformer in data_transformers:
        data = self_transformer(data)
    return data


def get_image(image_path):
    # Not all examples have images.
    if not os.path.isfile(image_path):
        return np.zeros((260, 346), dtype=np.uint8)
    return np.array(PIL.Image.open(image_path)).astype(np.uint8)


def get_disparity_image(disparity_image_path):
    disparity_image = get_image(disparity_image_path)
    invalid_disparity = (disparity_image == 255)
    # 0 ~ 36, inf
    disparity_image = (disparity_image / 7.)

    disparity_image[invalid_disparity] = float('inf')
    return disparity_image


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
    matrix_numpy[~noninf_mask] = maximum_value
    plot = plt.imshow(
        matrix_numpy, colormap, vmin=minimum_value, vmax=maximum_value)

    plot.axes.get_xaxis().set_visible(False)
    plot.axes.get_yaxis().set_visible(False)
    figure.savefig(filename, bbox_inches='tight', dpi=200)
    plt.close()


def main():
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = StereoMatching().to(device)
    param_key = 'params'
    
    load_path = '/mvsec/results/experiments/event_stereo_dense_split3_decay06/models/net_6_9100.pth'
    
    if load_path is not None:
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            model = model.module
        load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            load_net = load_net[param_key]
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        model.load_state_dict(load_net, strict=True)

    split_number = 3
    frame_index = 1700

    input_events_left = np.load(f'/data1/wzx/mvsec/dataset/indoor_flying_{split_number}/event0/00{frame_index}.npy')
    input_events_right = np.load(f'/data1/wzx/mvsec/dataset/indoor_flying_{split_number}/event1/00{frame_index}.npy')

    disparity_gt = torch.from_numpy(get_disparity_image(f'/data1/wzx/mvsec/dataset/indoor_flying_{split_number}/disparity_image/00{frame_index}.png')).float()

    transformed_events = transformer(input_events_left, input_events_right)

    model.eval()
    with torch.no_grad():
        output = model(transformed_events['left']['event_sequence'].to(device).unsqueeze(0), transformed_events['right']['event_sequence'].to(device).unsqueeze(0), is_train=False).view(260, 346).detach().cpu()

        gt = disparity_gt.view(260, 346).detach().cpu()
        noninf_mask = ~torch.isinf(gt)
        minimum_disparity = gt.min()
        maximum_disparity = gt[noninf_mask].max()

        save_matrix(
            filename=f'gt_{split_number}_{frame_index}.png',
            matrix=gt,
            minimum_value=minimum_disparity,
            maximum_value=maximum_disparity,
            colormap='jet',
        )

        # 估计出的disparity image，是否考虑到gt的inf的情况
        save_matrix(
            filename=f'no_inf_{split_number}_{frame_index}.png',
            matrix=output,
            minimum_value=minimum_disparity,
            maximum_value=maximum_disparity,
            colormap='jet',
        )
        output[~noninf_mask] = float('inf')
        save_matrix(
            filename=f'inf_{split_number}_{frame_index}.png',
            matrix=output,
            minimum_value=minimum_disparity,
            maximum_value=maximum_disparity,
            colormap='jet',
        )


if __name__ == '__main__':
    main()
