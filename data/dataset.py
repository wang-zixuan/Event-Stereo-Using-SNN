import numpy as np
import os
import PIL
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from data import dataset_constants
import skimage.morphology as morpho
import h5py
import torchHED
import cv2
from copy import copy, deepcopy


FRAMES_FILTER_FOR_TRAINING = {
    'indoor_flying': {
        1: list(range(80, 1260)),
        2: list(range(160, 1580)),
        3: list(range(125, 1815)),
        4: list(range(190, 290))
    }
}


FRAMES_FILTER_FOR_TEST = {
    'indoor_flying': {
        1: list(range(140, 1201)),
        2: list(range(120, 1421)),
        3: list(range(73, 1616)),
        4: list(range(190, 290))
    }
}


class Network(nn.Module):
    """VGG-based network."""

    def __init__(self):
        super(Network, self).__init__()

        self.moduleVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.moduleCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )

        state_dict = torch.load('pytorch-hed-network.pt')
        self.load_state_dict(state_dict, strict=True)

    def forward(self, tensor_in):
        blue_in = (tensor_in[:, 0:1, :, :] * 255.0) - 104.00698793
        green_in = (tensor_in[:, 1:2, :, :] * 255.0) - 116.66876762
        red_in = (tensor_in[:, 2:3, :, :] * 255.0) - 122.67891434

        tensor_in = torch.cat([blue_in, green_in, red_in], 1)

        feature1 = self.moduleVggOne(tensor_in)
        feature2 = self.moduleVggTwo(feature1)
        feature3 = self.moduleVggThr(feature2)
        feature4 = self.moduleVggFou(feature3)
        feature5 = self.moduleVggFiv(feature4)

        score1 = self.moduleScoreOne(feature1)
        score2 = self.moduleScoreTwo(feature2)
        score3 = self.moduleScoreThr(feature3)
        score4 = self.moduleScoreFou(feature4)
        score5 = self.moduleScoreFiv(feature5)

        sc_size = (tensor_in.shape[2], tensor_in.shape[3])

        score1 = F.interpolate(
            input=score1, size=sc_size, mode='bilinear', align_corners=False)
        score2 = F.interpolate(
            input=score2, size=sc_size, mode='bilinear', align_corners=False)
        score3 = F.interpolate(
            input=score3, size=sc_size, mode='bilinear', align_corners=False)
        score4 = F.interpolate(
            input=score4, size=sc_size, mode='bilinear', align_corners=False)
        score5 = F.interpolate(
            input=score5, size=sc_size, mode='bilinear', align_corners=False)

        return self.moduleCombine(
            torch.cat(
                [score1, score2, score3, score4, score5], 1)
        )


def accumulate_events(experiment_name, experiment_number, frame_index, dataset_folder):
    left_event_mask = np.zeros((1, dataset_constants.IMAGE_HEIGHT, dataset_constants.IMAGE_WIDTH))

    paths = dataset_constants.experiment_paths(experiment_name, experiment_number, dataset_folder)

    first_index = int(max(-1, (frame_index - np.ceil(0.5 / dataset_constants.TIME_BETWEEN_EXAMPLES))))

    left_event_sequences = []
    for previous_frame_index in range(first_index + 1, frame_index + 1):
        left_events_seq = np.load(paths['cam0']['event_file'] % previous_frame_index)
        # event mask #
        left_event_mask[0, left_events_seq[:, 2].astype(np.int), left_events_seq[:, 1].astype(np.int)] = 1

        left_event_sequences.append(left_events_seq)
    
    left_event_sequence = np.vstack(left_event_sequences)
    return left_event_sequence, left_event_mask


def get_examples_from_experiments(experiments, dataset_folder, frames_filter):
    examples = []
    for experiment_name, experiment_numbers in experiments.items():
        for experiment_number in experiment_numbers:
            examples += get_examples_from_experiment(experiment_name,
                                                     experiment_number,
                                                     dataset_folder,
                                                     frames_filter)
    return examples


def get_examples_from_experiment(experiment_name, experiment_number, dataset_folder, frames_filter):
    examples = []
    paths = dataset_constants.experiment_paths(experiment_name,
                                               experiment_number,
                                               dataset_folder)
    timestamps = np.loadtxt(paths['timestamps_file'])
    frames_number = timestamps.shape[0]

    depth_images_gt_file = h5py.File(paths['hdf_file'], 'r')
    depth_images_gt = np.array(depth_images_gt_file['davis']['left']['depth_image_rect'])

    for frame_index in range(frames_number):
        print(experiment_name, experiment_number, frame_index)
        example = dict()
        example['experiment_name'] = experiment_name
        example['experiment_number'] = experiment_number
        example['frame_index'] = frame_index

        if frame_index not in frames_filter[example['experiment_name']][example['experiment_number']]:
            continue

        example['timestamp'] = timestamps[frame_index]

        example['left_image_path'] = paths['cam0']['image_file'] % frame_index
        example['right_image_path'] = paths['cam1']['image_file'] % frame_index

        example['disparity_image_path'] = paths['disparity_file'] % frame_index

        example['left_event_npy'] = paths['cam0']['event_file'] % frame_index
        example['right_event_npy'] = paths['cam1']['event_file'] % frame_index

        # preprocessing
        depth = morpho.area_closing(depth_images_gt[frame_index], area_threshold=24)
        depth[depth == 0] = np.nan
        example['depth_image_gt'] = np.expand_dims(depth, 0)

        cur_example = {
            'experiment_name': example['experiment_name'],
            'experiment_number': example['experiment_number'],
            'frame_index': example['frame_index'],
            'left': {
                'image':
                get_image(example['left_image_path']),
                'cur_idx_event_sequence_path':
                example['left_event_npy'],
                'disparity_image_path':
                example['disparity_image_path'],
                'depth_image_gt':
                example['depth_image_gt']
            },
            'right': {
                'cur_idx_event_sequence_path':
                example['right_event_npy']
            },
            'timestamp': example['timestamp']
        }

        examples.append(cur_example)

    return examples


def filter_examples(examples, frames_filter):
    return [
        example for example in examples
        if frames_filter[example['experiment_name']][example['experiment_number']] is None or example['frame_index'] in frames_filter[example['experiment_name']][example['experiment_number']]
    ]


def get_image(image_path):
    # Not all examples have images.
    if not os.path.isfile(image_path):
        return np.zeros(
            (dataset_constants.IMAGE_HEIGHT, dataset_constants.IMAGE_WIDTH),
            dtype=np.uint8)
    return np.array(PIL.Image.open(image_path)).astype(np.uint8)


def get_disparity_image(disparity_image_path):
    disparity_image = get_image(disparity_image_path)
    invalid_disparity = (
        disparity_image == dataset_constants.INVALID_DISPARITY)
    # 0 ~ 36, inf
    disparity_image = (disparity_image / dataset_constants.DISPARITY_MULTIPLIER)

    disparity_image[invalid_disparity] = np.nan
    return disparity_image

def get_edge_map_with_depth_hed_new_binary(intensity_image_path, depth_image, net):
    edge_map_image = torchHED.process_file(intensity_image_path, net=net, is_path=True)
    edge_map_depth = torchHED.process_file(depth_image, net=net, is_path=False)
    
    edge_map_depth[edge_map_image == 0] = 0

    depth_image_invalid_mask = np.isnan(depth_image)
    edge_map_depth[depth_image_invalid_mask == True] = 0

    edge_map_depth[edge_map_depth < 25] = 0
    edge_map_depth[edge_map_depth >= 25] = 1

    height, width = depth_image_invalid_mask.shape

    for i in range(height):
        for j in range(width):
            if depth_image_invalid_mask[i][j]:
                edge_map_depth[i][j] = 0
                if i + 1 < height:
                    edge_map_depth[i + 1][j] = 0
                if i - 1 >= 0:
                    edge_map_depth[i - 1][j] = 0
                if j + 1 < width:
                    edge_map_depth[i][j + 1] = 0
                if j - 1 >= 0:
                    edge_map_depth[i][j - 1] = 0

    edge_map_depth = np.expand_dims(edge_map_depth, 0)
    return edge_map_depth


class MVSEC:
    def __init__(self, examples, dataset_folder, transformers):
        self.examples = examples
        self.dataset_folder = dataset_folder
        self.transformers = transformers

    def __len__(self):
        return len(self.examples)

    def shuffle(self, random_seed=0):
        random.seed(random_seed)
        random.shuffle(self.examples)

    def split_into_two(self, first_subset_size):
        return (self.__class__(self.examples[:first_subset_size],
                               self.dataset_folder,
                               self.transformers),
                self.__class__(self.examples[first_subset_size:],
                               self.dataset_folder,
                               self.transformers))

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        
        example = deepcopy(self.examples[index])
        example['left']['cur_idx_event_sequence'] = np.load(example['left']['cur_idx_event_sequence_path'])
        example['right']['cur_idx_event_sequence'] = np.load(example['right']['cur_idx_event_sequence_path'])

        del example['left']['cur_idx_event_sequence_path']
        del example['right']['cur_idx_event_sequence_path']

        for transformer in self.transformers:
            example = transformer(example)
        
        return example

    @classmethod
    def dataset(cls, dataset_folder, experiments, transformers, frames_filter):
        examples = get_examples_from_experiments(experiments, dataset_folder, frames_filter)
        return cls(examples, dataset_folder, transformers)


class IndoorFlying(MVSEC):
    @staticmethod
    def split(dataset_folder, split_number, transformers):
        if split_number == 1:
            dataset = IndoorFlying.dataset(dataset_folder,
                                           {'indoor_flying': [1]},
                                           transformers,
                                           FRAMES_FILTER_FOR_TEST)
            dataset.shuffle()
            validation_set, test_set = dataset.split_into_two(
                first_subset_size=200)
            return (IndoorFlying.dataset(dataset_folder,
                                         {'indoor_flying': [2, 3]},
                                         transformers,
                                         FRAMES_FILTER_FOR_TRAINING),
                    validation_set, test_set)
        elif split_number == 2:
            dataset = IndoorFlying.dataset(dataset_folder,
                                           {'indoor_flying': [2]},
                                           transformers,
                                           FRAMES_FILTER_FOR_TEST)
            dataset.shuffle()
            validation_set, test_set = dataset.split_into_two(
                first_subset_size=200)
            return (IndoorFlying.dataset(dataset_folder,
                                         {'indoor_flying': [1, 3]},
                                         transformers,
                                         FRAMES_FILTER_FOR_TRAINING),
                    validation_set, test_set)
        elif split_number == 3:
            dataset = IndoorFlying.dataset(dataset_folder,
                                           {'indoor_flying': [3]},
                                           transformers,
                                           FRAMES_FILTER_FOR_TEST)
            dataset.shuffle()
            validation_set, test_set = dataset.split_into_two(
                first_subset_size=200)
            return (IndoorFlying.dataset(dataset_folder,
                                         {'indoor_flying': [1, 2]},
                                         transformers,
                                         FRAMES_FILTER_FOR_TRAINING),
                    validation_set, test_set)
        else:
            raise ValueError('Test sequence should be equal to 1, 2 or 3.')


class OutdoorDay(MVSEC):
    @staticmethod
    def get_train_dataset(dataset_folder, time_horizon):
        return OutdoorDay.dataset(dataset_folder, {'outdoor_day': [1, 2]}, time_horizon, FRAMES_FILTER_FOR_TRAINING)
