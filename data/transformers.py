import torch
import numpy as np
import copy
import random
from data import dataset_constants
from data.dataset_constants import *


def absolute_time_to_relative(example):
    end_timestamp = example['cur_idx_event_sequence'][-1, 0]
    example['cur_idx_event_sequence'][:, 0] -= end_timestamp

    return example


def normalize_features_to_zero_mean_unit_std(example, feature_index):
    event_sequence = example['event_sequence']
    mean = event_sequence[:, feature_index].mean()
    std = event_sequence[:, feature_index].std()
    event_sequence[:, feature_index] = (event_sequence[:, feature_index] - mean) / (std + 1e-10)
    example['event_sequence'] = event_sequence
    return example


def normalize_polarity(example):
    return normalize_features_to_zero_mean_unit_std(example, 3)


def normalize_timestamp(example):
    # 5, 2, h, w
    event_sequence = example['cur_idx_event_sequence']
    mean = event_sequence[:, 0, :, :].mean()
    std = event_sequence[:, 0, :, :].std()
    event_sequence[:, 0, :, :] = (event_sequence[:, 0, :, :] - mean) / (std + 1e-10)
    example['cur_idx_event_sequence'] = event_sequence
    return example


class KeepRecentEventsGoundtruth(object):
    def __init__(self, number_of_events=15000):
        self._number_of_events = number_of_events

    def __call__(self, example):
        event_sequence = example['left']['event_sequence']
        first_index = int(max(0, len(event_sequence) - self._number_of_events))
        trimed_event_sequence = copy.deepcopy(event_sequence)
        trimed_event_sequence = trimed_event_sequence[first_index:]
        location_has_event = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.int)
        location_has_event[trimed_event_sequence[:, 2].astype(np.int), trimed_event_sequence[:, 1].astype(np.int)] = 1
        mask = (location_has_event == 0)
        example['left']['disparity_image_sparse'][mask] = float('inf')
        del example['left']['event_sequence']
        return example


class ApplyTransformersToLeftRight(object):
    def __init__(self, transformers):
        self.transformers = transformers

    def __call__(self, example):
        for camera_name in ['left', 'right']:
            for transformer in self.transformers:
                example[camera_name] = transformer(example[camera_name])
        return example


class EventsSplit(object):
    def __init__(self, slow_split, fast_split):
        self.slow_split = slow_split
        self.fast_split = fast_split

    def accumulate_events(self, example):
        cur_time_events = np.zeros((2, IMAGE_HEIGHT, IMAGE_WIDTH))

        for i in range(len(example)):
            if example[i, 3] == 1:
                cur_time_events[0, int(example[i, 2]), int(example[i, 1])] += 1
            else:
                cur_time_events[1, int(example[i, 2]), int(example[i, 1])] += 1

        return cur_time_events

    def __call__(self, example):
        event_sequence = example['cur_idx_event_sequence']
        seq_len = len(event_sequence)

        trans_example_slow = np.zeros((self.slow_split, 2, IMAGE_HEIGHT, IMAGE_WIDTH))

        trans_example_slow[0] = self.accumulate_events(event_sequence[0: seq_len])

        del example['cur_idx_event_sequence']
        example['cur_idx_event_sequence_slow'] = trans_example_slow

        return example


def dictionary_of_numpy_arrays_to_tensors(example):
    if isinstance(example, dict):
        return {
            key: dictionary_of_numpy_arrays_to_tensors(value)
            for key, value in example.items()
        }
    if isinstance(example, np.ndarray):
        return torch.from_numpy(example).float()
    return example


def init_transformers(slow_split=1, fast_split=3):
    single_view_transformers = [EventsSplit(slow_split, fast_split)]

    dataset_transformers = []
    dataset_transformers += [
        ApplyTransformersToLeftRight(single_view_transformers),
        dictionary_of_numpy_arrays_to_tensors
    ]
    return dataset_transformers
