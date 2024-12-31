from enum import Enum, auto
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


class Dataset(Enum):
    PARKING = auto()
    KITTI = auto()
    MALAGA = auto()


image_paths = {
    Dataset.PARKING: "data/parking/images/",
    Dataset.KITTI: "data/kitti05/kitti/05/image_0/",
    Dataset.MALAGA: "data/malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_800x600_Images/",
}


intrinsics = {
    Dataset.PARKING: np.array(
        [
            [331.37, 0, 320],
            [0, 369.568, 240],
            [0, 0, 1],
        ]
    ),
    Dataset.KITTI: np.array(
        [
            [707.0912, 0.0, 601.8873],
            [0.0, 707.0912, 183.1104],
            [0.0, 0.0, 1.0],
        ]
    ),
    Dataset.MALAGA: np.array(
        [
            [621.18428, 0.0, 404.00760],
            [0.0, 621.18428, 309.05989],
            [0.0, 0.0, 1.0],
        ]
    ),
}


def get_intrinsics(dataset: Dataset) -> np.ndarray:
    if dataset in intrinsics:
        return intrinsics[dataset]
    else:
        raise ValueError(f"Intrinsic matrix not defined for dataset {dataset}.")


def get_image_iterator(dataset: Dataset):
    if dataset not in image_paths:
        raise ValueError(f"Path not defined for dataset {dataset}.")

    path = image_paths[dataset]
    files = sorted(os.listdir(path))

    if dataset == Dataset.MALAGA:
        files = [file for file in files if "_left" in file]

    for file in files:
        filepath = os.path.join(path, file)
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            yield image.astype(np.uint8)
        else:
            print(f"Warning: File '{file}' could not be read as an image.")
