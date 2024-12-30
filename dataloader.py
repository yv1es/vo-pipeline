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
}

intrinsics = {
    Dataset.PARKING: np.array([[331.37, 0, 320], [0, 369.568, 240], [0, 0, 1]]),
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

    for file in files:
        filepath = os.path.join(path, file)
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            yield image
        else:
            print(f"Warning: File '{file}' could not be read as an image.")
