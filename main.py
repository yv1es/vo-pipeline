from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
import parameters as params

from bootstrap import bootstrapping_klt, bootstrapping_sift
from continuous_vo import process
from dataloader import get_image_iterator, get_intrinsics, Dataset
from visualizer import Visualizer


DATASET = Dataset.MALAGA
DATASET = Dataset.PARKING
DATASET = Dataset.KITTI


def main():
    images = get_image_iterator(DATASET)
    K = get_intrinsics(DATASET)

    # Create Visualizer
    visualizer = Visualizer()

    # Skip frames in the beginning for debugging
    # skipped_images = np.array(list(islice(images, 700)))

    if DATASET == Dataset.KITTI:
        params.TRIANGULATION_ANGLE_THRESHOLD = np.deg2rad(1.5)

    # Bootstrap
    bootstrap_images = np.array(list(islice(images, params.K_BOOTSTRAP)))
    pose, s = bootstrapping_klt(bootstrap_images, K, plot=False)
    pose_history = [pose]
    image_prev = next(images)

    # Continuous operation
    for image_curr in images:
        s = process(s, image_curr, image_prev, K, pose_history, visualizer=visualizer)
        image_prev = image_curr

    # Final show
    plt.show(block=True)


if __name__ == "__main__":
    main()
