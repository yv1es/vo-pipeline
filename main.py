from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice

from dataloader import get_image_iterator, get_intrinsics, Dataset
import parameters as params

BAR = 50 * "-"


@dataclass
class S:
    K: int  # Keypoint count
    M: int  # Candidate keypoint count

    P: np.ndarray  # Keypoints (2, K)
    K: np.ndarray  # Landmarks (3, K)

    C: np.ndarray  # Candidate keypoints (2, M)
    F: np.ndarray  # Candidate first observations (2, M)
    Tau: np.ndarray  # Candidate first camera poses (12, M)


def bootstrapping(images: np.ndarray):
    print(BAR)
    print("Bootstrapping")
    print()

    N, H, W = images.shape
    print(f"Got {N} images with resoultion {H} x {W}")


def main():
    DATASET = Dataset.PARKING

    fig, ax = plt.subplots()

    images_iterator = get_image_iterator(DATASET)
    K = get_intrinsics(DATASET)

    # Take first K_BOOTSTRAP images for bootstrapping
    bootstrap_images = np.array(list(islice(images_iterator, params.K_BOOTSTRAP)))
    S_init = bootstrapping(bootstrap_images)

    plt.ion()  # Turn on interactive mode for live updates

    for image in images_iterator:
        ax.clear()  # Clear previous frame
        ax.imshow(image, cmap="gray")  # Display the current frame
        ax.axis("off")  # Hide axes for better visualization
        plt.pause(0.03)  # Pause to simulate frame rate (adjust as needed)

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the final frame open


if __name__ == "__main__":
    main()
