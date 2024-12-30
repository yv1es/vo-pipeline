import cv2
import os
import matplotlib.pyplot as plt

# Dataset paths
PARKING = "data/parking/images/"
KITTI = "data/kitti05/kitti/05/image_0/"


def image_iterator(path):
    """
    An iterator that yields images as NumPy arrays from a folder in alphabetical order.

    Args:
        path (str): The folder containing the images.

    Yields:
        np.ndarray: The image as a NumPy array.
    """
    files = sorted(os.listdir(path))

    for file in files:
        filepath = os.path.join(path, file)
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            yield image
        else:
            print(f"Warning: File '{file}' could not be read as an image.")


if __name__ == "__main__":
    # Usage example
    fig, ax = plt.subplots()
    plt.ion()  # Turn on interactive mode for live updates

    for image in image_iterator(PARKING):
        ax.clear()  # Clear previous frame
        ax.imshow(image, cmap="gray")  # Display the current frame
        ax.axis("off")  # Hide axes for better visualization
        plt.pause(0.03)  # Pause to simulate frame rate (adjust as needed)

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the final frame open
