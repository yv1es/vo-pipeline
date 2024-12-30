from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice


from bootstrap import bootstrapping_sift
from dataloader import get_image_iterator, get_intrinsics, Dataset
import parameters as params
from state import S

BAR = 50 * "-"


# @dataclass
# class S:
#     P: np.ndarray  # Keypoints (2, K)
#     X: np.ndarray  # Landmarks (3, K)
#
#     C: np.ndarray  # Candidate keypoints (2, M)
#     F: np.ndarray  # Candidate first observations (2, M)
#     Tau: np.ndarray  # Candidate first camera poses (12, M)
#


def process(
    state: S,
    pose_prev: np.ndarray,
    image: np.ndarray,
    image_prev: np.ndarray,
    K: np.ndarray,
    debug=False,
) -> tuple[np.ndarray | None, S]:
    pass

    # if there are enough keypoins and landmarks from previouse (in s) (meaning K > params.MIN_LOCALIZATION_K) then
    #   use klt to find the corresponding keypoints in the current frame
    #   use the landmarks and the keypoint correspondances and p3p ransac to find the current camera pose since
    #   this camera pose is returned as a 4x4 matrix (R|t) homogenous
    #   keypoint which are not ransac inliers are removed from s keypoints or landmarks
    #
    # else if there are not enough keypoints just return none as pose
    #
    # determening new keypoins:
    #    in s.C are keypoint candidates from image_prev find the in the current frame using KLT tracker
    #    in s.F are the candidates pixel coordinates when it was first detected, compute the bearing difference between
    #    the candidate location in F and on the current frame, when it is greater than a threshold, triangulate the point
    #    using the F, current frame and the camera pose corresponding to F saved in Tau
    #    add the resulting keypoint to s.P and s.X
    #
    #
    # finding new candidate keypoins:
    #
    # on the image find good features to track with shi tomasi
    # put them into s.C
    # put them into F
    # finally put the current pose of the camera into Tau


def main():
    DATASET = Dataset.PARKING

    fig, ax = plt.subplots()

    images = get_image_iterator(DATASET)
    K = get_intrinsics(DATASET)

    # Take first K_BOOTSTRAP images for bootstrapping
    bootstrap_images = np.array(list(islice(images, params.K_BOOTSTRAP)))
    pose, s = bootstrapping_sift(bootstrap_images, K)

    pose_history = [np.eye(4), pose]

    img_prev = next(images)
    for img in images:
        pose, s = process(s, pose, img, img_prev, K, debug=True)

        img_prev = img
        pose_history.append(pose)


if __name__ == "__main__":
    main()
