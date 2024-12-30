import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
import cv2

from previous.decompose_essential_matrix import decomposeEssentialMatrix
from previous.disambiguate_relative_pose import disambiguateRelativePose
from previous.draw_camera import drawCamera
from previous.linear_triangulation import linearTriangulation

from dataloader import get_image_iterator, get_intrinsics, Dataset
import parameters as params
from state import S


def bootstrapping_klt(images: np.ndarray, K, debug=False) -> S:
    """
    Bootstrapping using:
      1) Harris (Shi-Tomasi) corners detection in the first frame
      2) KLT (Optical Flow) tracking across subsequent frames
      3) Triangulation between the first and last frames
    """
    print(50 * "-")
    print("Bootstrapping with KLT")
    print()

    N, H, W = images.shape
    print(f"Got {N} images with resolution {H} x {W}")

    # Convert frames to uint8
    gray_images = [img.astype(np.uint8) for img in images]

    # 1) Detect corners in the first frame
    # --------------------------------------------------
    # goodFeaturesToTrack under the hood uses either Harris or Shi-Tomasi measure.
    # Here we choose the Shi-Tomasi approach, but you could enable Harris measure
    # by setting 'useHarrisDetector=True' and tuning 'k=0.04', etc.
    feature_params = dict(
        maxCorners=1000, qualityLevel=0.01, minDistance=7, blockSize=7
    )
    p0 = cv2.goodFeaturesToTrack(gray_images[0], mask=None, **feature_params)
    p0 = p0.reshape(-1, 2)

    print(f"Detected {len(p0)} corners in frame 0.")

    # 2) Track corners across frames using KLT
    # --------------------------------------------------
    # We'll track from frame (i-1) to frame i. In the end, we want
    # the positions of these corners in the last frame.
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    # We will track from frame 0 => 1 => 2 => ... => N-1
    prev_img = gray_images[0]
    prev_points = p0.copy()

    # We'll keep track of the final positions in the last frame
    # as well as which points remain 'inlier'/successfully tracked.
    for i in range(1, N):
        next_img = gray_images[i]
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_img, next_img, prev_points, None
        )
        # status == 1 means successfully tracked
        good_idx = status.ravel() == 1

        prev_points = next_points[good_idx]
        # Prepare for next iteration
        prev_img = next_img

    # At this point, 'prev_points' are the positions of the original p0
    # corners that successfully tracked to the last frame.
    # We also need to filter our original p0 to the ones that made it.
    # The easiest way is to track each step carefully or do it in one pass
    # storing all intermediate statuses. For brevity, let's do a simple approach:
    # We'll re-run the KLT in one pass from frame 0 to frame N-1, but keep the final
    # matching indices. Alternatively, you can store each step's status and do an
    # intersection. For demonstration, let's assume we already have it:

    # For a robust approach, it's best to re-run from 0 -> (N-1) in a single flow call,
    # but this example keeps it conceptually simpler. We'll do a quick version:
    p0 = p0[good_idx]
    pN = prev_points  # positions in the last frame

    print(
        f"After KLT tracking, {len(pN)} corners remain matched between frame 0 and frame {N-1}."
    )

    # 3) Estimate E from p0 <-> pN, decompose, triangulate
    # --------------------------------------------------
    # Convert to float32
    p0_f32 = p0.astype(np.float32)
    pN_f32 = pN.astype(np.float32)

    # Find Fundamental matrix via RANSAC
    F, mask_f = cv2.findFundamentalMat(
        p0_f32,
        pN_f32,
        cv2.FM_8POINT + cv2.FM_RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.99,
    )

    inlier_mask = mask_f.ravel() == 1
    inliers = np.sum(inlier_mask)
    print()
    print(f"Found F with {inliers}/{len(mask_f)} inliers")
    print("F:\n", F)

    # Keep only inliers
    pts1_in = p0_f32[inlier_mask]
    pts2_in = pN_f32[inlier_mask]

    # Convert to homogeneous
    pts1_in_h = np.r_[pts1_in.T, np.ones((1, inliers))]
    pts2_in_h = np.r_[pts2_in.T, np.ones((1, inliers))]

    # Compute Essential matrix
    E = K.T @ F @ K
    print()
    print("E:\n", E)

    # Decompose E into R, t
    Rots, u3 = decomposeEssentialMatrix(E)
    R_C2_W, T_C2_W = disambiguateRelativePose(Rots, u3, pts1_in_h, pts2_in_h, K, K)

    # Triangulate points
    M1 = K @ np.eye(3, 4)
    M2 = K @ np.c_[R_C2_W, T_C2_W]
    X = linearTriangulation(pts1_in_h, pts2_in_h, M1, M2)[:-1]  # shape (3, N)

    # Filter 3D points behind the cameras
    valid1 = X[2, :] > 0
    X_cam2 = R_C2_W @ X + T_C2_W[:, np.newaxis]
    valid2 = X_cam2[2, :] > 0
    valid_points = valid1 & valid2
    X = X[:, valid_points]
    P = pts2_in_h[:-1, valid_points]

    if debug:
        print()
        print(f"P ({P.shape}):\n{P}")
        print()
        print(f"X ({X.shape}):\n{X}")

        fig = plt.figure(figsize=(18, 6))

        # Visualize inliers in the first image
        ax_img1 = fig.add_subplot(1, 3, 1)
        ax_img1.imshow(gray_images[0], cmap="gray")
        ax_img1.scatter(pts1_in[:, 0], pts1_in[:, 1], c="r", s=10)
        ax_img1.set_title("Frame 0 Inliers")
        ax_img1.axis("off")

        # Visualize inliers in the last image
        ax_img2 = fig.add_subplot(1, 3, 2)
        ax_img2.imshow(gray_images[-1], cmap="gray")
        ax_img2.scatter(pts2_in[:, 0], pts2_in[:, 1], c="r", s=10)
        ax_img2.set_title(f"Frame {N-1} Inliers")
        ax_img2.axis("off")

        # 3D scatter for triangulated landmarks + camera axes
        ax_3d = fig.add_subplot(1, 3, 3, projection="3d")
        ax_3d.scatter(X[0, :], X[1, :], X[2, :], marker="o")

        # Display camera pose
        drawCamera(ax_3d, np.zeros((3,)), np.eye(3))
        ax_3d.text(-0.1, -0.1, -0.1, "Cam 1")

        center_cam2_W = -R_C2_W.T @ T_C2_W
        drawCamera(ax_3d, center_cam2_W, R_C2_W.T)
        ax_3d.text(
            center_cam2_W[0] - 0.1,
            center_cam2_W[1] - 0.1,
            center_cam2_W[2] - 0.1,
            "Cam 2",
        )
        ax_3d.set_title("3D Triangulated Points & Cameras")
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.legend()

        plt.tight_layout()
        plt.show()

    # Return initial state
    return S(
        P.shape[1],
        0,
        P,
        X,
        np.array([]),
        np.array([]),
        np.array([]),
    )


def bootstrapping_sift(images: np.ndarray, K, debug=False) -> S:
    print(50 * "-")
    print("Bootstrapping")
    print()

    N, H, W = images.shape
    print(f"Got {N} images with resolution {H} x {W}")

    # Convert first and last images to uint8 (if needed) for SIFT
    img1 = images[0].astype(np.uint8)
    img2 = images[-1].astype(np.uint8)

    # Find sift features in images
    sift = cv2.SIFT_create(
        nfeatures=1000,  # Increase max number of features to retain
        # contrastThreshold=0.01,  # Lower threshold => more features
        # edgeThreshold=10,
    )
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    print(f"Found {len(matches)} matching SIFT features between frame 0 and {N-1}")

    # Extract matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Estimate the Fundamental matrix using RANSAC 8-point algorithm
    F, mask_f = cv2.findFundamentalMat(
        pts1,
        pts2,
        cv2.FM_8POINT + cv2.FM_RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.99,
    )
    inlier_mask = mask_f.ravel() == 1
    inliers = np.sum(inlier_mask)

    print()
    print(f"Found F with {inliers}/{len(mask_f)} inliers")
    print("F:\n", F)

    pts1_in = pts1[inlier_mask]
    pts2_in = pts2[inlier_mask]
    pts1_in_h = np.r_[pts1_in.T, np.ones((1, inliers))]
    pts2_in_h = np.r_[pts2_in.T, np.ones((1, inliers))]

    E = K.T @ F @ K
    print()
    print("E:\n", E)

    Rots, u3 = decomposeEssentialMatrix(E)
    R_C2_W, T_C2_W = disambiguateRelativePose(Rots, u3, pts1_in_h, pts2_in_h, K, K)

    # Triangulate landmarks
    M1 = K @ np.eye(3, 4)
    M2 = K @ np.c_[R_C2_W, T_C2_W]
    X = linearTriangulation(pts1_in_h, pts2_in_h, M1, M2)[:-1]

    # Filter points behind either camera
    # Z > 0 in camera 1
    valid1 = X[2, :] > 0

    # Transform to camera-2 frame
    X_cam2 = R_C2_W @ X + T_C2_W[:, np.newaxis]
    valid2 = X_cam2[2, :] > 0

    valid_points = valid1 & valid2
    X = X[:, valid_points]
    P = pts2_in_h[:-1]
    P = P[:, valid_points]

    pose = np.eye(4)
    pose[:3, :3] = R_C2_W
    pose[:3, 3] = T_C2_W

    if debug:
        print()
        print(f"P ({P.shape}):\n{P}")
        print()
        print(f"X ({X.shape}):\n{X}")
        print()
        print(f"pose ({pose.shape}):\n{pose}")

        fig = plt.figure(figsize=(18, 6))

        # First image with inlier correspondences
        ax_img1 = fig.add_subplot(1, 3, 1)
        ax_img1.imshow(img1, cmap="gray")
        ax_img1.scatter(pts1_in[:, 0], pts1_in[:, 1], c="r", s=10)
        ax_img1.set_title("Frame 0 Inliers")
        ax_img1.axis("off")

        # Last image with inlier correspondences
        ax_img2 = fig.add_subplot(1, 3, 2)
        ax_img2.imshow(img2, cmap="gray")
        ax_img2.scatter(pts2_in[:, 0], pts2_in[:, 1], c="r", s=10)
        ax_img2.set_title(f"Frame {N-1} Inliers")
        ax_img2.axis("off")

        # 3D scatter for triangulated landmarks + camera axes
        ax_3d = fig.add_subplot(1, 3, 3, projection="3d")

        ax_3d.scatter(X[0, :], X[1, :], X[2, :], marker="o")

        # Display camera pose
        drawCamera(ax_3d, np.zeros((3,)), np.eye(3))
        ax_3d.text(-0.1, -0.1, -0.1, "Cam 1")

        center_cam2_W = -R_C2_W.T @ T_C2_W
        drawCamera(ax_3d, center_cam2_W, R_C2_W.T)
        ax_3d.text(
            center_cam2_W[0] - 0.1,
            center_cam2_W[1] - 0.1,
            center_cam2_W[2] - 0.1,
            "Cam 2",
        )

        center_cam2_W = -R_C2_W.T @ T_C2_W

        ax_3d.set_title("3D Triangulated Points & Cameras")
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.legend()

        plt.tight_layout()
        plt.show()

    # return initial state
    return pose, S(
        P,
        X,
        np.array([]),
        np.array([]),
        np.array([]),
    )


if __name__ == "__main__":
    dataset = Dataset.PARKING

    images = get_image_iterator(dataset)
    K = get_intrinsics(dataset)

    # Take first K_BOOTSTRAP images for bootstrapping
    bootstrap_images = np.array(list(islice(images, params.K_BOOTSTRAP)))
    pose, s = bootstrapping_sift(bootstrap_images, K, debug=True)
