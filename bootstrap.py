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


def bootstrapping_sift(images: np.ndarray, K, plot=False) -> S:
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
        nfeatures=1000,
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

    if plot:
        print()
        print(f"P ({P.shape}):\n{P}")
        print()
        print(f"X ({X.shape}):\n{X}")
        print()
        print(f"pose ({pose.shape}):\n{pose}")

        fig = plt.figure(figsize=(18, 6))

        ax_img1 = fig.add_subplot(1, 3, 1)
        ax_img1.imshow(img1, cmap="gray")
        ax_img1.scatter(pts1_in[:, 0], pts1_in[:, 1], c="r", s=10)
        ax_img1.set_title("Frame 0 Inliers")
        ax_img1.axis("off")

        ax_img2 = fig.add_subplot(1, 3, 2)
        ax_img2.imshow(img2, cmap="gray")
        ax_img2.scatter(pts2_in[:, 0], pts2_in[:, 1], c="r", s=10)
        ax_img2.set_title(f"Frame {N-1} Inliers")
        ax_img2.axis("off")

        ax_3d = fig.add_subplot(1, 3, 3, projection="3d")
        ax_3d.view_init(elev=0, azim=-90)
        ax_3d.scatter(X[0, :], X[1, :], X[2, :], marker="o")
        print(X)

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
        1,
        P,
        X,
        np.array([]).reshape((2, -1)),
        np.array([]).reshape((2, -1)),
        np.array([]).reshape((12, -1)),
    )


def bootstrapping_klt(images: np.ndarray, K, plot=False):
    print("-" * 50)
    print("Bootstrapping with Shi-Thomasi, KLT and 8-point RANSAC")
    print()

    num_frames, h, w = images.shape
    print(f"Got {num_frames} images with resolution {h} x {w}")

    init_pts = cv2.goodFeaturesToTrack(images[0], 3000, 0.001, 5.0).reshape(-1, 2)
    pts_prev = init_pts.reshape(-1, 1, 2)
    img_prev = images[0]
    for i in range(1, num_frames):
        img_next = images[i]
        flow, sts, _ = cv2.calcOpticalFlowPyrLK(
            img_prev, img_next, init_pts.reshape(-1, 1, 2), None
        )
        ok = sts.reshape(-1) == 1
        pts_prev = flow[ok].reshape(-1, 1, 2)
        img_prev = img_next
    ptsN = pts_prev.squeeze(1)
    print(f"After KLT tracking, {len(ptsN)} corners remain to frame {num_frames-1}.")

    ptsN_single, sts, _ = cv2.calcOpticalFlowPyrLK(
        images[0], images[-1], init_pts.reshape(-1, 1, 2), None
    )
    ok = sts.reshape(-1) == 1
    p0_in = init_pts[ok]
    pN_in = ptsN_single[ok].squeeze(1)
    print(
        f"Single-pass KLT check: {len(pN_in)} / {len(init_pts)} survive to last frame."
    )

    F, maskF = cv2.findFundamentalMat(
        p0_in, pN_in, cv2.FM_8POINT + cv2.FM_RANSAC, 1.0, 0.99
    )
    maskF = maskF.ravel()
    inliers = maskF.sum()
    print()
    print(f"Found F with {inliers}/{len(maskF)} inliers")
    print("F:\n", F)

    pts1 = p0_in[maskF == 1]
    pts2 = pN_in[maskF == 1]
    pts1_h = np.r_[pts1.T, np.ones((1, inliers))]
    pts2_h = np.r_[pts2.T, np.ones((1, inliers))]

    E = K.T @ F @ K
    print()
    print("E:\n", E)

    Rots, u3 = decomposeEssentialMatrix(E)
    R, t = disambiguateRelativePose(Rots, u3, pts1_h, pts2_h, K, K)
    M1 = K @ np.eye(3, 4)
    M2 = K @ np.c_[R, t]
    X = linearTriangulation(pts1_h, pts2_h, M1, M2)[:-1]
    ok1 = X[2] > 0
    X_cam2 = R @ X + t[:, None]
    ok2 = X_cam2[2] > 0
    valid = ok1 & ok2
    X = X[:, valid]
    P = pts2_h[:-1, valid]

    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t

    if plot:
        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(images[0], cmap="gray")
        ax1.scatter(pts1[:, 0], pts1[:, 1], c="r", s=10)
        ax1.set_title("Frame 0 Inliers")
        ax1.axis("off")
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(images[-1], cmap="gray")
        ax2.scatter(pts2[:, 0], pts2[:, 1], c="r", s=10)
        ax2.set_title(f"Frame {num_frames-1} Inliers")
        ax2.axis("off")
        for a, b in zip(pts1, pts2):
            a = tuple(a.astype(int))
            b = tuple(b.astype(int))
            ax2.plot([a[0], b[0]], [a[1], b[1]], color="green", linewidth=1)
            ax2.scatter(b[0], b[1], color="red", s=10)
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        ax3.scatter(X[0], X[1], X[2], marker="o")
        drawCamera(ax3, np.zeros(3), np.eye(3))
        ax3.text(-0.1, -0.1, -0.1, "Cam 1")
        c2 = -R.T @ t
        drawCamera(ax3, c2, R.T)
        ax3.text(c2[0] - 0.1, c2[1] - 0.1, c2[2] - 0.1, "Cam 2")
        ax3.set_title("3D Triangulated Points & Cameras")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_zlabel("Z")
        ax3.legend()
        plt.tight_layout()
        plt.show()

    return pose, S(
        1,
        P,
        X,
        np.array([]).reshape((2, 0)),
        np.array([]).reshape((2, 0)),
        np.array([]).reshape((12, 0)),
    )


if __name__ == "__main__":
    dataset = Dataset.PARKING

    images = get_image_iterator(dataset)
    K = get_intrinsics(dataset)

    # Take first K_BOOTSTRAP images for bootstrapping
    bootstrap_images = np.array(list(islice(images, params.K_BOOTSTRAP)))
    pose, s = bootstrapping_klt(bootstrap_images, K, plot=True)
