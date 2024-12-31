from collections import defaultdict
from state import S
import cv2
import numpy as np
import parameters as params
import time

from visualizer import Visualizer


def process(
    state: S,
    image_curr: np.ndarray,
    image_prev: np.ndarray,
    K: np.ndarray,
    pose_history: list[np.ndarray],
    visualizer: Visualizer | None = None,
) -> S:
    state.i += 1
    start_time = time.time()
    print(60 * "*")
    print(f"Processing frame {state.i}")
    print(60 * "*")

    print()
    print("......... Localizing camera pose .........")

    # Previous frame tracked keypoints in 2D
    p_prev = state.P.T.reshape(-1, 1, 2).astype(np.float32)  # shape: (N,1,2)
    num_keypoints = state.P.shape[1]
    if num_keypoints < params.MIN_LOCALIZATION_K:
        raise AssertionError(
            f"Not enough keypoints to localize: {num_keypoints}/{params.MIN_LOCALIZATION_K}"
        )

    # Use KLT to find point correspondences
    p_curr, status, _ = cv2.calcOpticalFlowPyrLK(
        image_prev,
        image_curr,
        p_prev,
        None,
        winSize=(params.KLT_WINDOW, params.KLT_WINDOW),
        maxLevel=params.KLT_PYR_LEVELS,
    )
    status = status.reshape(-1)
    # p_prev_klt = p_prev[status == 1].reshape(-1, 2)
    p_curr_klt = p_curr[status == 1].reshape(-1, 2)
    X_klt = state.X[:, status == 1]
    print(f"KLT found {np.sum(status == 1)} keypoint correspondences")

    # Use PnP RANSAC for camera localization
    print(f"Running PnP RANSAC on {X_klt.shape[1]} keypoints-landmark correspondences")
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        X_klt.T,  # shape: (N_in, 3)
        p_curr_klt,
        K,
        distCoeffs=None,
        flags=cv2.SOLVEPNP_P3P,
        reprojectionError=2.0,
        confidence=0.99,
        iterationsCount=params.PNP_RANSAC_ITERATIONS,
    )

    if not success or inliers is None or len(inliers) <= params.MIN_INLIERS:
        raise AssertionError(
            f"PnP RANSAC failed! Maybe not enough inliers ({len(inliers)})"
        )

    inliers = inliers.flatten()
    R, _ = cv2.Rodrigues(rvec)
    pose_curr = np.zeros((3, 4))
    pose_curr[:3, :3] = R
    pose_curr[:3, 3] = tvec.flatten()
    pose_history.append(pose_curr)

    # Filter out outliers in state
    p1_inliers = p_curr_klt[inliers]
    X_inliers = X_klt[:, inliers]
    state.P = p1_inliers.T
    state.X = X_inliers

    print()
    print(f"Found pose with {len(inliers)} inliers\n{pose_curr}")

    print()
    print("......... Triangulating new landmarks .........")

    if state.C.shape[1] > 0:
        # Find candidate correspondences using KLT
        c_prev = state.C.T.reshape(-1, 1, 2).astype(np.float32)
        c_curr, status_klt, _ = cv2.calcOpticalFlowPyrLK(
            image_prev,
            image_curr,
            c_prev,
            None,
            winSize=(params.KLT_WINDOW, params.KLT_WINDOW),
            maxLevel=params.KLT_PYR_LEVELS,
        )
        status_klt = status_klt.reshape(-1)
        c_prev_klt = c_prev[status_klt == 1].reshape(-1, 2)
        c_curr_klt = c_curr[status_klt == 1].reshape(-1, 2)
        F_klt = state.F[:, status_klt == 1]
        Tau_klt = state.Tau[:, status_klt == 1]
        print(
            f"KLT found {np.sum(status_klt == 1)} out of {state.C.shape[1]} candidate correspondences"
        )

        # Compute angles between triangulation rays
        R_end = pose_curr[:, :3]
        t_end = pose_curr[:, 3]
        K_inv = np.linalg.inv(K)
        angles = []
        for i_cand in range(c_curr_klt.shape[0]):
            # The pose from which the candidate was created
            M_start = Tau_klt[:, i_cand].reshape(3, 4)
            R_start = M_start[:, :3]
            t_start = M_start[:, 3]

            x_start, y_start = F_klt[0, i_cand], F_klt[1, i_cand]

            d_start_cam = K_inv @ np.array([x_start, y_start, 1.0], dtype=np.float32)
            d_start_world = R_start.T @ d_start_cam

            x_end, y_end = c_curr_klt[i_cand, 0], c_curr_klt[i_cand, 1]

            d_end_cam = K_inv @ np.array([x_end, y_end, 1.0], dtype=np.float32)
            d_end_world = R_end.T @ d_end_cam

            d_start_world_norm = d_start_world / np.linalg.norm(d_start_world)
            d_end_world_norm = d_end_world / np.linalg.norm(d_end_world)

            dot_product = np.dot(d_start_world_norm, d_end_world_norm)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle = np.arccos(dot_product)

            angles.append(angle)
        angles = np.array(angles)

        # Triangulate points with angle above threshold
        to_triangulate = np.where(angles > params.TRIANGULATION_ANGLE_THRESHOLD)[0]
        print(
            f"{len(to_triangulate)} candidates have large enough angle for triangulation "
            f"(threshold={params.TRIANGULATION_ANGLE_THRESHOLD} rad)"
        )
        if len(to_triangulate) > 0:
            M_end = K @ pose_curr

            # Group together candidates which share the same M_start
            pose_dict = defaultdict(list)
            for idx in to_triangulate:
                M_start = K @ Tau_klt[:, idx].reshape((3, 4))
                M_start_key = tuple(M_start.flatten())
                pose_dict[M_start_key].append(idx)

            # Triangulate candidates which have the same M_start together and append to state
            for M_start_key, group_idx in pose_dict.items():
                M_start = np.array(M_start_key).reshape(3, 4)
                c_start_group = F_klt[:, group_idx]  # shape (2, N)
                c_curr_group = c_curr_klt[group_idx].T  # shape (2, N)

                X_new_h = cv2.triangulatePoints(
                    M_start, M_end, c_start_group, c_curr_group
                )
                X_new_h /= X_new_h[3, :]
                X_new = X_new_h[:3, :]  # shape (3, N)

                # Same "z>0" check to ensure points are in front of both cameras
                R_start = M_start[:, :3]
                t_start = M_start[:, 3]
                C1 = R_start @ X_new + t_start.reshape(3, 1)
                C2 = R_end @ X_new + t_end.reshape(3, 1)

                mask_in_front = (C1[2, :] > 0) & (C2[2, :] > 0)

                X_new = X_new[:, mask_in_front]
                c_curr_group_filtered = c_curr_group[:, mask_in_front]

                # Append to state
                state.P = np.hstack((state.P, c_curr_group_filtered))
                state.X = np.hstack((state.X, X_new))

            # Remove triangulated candidates from the 'C/F/Tau' sets
            mask_keep = np.ones(len(c_prev_klt), dtype=bool)
            mask_keep[to_triangulate] = False
            c_prev_klt = c_prev_klt[mask_keep]
            c_curr_klt = c_curr_klt[mask_keep]
            F_klt = F_klt[:, mask_keep]
            Tau_klt = Tau_klt[:, mask_keep]

        # Keep successfully tracked but not-yet-triangulated candidates
        state.C = c_curr_klt.T
        state.F = F_klt
        state.Tau = Tau_klt

    print()
    print("......... Adding fresh candidates .........")

    if state.C.shape[1] < params.MAX_CANDIDATES:
        # Detect new features (Shi-Tomasi)
        corners = cv2.goodFeaturesToTrack(
            image_curr,
            maxCorners=3000,
            qualityLevel=0.01,
            minDistance=10.0,
        )
        if corners is not None:
            corners = corners.reshape(-1, 2)
            print(f"Detected {len(corners)} potential candidates")
        else:
            print("No corners detected")

        # Filter out corners that are too close to existing points
        if corners is not None and len(corners) > 0:
            if state.C.size > 0:
                existing_points = np.hstack([state.P, state.C]).T
                dist_matrix = np.linalg.norm(
                    corners[:, None, :] - existing_points[None, :, :], axis=2
                )
                min_dist = 3.0
                valid_mask = np.all(dist_matrix > min_dist, axis=1)
                distinct_corners = corners[valid_mask]
            else:
                distinct_corners = corners

            if distinct_corners.size > 0:
                print(f"Detected {len(distinct_corners)} viable fresh candidates")
                distinct_corners = distinct_corners[
                    : params.MAX_CANDIDATES - state.C.shape[1]
                ]
                print(f"Adding {len(distinct_corners)} fresh candidates")

                new_candidate_poses = np.tile(
                    pose_curr.flatten()[:, None], (1, distinct_corners.shape[0])
                )
                state.C = np.hstack((state.C, distinct_corners.T))
                state.F = np.hstack((state.F, distinct_corners.T))
                state.Tau = np.hstack((state.Tau, new_candidate_poses))
    else:
        print("Already tracking enough candidates")

    print()
    print("......... State .........")
    print(f"Currently tracking {state.P.shape[1]} keypoints")
    print(f"Currently tracking {state.C.shape[1]} candidates")

    # Optional: visualize results
    if visualizer is not None:
        visualizer.update(
            state,
            image_curr,
            pose_history,
        )

    # FPS calculation
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = 1 / elapsed_time
    print()
    print(f"FPS: {fps:.1f}")
    print()
    print()

    return state
