import numpy as np

# Bootstrapping
K_BOOTSTRAP = 10

# Localication
MIN_LOCALIZATION_K = 5
PNP_RANSAC_ITERATIONS = 5000

MIN_INLIERS = 0

MAX_TRACKED_KEYPOINTS = 5000
MAX_CANDIDATES = 5000

TRIANGULATION_ANGLE_THRESHOLD = np.deg2rad(2.2)

KLT_WINDOW = 9
KLT_PYR_LEVELS = 6
