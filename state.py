from dataclasses import dataclass
import numpy as np


@dataclass
class S:
    i: int  # Frame number

    P: np.ndarray  # Keypoints (2, K)
    X: np.ndarray  # Landmarks (3, K)

    C: np.ndarray  # Candidate keypoints (2, M)
    F: np.ndarray  # Candidate first observations (2, M)
    Tau: np.ndarray  # Candidate first camera poses (16, M)
