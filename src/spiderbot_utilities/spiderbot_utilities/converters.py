"""Generic converter functions."""

import numpy as np


def rpy_to_matrix(rpy):
    """Convert a list of roll, pitch, yaw angles (in radians) to a matrix."""
    r, p, y = rpy
    cx, cy, cz = np.cos([r, p, y])
    sx, sy, sz = np.sin([r, p, y])

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return Rx @ Ry @ Rz
