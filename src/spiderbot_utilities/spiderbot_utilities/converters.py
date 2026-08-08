"""Generic converter functions."""

import numpy as np


def matrix_to_rpy(mat):
    """Convert a matrix to a list of roll, pitch, yaw angles (in radians)."""
    roll = np.atan2(mat[2, 1], mat[2, 2])
    pitch = np.atan2(-mat[2, 0], np.sqrt(mat[2, 1] ** 2 + mat[2, 2] ** 2))
    yaw = np.atan2(mat[1, 0], mat[0, 0])

    return [roll, pitch, yaw]


def rpy_to_matrix(rpy):
    """Convert a list of roll, pitch, yaw angles (in radians) to a matrix."""
    r, p, y = rpy
    cx, cy, cz = np.cos([r, p, y])
    sx, sy, sz = np.sin([r, p, y])

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return Rx @ Ry @ Rz
