import torch
import numpy as np
from tqdm.auto import tqdm
from typing import Tuple
from scipy.spatial.transform import Rotation


def inverse_homogeneous_transform(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a 4x4 homogeneous transformation matrix.

    Parameters
    ----------
    matrix : np.ndarray
        A (4, 4) homogeneous transformation matrix.

    Returns
    -------
    np.ndarray
        The inverse homogeneous transformation matrix (4, 4).

    Raises
    ------
    ValueError
        If the input is not a 4x4 matrix.
    """
    if matrix.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 numpy array")

    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]

    inverse_rotation = rotation.T
    inverse_translation = -inverse_rotation @ translation

    inverse_matrix = np.eye(4)
    inverse_matrix[:3, :3] = inverse_rotation
    inverse_matrix[:3, 3] = inverse_translation

    return inverse_matrix


def homogeneous_transform(
    translation: Tuple[float, float, float],
    rotation: Tuple[float, float, float, float]
) -> np.ndarray:
    """
    Generate a homogeneous transformation matrix from a translation vector
    and a quaternion rotation.

    Parameters
    ----------
    translation : tuple of float
        Translation vector (x, y, z).
    rotation : tuple of float
        Quaternion rotation (x, y, z, w).

    Returns
    -------
    np.ndarray
        A (4, 4) homogeneous transformation matrix.

    Raises
    ------
    ValueError
        If inputs have invalid shapes or the quaternion has zero norm.
    """
    translation = np.array(translation, dtype=float)
    rotation = np.array(rotation, dtype=float)

    if translation.shape != (3,) or rotation.shape != (4,):
        raise ValueError("Translation must be length 3 and rotation quaternion length 4.")

    norm = np.linalg.norm(rotation)
    if norm == 0:
        raise ValueError("Rotation quaternion cannot be zero.")
    rotation /= norm

    rotation_matrix = Rotation.from_quat(rotation).as_matrix()

    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = translation

    return homogeneous_matrix


def points_within_distance(
    x: float,
    y: float,
    points_with_probs: np.ndarray,
    distance: float
) -> np.ndarray:
    """
    Find all points within a specified distance from a given (x, y) location.

    Parameters
    ----------
    x : float
        X-coordinate of the center point.
    y : float
        Y-coordinate of the center point.
    points_with_probs : np.ndarray
        Array of shape (N, 4), where each row is [x, y, z, prob].
    distance : float
        Maximum allowed distance from the (x, y) center.

    Returns
    -------
    np.ndarray
        Subset of points within the specified distance.
    """
    xy_coordinates = points_with_probs[:, :2]
    distances = np.linalg.norm(xy_coordinates - np.array([x, y]), axis=1)
    mask = distances <= distance
    return points_with_probs[mask]


def pointcloud_to_pointmap(
    pointcloud: np.ndarray,
    voxel_size: float = 0.1,
    x_y_bounds: Tuple[float, float] = (-1.5, 1.5),
    z_bounds: Tuple[float, float] = (-1.4, 0.9)
) -> np.ndarray:
    """
    Convert a point cloud to a voxelized point map.

    Parameters
    ----------
    pointcloud : np.ndarray
        Array of shape (N, 3), representing 3D points.
    voxel_size : float, optional
        Voxel size for discretization.
    x_y_bounds : tuple of float, optional
        (min, max) bounds for x and y dimensions.
    z_bounds : tuple of float, optional
        (min, max) bounds for z dimension.

    Returns
    -------
    np.ndarray
        A voxelized occupancy grid (Z, X, Y).
    """
    if pointcloud.ndim != 2 or pointcloud.shape[1] != 3:
        raise ValueError("Pointcloud must have shape (N, 3).")

    x_y_width = round((x_y_bounds[1] - x_y_bounds[0]) / voxel_size)
    z_width = round((z_bounds[1] - z_bounds[0]) / voxel_size)

    shifted = pointcloud.copy()
    shifted[:, 0] += abs(x_y_bounds[0])
    shifted[:, 1] += abs(x_y_bounds[0])
    shifted[:, 2] += abs(z_bounds[0])

    idx_x = np.floor(shifted[:, 0] / voxel_size).astype(int)
    idx_y = np.floor(shifted[:, 1] / voxel_size).astype(int)
    idx_z = np.floor(shifted[:, 2] / voxel_size).astype(int)

    valid_mask = (
        (idx_x >= 0) & (idx_x < x_y_width) &
        (idx_y >= 0) & (idx_y < x_y_width) &
        (idx_z >= 0) & (idx_z < z_width)
    )

    point_map = np.zeros((z_width, x_y_width, x_y_width), dtype=float)
    point_map[idx_z[valid_mask], idx_x[valid_mask], idx_y[valid_mask]] = 1.0

    return point_map


def pointmap_to_pointcloud(
    pointmap: torch.Tensor,
    voxel_size: float = 0.1,
    x_y_bounds: Tuple[float, float] = (-1.5, 1.5),
    z_bounds: Tuple[float, float] = (-1.4, 0.9),
    prediction_thresh: float = 0.8,
    torch_device: str = "cpu"
) -> np.ndarray:
    """
    Convert a voxelized point map to a point cloud.

    Parameters
    ----------
    pointmap : torch.Tensor
        Voxelized point map (Z, X, Y).
    voxel_size : float, optional
        Size of each voxel.
    x_y_bounds : tuple of float, optional
        (min, max) bounds for x and y dimensions.
    z_bounds : tuple of float, optional
        (min, max) bounds for z dimension.
    prediction_thresh : float, optional
        Threshold for considering a voxel as occupied.
    torch_device : str, optional
        Torch device ("cpu" or "cuda").

    Returns
    -------
    np.ndarray
        Array of shape (N, 3), representing reconstructed 3D points.
    """
    local_pm = pointmap.to(torch_device).cpu().numpy()
    z_idx, x_idx, y_idx = np.where(local_pm > prediction_thresh)

    points_arr = np.vstack([x_idx, y_idx, z_idx]).T.astype(float)

    points_arr[:, 0] = points_arr[:, 0] * voxel_size + voxel_size / 2 - abs(x_y_bounds[0])
    points_arr[:, 1] = points_arr[:, 1] * voxel_size + voxel_size / 2 - abs(x_y_bounds[0])
    points_arr[:, 2] = points_arr[:, 2] * voxel_size + voxel_size / 2 - abs(z_bounds[0])

    return points_arr

