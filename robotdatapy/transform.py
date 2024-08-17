import numpy as np
from scipy.spatial.transform import Rotation as Rot
from typing import List

def transform_vec(T, vec):
    unshaped_vec = vec.reshape(-1)
    resized_vec = np.concatenate(
        [unshaped_vec, np.zeros((T.shape[0] - 1 - unshaped_vec.shape[0]))]).reshape(-1)
    resized_vec = np.concatenate(
        [resized_vec, np.ones((T.shape[0] - resized_vec.shape[0]))]).reshape((-1, 1))
    transformed = T @ resized_vec
    return transformed.reshape(-1)[:unshaped_vec.shape[0]].reshape(vec.shape) 

def transform(T, vecs, axis=0):
    if len(vecs.reshape(-1)) == 2 or len(vecs.reshape(-1)) == 3:
        return transform_vec(T, vecs)
    vecs_horz_stacked = vecs if axis==1 else vecs.T
    zero_padded_vecs = np.vstack(
        [vecs_horz_stacked, np.zeros((T.shape[0] - 1 - vecs_horz_stacked.shape[0], vecs_horz_stacked.shape[1]))]
    )
    one_padded_vecs = np.vstack(
        [zero_padded_vecs, np.ones((1, vecs_horz_stacked.shape[1]))]
    )
    transformed = T @ one_padded_vecs
    transformed = transformed[:vecs_horz_stacked.shape[0],:] 
    return transformed if axis == 1 else transformed.T

# gives scalar value to magnitude of translation
def T_mag(T, deg2m):
    R = Rot.from_matrix(T[0:3, 0:3])
    t = T[0:2, 3]
    rot_mag = R.as_euler('xyz', degrees=True)[2] / deg2m
    t_mag = np.linalg.norm(t)
    return np.abs(rot_mag) + np.abs(t_mag)

def transform_to_xytheta(T):
    dim = T.shape[1] - 1
    if dim == 2:
        T = T2d_2_T3d(T)
    x = T[0,3]
    y = T[1,3]
    psi = Rot.from_matrix(T[:3,:3]).as_euler('xyz', degrees=False)[2]
    return x, y, psi

def transform_to_xyzrpy(T, degrees=False):
    """
    Converts a 4x4 rigid body transformation matrix to a 6D vector of x, y, z, roll, pitch, yaw

    Args:
        T (np.array, shape=(4,4)): Rigid body transformation matrix
        degrees (bool, optional): Report roll, pitch, and yaw in degrees rather than radians. Defaults to False.

    Returns:
        np.array, shape=(6,): 6 dimensional vector of x, y, z, roll, pitch, yaw
    """
    assert T.shape[0] == T.shape[1], "T must be square"
    assert T.shape[0] == 4
    xyzrpy = np.zeros(6)
    xyzrpy[:3] = T[:3,3]
    xyzrpy[3:] = Rot.from_matrix(T[:3,:3]).as_euler('ZYX', degrees=degrees)[::-1]
    return xyzrpy

def transform_to_xyz_quat(T, separate=False):
    """
    Converts a 4x4 rigid body transformation matrix to a 6D vector of x, y, z, and quaternion (x, y, z, w)

    Args:
        T (np.array, shape=(4,4)): Rigid body transformation matrix
        separate (bool, optional): Return translation and quaternion as two separate arrays. 
            Defaults to False.

    Returns:
        np.array, shape=(7,): 7 dimensional vector of x, y, z, qx, qy, qz, qw
    """
    assert T.shape[0] == T.shape[1], "T must be square"
    assert T.shape[0] == 4
    xyz_quat = np.zeros(7)
    xyz_quat[:3] = T[:3,3]
    xyz_quat[3:] = Rot.from_matrix(T[:3,:3]).as_quat()
    if separate:
        return xyz_quat[:3], xyz_quat[3:]
    else:
        return xyz_quat

def xytheta_to_transform(x, y, psi, degrees=False, dim=3):
    assert dim == 3 or dim == 2, "supports dimension 2 or 3 only"
    T = np.eye(dim + 1)
    T[:2,:2] = Rot.from_euler('xyz', [0, 0, psi], degrees=degrees).as_matrix()[:2,:2]
    T[0,dim] = x
    T[1,dim] = y
    return T

def xyz_quat_to_transform(xyz, quat):
    xyz = np.array(xyz)
    quat = np.array(quat)
    T = np.eye(4)
    T[:3,:3] = Rot.from_quat(quat).as_matrix()
    T[:3,3] = xyz.reshape(-1)
    return T

def T3d_2_T2d(T3d):
    print("The function T3d_2_T2d is deprecated. " + 
          "Does not handle 3D to 2D conversion properly " +
          "(when roll and pitch are involved).")
    T2d = np.delete(np.delete(T3d, 2, axis=0), 2, axis=1)
    return T2d

def T2d_2_T3d(T2d):
    T3d = np.eye(4)
    T3d[:2,:2] = T2d[:2,:2]
    T3d[:2,3] = T2d[:2,2]
    return T3d

# from camera to body FLU coordinates transform
T_FLURDF = np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
], dtype=np.float64)

# from body FLU to camera coordinates transform
T_RDFFLU = np.array([
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
], dtype=np.float64)

def aruns(pts1, pts2, weights=None):
    """Aruns method for 3D registration

    Args:
        pts1 (numpy.array, shape(n,2 or 3)): initial set of points
        pts2 (numpy.array, shape(n,2 or 3)): second set of points that should be aligned to first set
        weights (numpy.array, shape(n), optional): weights applied to associations. Defaults to None.

    Returns:
        numpy.array, shape(3,3 or 4,4): rigid body transformation to align pts2 with pts1
    """
    if weights is None:
        weights = np.ones((pts1.shape[0],1))
    weights = weights.reshape((-1,1))
    mean1 = (np.sum(pts1 * weights, axis=0) / np.sum(weights)).reshape(-1)
    mean2 = (np.sum(pts2 * weights, axis=0) / np.sum(weights)).reshape(-1)
    pts1_mean_reduced = pts1 - mean1
    pts2_mean_reduced = pts2 - mean2
    assert pts1_mean_reduced.shape == pts2_mean_reduced.shape
    H = pts1_mean_reduced.T @ (pts2_mean_reduced * weights)
    U, s, Vh = np.linalg.svd(H)
    R = U @ Vh
    if np.allclose(np.linalg.det(R), -1.0):
        Vh_prime = Vh.copy()
        Vh_prime[-1,:] *= -1.0
        R = U @ Vh_prime
    t = mean1.reshape((-1,1)) - R @ mean2.reshape((-1,1))
    T = np.concatenate([np.concatenate([R, t], axis=1), np.hstack([np.zeros((1, R.shape[0])), [[1]]])], axis=0)
    return T

def mean(transforms: List[np.ndarray]) -> np.ndarray:
    """
    Finds the mean rigid transform from a list of transforms

    Args:
        transforms (List[np.ndarray, shape=(3,3) or (4,4)]): list of transforms to average

    Returns:
        np.ndarray, shape=(3,3) or (4,4): mean transform
    """
    assert len(transforms) > 0, "transforms must have at least one element"
    shape = transforms[0].shape
    assert all([T.shape == shape for T in transforms]), "All transforms must have the same shape"

    if shape == (3,3):
        assert False, "Not implemented"
    elif shape == (4,4):
        mean_rot = Rot.mean(Rot.from_matrix([T[:3,:3] for T in transforms]))
        mean_t = np.mean([T[:3,3] for T in transforms], axis=0)
        T = np.eye(4)
        T[:3,:3] = mean_rot.as_matrix()
        T[:3,3] = mean_t
        return T