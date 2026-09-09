###########################################################
#
# odom_data.py
#
# Interface for robot odometry (position / velocity) data
#
# Authors: Mason Peterson
#
# June 24, 2026
#
###########################################################

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from robotdatapy.data.robot_data import RobotData
from robotdatapy.data.pose_data import PoseData
from robotdatapy.data.vel_data import VelData

class OdomData(RobotData):

    def __init__(self, times: np.ndarray, positions: np.ndarray, orientations: np.ndarray,
            linear_velocities: np.ndarray, angular_velocities: np.ndarray,
            interp: bool = True, causal: bool = False, time_tol: float = 0.1,
            T_premultiply=None, T_postmultiply=None) -> 'OdomData':
        """
        OdomData constructor

        Args:
            times (np.ndarray, shape=(n,)): timestamps
            positions (np.array, shape(n,3)): xyz positions of the poses
            orientations (np.array, shape(n,4)): quaternions of the poses
            linear_velocities (np.ndarray, shape=(n,3)): x,y,z velocities
            angular_velocities (np.ndarray, shape=(n,3)): velocities about x, y, z axes
            interp (bool, optional): interpolate between closest times, else, choose closest time. Defaults to True.
            causal (bool, optional): if true, returns nearest velocity *before* desired time. Defaults to False.
            time_tol (float, optional): allowable time difference between desired time and returned data. Defaults to 0.1.
            T_premultiply (np.array, shape(4,4)): Rigid transform to premultiply to the pose.
            T_postmultiply (np.array, shape(4,4)): Rigid transform to postmultiply to the pose.

        Returns:
            OdomData: OdomData object
        """
        super().__init__(time_tol=time_tol, interp=interp, causal=causal)
        self.set_times(np.array(times))
        self.pose_data = PoseData(
            times=times,
            positions=positions,
            orientations=orientations,
            interp=interp,
            causal=causal,
            time_tol=time_tol,
            T_premultiply=T_premultiply,
            T_postmultiply=T_postmultiply
        )

        self.vel_data = VelData.from_numpy(
            times=times,
            linear_velocities=linear_velocities,
            angular_velocities=angular_velocities,
            interp=interp,
            causal=causal,
            time_tol=time_tol,
            R_premultiply=T_premultiply[:3,:3] if T_premultiply is not None else None,
            R_postmultiply=T_postmultiply[:3,:3] if T_postmultiply is not None else None
        )

    @classmethod
    def from_numpy(cls, times: np.ndarray, positions: np.ndarray, orientations: np.ndarray,
            linear_velocities: np.ndarray, angular_velocities: np.ndarray, **kwargs) -> 'OdomData':
        """
        OdomData constructor

        Args:
            times (np.ndarray, shape=(n,)): timestamps
            positions (np.array, shape(n,3)): xyz positions of the poses
            orientations (np.array, shape(n,4)): quaternions of the poses
            linear_velocities (np.ndarray, shape=(n,3)): x,y,z velocities
            angular_velocities (np.ndarray, shape=(n,3)): velocities about x, y, z axes

        Returns:
            OdomData: OdomData object
        """
        return cls(times=times, positions=positions, orientations=orientations,
            linear_velocities=linear_velocities, angular_velocities=angular_velocities, **kwargs)

    @classmethod
    def from_npy(cls, filename, **kwargs) -> 'OdomData':
        """
        Generates an OdomData object from a npy file. Numpy array structure should be n x 14, 
        in the following order: time, positions (xyz), orientations (xyzw), linear velocities,
        angular velocities.

        Args:
            filename (str): .npy file

        Returns:
            OdomData: OdomData object
        """
        array: np.ndarray = np.load(filename)
        return cls(
            times=array[:,0],
            positions=array[:,1:4],
            orientations=array[:,4:8],
            linear_velocities=array[:,8:11],
            angular_velocities=array[:,11:14],
            **kwargs
        )

    @classmethod
    def from_pose_data(cls, pose_data: PoseData, interp: bool = True, causal: bool = False,
            time_tol: float = 0.1) -> 'OdomData':
        """
        Builds an OdomData from a PoseData by finite-differencing the pose stream.

        Velocities are estimated at each original pose timestamp, so all N samples
        are kept and remain co-located with the poses:
            - linear: central difference of position, one-sided at the endpoints,
              v[k] = (p[k+1] - p[k-1]) / (t[k+1] - t[k-1]).
            - angular: world frame, from the relative rotation over the same
              interval, dR = R[k+1] * R[k-1]^-1, omega[k] = rotvec(dR) / dt.

        Any T_premultiply / T_postmultiply already set on ``pose_data`` is baked
        into the differenced world poses (obtained via ``all_poses``); the returned
        OdomData carries no additional transform.

        Non-advancing timestamps (dt <= 0, e.g. duplicate samples) forward-carry
        the last finite linear velocity rather than dividing by zero, and a final
        nan_to_num guarantees finite output regardless of the input timing.

        Args:
            pose_data (PoseData): source poses.
            interp (bool, optional): passed to the OdomData constructor. Defaults to True.
            causal (bool, optional): passed to the OdomData constructor. Defaults to False.
            time_tol (float, optional): passed to the OdomData constructor. Defaults to 0.1.

        Returns:
            OdomData: odometry with finite-differenced linear and angular velocities.
        """
        poses = pose_data.all_poses()  # (n,4,4) with any transforms applied
        times = np.asarray(pose_data.times, dtype=float)
        positions = poses[:, :3, 3]
        rotations = Rot.from_matrix(poses[:, :3, :3])
        orientations = rotations.as_quat()  # xyzw

        n = len(times)
        linear_velocities = np.zeros((n, 3))
        angular_velocities = np.zeros((n, 3))
        last_lin = np.zeros(3)
        for k in range(n):
            kp = min(k + 1, n - 1)
            km = max(k - 1, 0)
            dt = times[kp] - times[km]
            if dt <= 0:
                # duplicate / non-advancing timestamp: forward-carry last finite
                # velocity instead of dividing by zero.
                linear_velocities[k] = last_lin
                continue
            linear_velocities[k] = (positions[kp] - positions[km]) / dt
            last_lin = linear_velocities[k]
            dR = rotations[kp] * rotations[km].inv()  # relative rotation (world frame)
            angular_velocities[k] = dR.as_rotvec() / dt
        linear_velocities = np.nan_to_num(linear_velocities, nan=0.0, posinf=0.0, neginf=0.0)
        angular_velocities = np.nan_to_num(angular_velocities, nan=0.0, posinf=0.0, neginf=0.0)

        return cls(times=times, positions=positions, orientations=orientations,
            linear_velocities=linear_velocities, angular_velocities=angular_velocities,
            interp=interp, causal=causal, time_tol=time_tol)

    def set_T_premultiply(self, T_premultiply: np.ndarray):
        self.pose_data.T_premultiply = T_premultiply
        self.vel_data.R_premultiply = T_premultiply[:3,:3] if T_premultiply is not None else None
        return

    def set_T_postmultiply(self, T_postmultiply: np.ndarray):
        self.pose_data.T_postmultiply = T_postmultiply
        self.vel_data.R_postmultiply = T_postmultiply[:3,:3] if T_postmultiply is not None else None
        return
    
    def lin_vel(self, t: float) -> np.ndarray:
        """
        Linear velocities (x,y,z)

        Args:
            t (np.float): Desired time for retrieving velocity.

        Returns:
            np.ndarray, shape(3,): x,y,z linear velocities
        """
        return self.vel_data.lin_vel(t)

    def ang_vel(self, t: float) -> np.ndarray:
        """
        Angular velocities (about x,y,z)

        Args:
            t (np.float): Desired time for retrieving velocity.

        Returns:
            np.ndarray, shape(3,): x,y,z axis based angular velocities
        """
        return self.vel_data.ang_vel(t)

    def position(self, t: float) -> np.ndarray:
        """
        Position at time t.

        Args:
            t (float): time

        Returns:
            np.array, shape(3,): position in xyz
        """
        return self.pose_data.position(t)

    def orientation(self, t: float) -> np.ndarray:
        """
        Orientation at time t.

        Args:
            t (float): time

        Returns:
            np.array, shape(4,): orientation as a quaternion (x, y, z, w)
        """
        return self.pose_data.orientation(t)

    def pose(self, t: float) -> np.ndarray:
        """
        Pose at time t.

        Args:
            t (float): time

        Returns:
            np.array, shape(4,4): Rigid body transform
        """
        return self.pose_data.pose(t)
