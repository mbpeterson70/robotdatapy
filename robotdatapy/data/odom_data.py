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

# TODO: implement T_premultiply and T_postmultiply

import numpy as np

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
