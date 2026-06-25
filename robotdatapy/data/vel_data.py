###########################################################
#
# vel_data.py
#
# Interface for robot velocity data
#
# Authors: Mason Peterson
#
# June 24, 2026
#
###########################################################

# TODO: Implement R_premultiply and R_postmultiply

import numpy as np

from robotdatapy.data.robot_data import RobotData

class VelData(RobotData):

    def __init__(self, times: np.ndarray, linear_velocities: np.ndarray, angular_velocities: np.ndarray, 
            interp: bool = True, causal: bool = False, time_tol: float = 0.1) -> 'VelData':
        """
        VelData constructor

        Args:
            times (np.ndarray, shape=(n,)): timestamps
            linear_velocities (np.ndarray, shape=(n,3)): x,y,z velocities
            angular_velocities (np.ndarray, shape=(n,3)): velocities about x, y, z axes
            interp (bool, optional): interpolate between closest times, else, choose closest time. Defaults to True.
            causal (bool, optional): if true, returns nearest velocity *before* desired time. Defaults to False.
            time_tol (float, optional): allowable time difference between desired time and returned data. Defaults to 0.1.

        Returns:
            VelData: VelData object
        """
        super().__init__(time_tol=time_tol, interp=interp, causal=causal)
        self.set_times(np.array(times))
        self.linear_velocities = np.array(linear_velocities)
        self.angular_velocities = np.array(angular_velocities)

    
    @classmethod
    def from_numpy(cls, times: np.ndarray, linear_velocities: np.ndarray, angular_velocities: np.ndarray, **kwargs):
        """
        VelData constructor

        Args:
            times (np.ndarray, shape=(n,)): timestamps
            linear_velocities (np.ndarray, shape=(n,3)): x,y,z velocities
            angular_velocities (np.ndarray, shape=(n,3)): velocities about x, y, z axes

        Returns:
            VelData: VelData object
        """
        return cls(times, linear_velocities, angular_velocities, **kwargs)

    def lin_vel(self, t: float) -> np.ndarray:
        """
        Linear velocities (x,y,z)

        Args:
            t (np.float): Desired time for retrieving velocity.

        Returns:
            np.ndarray, shape(3,): x,y,z linear velocities
        """
        return self.get_val(self.linear_velocities, t)

    def ang_vel(self, t: float) -> np.ndarray:
        """
        Angular velocities (about x,y,z)

        Args:
            t (np.float): Desired time for retrieving velocity.

        Returns:
            np.ndarray, shape(3,): x,y,z axis based angular velocities
        """
        return self.get_val(self.angular_velocities, t)
