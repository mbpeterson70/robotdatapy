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

    def __init__(self, times: np.ndarray, linear_velocities: np.ndarray, 
            angular_velocities: np.ndarray, interp: bool = True, causal: bool = False, 
            time_tol: float = 0.1, R_premultiply: np.ndarray = None, 
            R_postmultiply: np.ndarray = None) -> 'VelData':
        """
        VelData constructor

        Args:
            times (np.ndarray, shape=(n,)): timestamps
            linear_velocities (np.ndarray, shape=(n,3)): x,y,z velocities
            angular_velocities (np.ndarray, shape=(n,3)): velocities about x, y, z axes
            interp (bool, optional): interpolate between closest times, else, choose closest time.
                Defaults to True.
            causal (bool, optional): if true, returns nearest velocity *before* desired time.
                Defaults to False.
            time_tol (float, optional): allowable time difference between desired time and returned 
                data. Defaults to 0.1.
            R_premultiply (np.ndarray, shape=(3,3), optional): Rotation matrix to pre-mulitply
                apply to velocities.
            R_postmultiply (np.ndarray, shape=(3,3), optional): Rotation matrix to post-mulitply
                apply to velocities.

        Returns:
            VelData: VelData object
        """
        super().__init__(time_tol=time_tol, interp=interp, causal=causal)
        self.set_times(np.array(times))
        self.linear_velocities = np.array(linear_velocities)
        self.angular_velocities = np.array(angular_velocities)
        self.R_premultiply = R_premultiply
        self.R_postmultiply = R_postmultiply

    
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

    def lin_vel(self, t: float, multiply=True) -> np.ndarray:
        """
        Linear velocities (x,y,z)

        Args:
            t (np.float): Desired time for retrieving velocity.
            multiply(bool): Apply pre and post rotations.

        Returns:
            np.ndarray, shape(3,): x,y,z linear velocities
        """
        vel = self._unrotated_lin_vel(t)
        if not multiply:
            return vel
        if self.R_premultiply is not None:
            vel = self.R_premultiply @ vel
        if self.R_postmultiply is not None:
            vel = self.R_postmultiply @ vel
        return vel

    def ang_vel(self, t: float, multiply=True) -> np.ndarray:
        """
        Angular velocities (about x,y,z)

        Args:
            t (np.float): Desired time for retrieving velocity.
            multiply(bool): Apply pre and post rotations.

        Returns:
            np.ndarray, shape(3,): x,y,z axis based angular velocities
        """
        vel = self._unrotated_ang_vel(t)
        if not multiply:
            return vel
        if self.R_premultiply is not None:
            vel = self.R_premultiply @ vel
        if self.R_postmultiply is not None:
            vel = self.R_postmultiply @ vel
        return vel

    def _unrotated_lin_vel(self, t: float) -> np.ndarray:
        """
        Linear velocities (x,y,z) without applying pre/post rotations

        Args:
            t (np.float): Desired time for retrieving velocity.

        Returns:
            np.ndarray, shape(3,): x,y,z linear velocities
        """
        return self.get_val(self.linear_velocities, t, interp=self.interp)
            

    def _unrotated_ang_vel(self, t: float) -> np.ndarray:
        """
        Angular velocities (about x,y,z) without applying pre/post rotations

        Args:
            t (np.float): Desired time for retrieving velocity.

        Returns:
            np.ndarray, shape(3,): x,y,z axis based angular velocities
        """
        return self.get_val(self.angular_velocities, t, interp=self.interp)
