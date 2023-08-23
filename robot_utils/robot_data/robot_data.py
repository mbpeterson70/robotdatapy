import numpy as np

class RobotData():
    """
    Parent class for easy access to robotics data over time
    """
    def __init__(self, time_tol=.1, t0=None, interp=False):
        self.time_tol = time_tol
        if t0 is not None:
            self.times -= self.times[0] + t0
        self.interp = interp
            
    def idx(self, t):
        """
        Finds the index of pose info closes to the desired time.

        Args:
            t (float): time

        Returns:
            int: Index of pose info closest to desired time or None if no time is available within 
                tolerance.
        """
        op1_exists = np.where(self.times <= t)[0].shape[0]
        op2_exists = np.where(self.times >= t)[0].shape[0]
        if not op1_exists and not op2_exists:
            idx = None
        if op1_exists:
            op1 = np.where(self.times <= t)[0][-1]
        if op2_exists:
            op2 = np.where(self.times >= t)[0][0]
            
        if not op1_exists: 
            idx = op2 if not self.interp else [op2, op2]
        elif not op2_exists: 
            idx = op1 if not self.interp else [op1, op1]
        elif self.interp:
            idx = [op1, op2]
        elif abs(t - self.times[op1]) < abs(t - self.times[op2]): 
            idx = op1
        else: 
            idx = op2
        
        if self.interp:
            return idx
        
        # check to make sure found time is close enough
        if abs(self.times[idx] - t) > self.time_tol: 
            return None
        else: 
            return idx
        
    def get_val(self, val, t):
        idx = self.idx(t)
        if idx is None:
            return None
        else:
            return val[idx]
