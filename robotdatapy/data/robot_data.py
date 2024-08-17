import numpy as np

from rosbags.highlevel import AnyReader
from pathlib import Path

from robotdatapy.exceptions import NoDataNearTimeException

class RobotData():
    """
    Parent class for easy access to robotics data over time
    """
    def __init__(self, time_tol=.1, interp=False, causal=False):
        self.time_tol = time_tol
        self.interp = interp
        self.causal = causal
        assert not (self.interp and self.causal), "Cannot interpolate and be causal"
        
    def set_t0(self, t0):
        self.times += -self.times[0] + t0
        
    def set_times(self, times):
        self.times = np.array(times)

    @property
    def t0(self):
        return self.times[0]
    
    @property
    def tf(self):
        return self.times[-1]
            
    def idx(self, t, force_single=False, force_double=False):
        """
        Finds the index of pose info closes to the desired time.

        Args:
            t (float): time
            force_single (bool): If True, will only return a single index.
            force_double (bool): If True, will return a list of two indices.

        Returns:
            int: Index of pose info closest to desired time. Returns two indices for using
                interpolating between two pieces of data.
        """
        assert not (force_single and force_double), "Cannot force single and double"
        find_double = (self.interp or force_double) and not force_single

        op1_exists = np.where(self.times <= t)[0].shape[0]
        op2_exists = np.where(self.times >= t)[0].shape[0]
        if (not op1_exists and not op2_exists) or \
            (not op1_exists and self.causal):
            raise NoDataNearTimeException(t_desired=t)
        if op1_exists:
            op1 = np.where(self.times <= t)[0][-1]
        if op2_exists:
            op2 = np.where(self.times >= t)[0][0]

        if self.causal:
            idx = op1
        elif not op1_exists: 
            idx = op2 if not find_double else [op2, op2]
        elif not op2_exists: 
            idx = op1 if not find_double else [op1, op1]
        elif find_double:
            idx = [op1, op2]
        elif abs(t - self.times[op1]) < abs(t - self.times[op2]): 
            idx = op1
        else: 
            idx = op2
        
        if find_double and (abs(self.times[idx[0]] - t) > self.time_tol or \
                            abs(self.times[idx[1]] - t) > self.time_tol):
            raise NoDataNearTimeException(t_desired=t, 
                                          t_closest=[self.times[idx[0]], self.times[idx[1]]])
        elif find_double:
            return idx
        # check to make sure found time is close enough
        elif abs(self.times[idx] - t) > self.time_tol:
            raise NoDataNearTimeException(t_desired=t, t_closest=self.times[idx])
        else: 
            return idx
        
    def get_val(self, vals, t, **kwargs):
        idx = self.idx(t, **kwargs)
        return vals[idx]
    
    def nearest_time(self, t, force_double=False) -> float:
        """
        Returns the time nearest to the desired time.

        Args:
            t (float): Desired time.
            force_double (bool, optional): If set to true, will return the two nearest times 
                (on plus and minus side). Defaults to False.

        Returns:
            float: Nearest time in the data.
        """
        return self.get_val(self.times, t, force_single=not force_double)
    
    def clip(self, t0, tf):
        assert False, "clip function not implemented for this data type."
        
    def __len__(self):
        return len(self.times)
    
    @classmethod
    def topic_t0(cls, bag, topic):
        with AnyReader([Path(bag)]) as reader:
            connections = [x for x in reader.connections if x.topic == topic]
            if len(connections) == 0:
                assert False, f"topic {topic} not found in bag file {bag}"
            for (connection, timestamp, rawdata) in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                try:
                    t = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
                except:
                    t = timestamp
                return t
