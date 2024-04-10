import numpy as np
from rosbags.highlevel import AnyReader
from pathlib import Path
from copy import deepcopy
import os

from robot_utils.robot_data.robot_data import RobotData
    
class GeneralData(RobotData):
    """
    Class for easy access to generic robot data over time
    """
    
    def __init__(self, data_type, data_file=None, data=None, times=None, topic=None, field=None, time_tol=.1, causal=False, t0=None): 
        """
        Class for easy access to object poses over time

        Args:
            data_type (str): 'bag', 'list'
            data_file (str): File path to data
            topic (str, optional): ROS topic, necessary only for bag file_type. Defaults to None.
            field (str, optional): ROS topic field if the whole field is not required. Subfields 
                should be separated by '/'. Defaults to None.
            time_tol (float, optional): Tolerance used when finding a pose at a specific time. If 
                no pose is available within tolerance, None is returned. Defaults to .1.
            t0 (float, optional): Local time at the first msg. If not set, uses global time from 
                the data_file. Defaults to None.
        """
        super().__init__(time_tol=time_tol, interp=False, causal=causal)
        assert data_type == 'bag' or data_type == 'list', "only bag or lists supported currently"
        if data_type == 'bag':
            assert topic is not None, "topic must be provided"
            self._extract_bag_data(bag_file=os.path.expanduser(os.path.expandvars(data_file)), topic=topic, field=field)
        elif data_type == 'list':
            assert data is not None and times is not None
            self._data = deepcopy(data)
            self.set_times(np.array(times))
        if t0 is not None:
            self.set_t0(t0)                
            
    def _extract_bag_data(self, bag_file, topic, field=None):
        """
        Extracts pose data from ROS bag file. Assumes msg is of type PoseStamped.

        Args:
            bag_file (str): ROS bag file path
            topic (str): ROS pose topic
            field (str, optional): ROS topic field if the whole field is not required. Subfields 
                should be separated by '/'. Defaults to None.
        """
        times = []
        data = []
        if field is not None:
            sub_attrs = field.strip().split('/')
        with AnyReader([Path(bag_file)]) as reader:
            connections = [x for x in reader.connections if x.topic == topic]
            for (connection, timestamp, rawdata) in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                times.append(msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9)
                item = msg
                if field is not None:
                    for attr in sub_attrs:
                         item = getattr(item, attr)
                data.append(item)
        self.set_times(np.array(times))
        self._data = data
            
    def data(self, t):
        """
        Data at time t.

        Args:
            t (float): time

        Returns:
            any: data item at time t
        """
        return self.get_val(self._data, t)