import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from pathlib import Path
from copy import deepcopy
import os

from robotdatapy.data.robot_data import RobotData
    
class GeneralData(RobotData):
    """
    Class for easy access to generic robot data over time
    """
    
    def __init__(
        self, 
        data=None, 
        times=None, 
        time_tol=.1, 
        causal=False, 
        t0=None,
    ): 
        """
        Class for easy access to object poses over time

        Args:
            data (list): list of data items corresponding to times
            times (list): list of times corresponding to data items
            time_tol (float, optional): Tolerance used when finding a pose at a specific time. If 
                no pose is available within tolerance, None is returned. Defaults to .1.
            causal (bool): if True, only use data that is available at the time requested.
            t0 (float, optional): Local time at the first msg. If not set, uses global time from 
                the data_file. Defaults to None.
        """
        super().__init__(time_tol=time_tol, interp=False, causal=causal)
        self._data = deepcopy(data)
        self.set_times(np.array(times))
        if t0 is not None:
            self.set_t0(t0) 
            
    @classmethod
    def from_bag(cls, path, topic, field=None, custom_msg_types=None, custom_msg_paths=None, ros_distro=None, **kwargs):
        """
        Create GeneralData object from bag file

        Args:
            path (str): ROS bag file path
            topic (str): ROS pose topic
            field (str, optional): ROS topic field if the whole field is not required. Subfields 
                should be separated by '/'. Defaults to None.
            custom_msg_types(str/List(str), optional): A list of custom message types used in the 
                bag file data. Example: ['bar_msgs/msg/Bar', 'foo_msgs/msg/Foo']. Defaults to None.
            custom_msg_paths(str/List(str), optional): A list of paths to custom message types used.
            ros_distro (str, optional): ROS version. Choose from ['foxy']. Defaults to None.

        Returns:
            GeneralData: GeneralData object
        """
        assert topic is not None, "topic must be provided"
        assert (custom_msg_types is None) == (custom_msg_paths is None), "custom_msg_types and custom_msg_paths must be provided together"
        assert custom_msg_types is None or ros_distro is not None, "ros_distro must be provided if custom_msg_types is not None"
        if ros_distro is None:
            typestore = None
        elif ros_distro == 'foxy':
            typestore = get_typestore(Stores.ROS2_FOXY)
        elif ros_distro == 'humble':
            typestore = get_typestore(Stores.ROS2_HUMBLE)
        else:
            raise ValueError("ros_distro must be one of ['foxy']")
        if custom_msg_types is not None:
            typestore = cls._register_custom_msg_types(custom_msg_types, custom_msg_paths, typestore)
        bag_file = os.path.expanduser(os.path.expandvars(path))
        
        times = []
        data = []
        if field is not None:
            sub_attrs = field.strip().split('/')
        with AnyReader([Path(bag_file)], default_typestore=typestore) as reader:
            connections = [x for x in reader.connections if x.topic == topic]
            for (connection, timestamp, rawdata) in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                header = getattr(msg, 'header', None)
                if header is None:
                    times.append(timestamp*1e-9)
                else:
                    times.append(msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9)
                item = msg
                if field is not None:
                    for attr in sub_attrs:
                         item = getattr(item, attr)
                data.append(item)
        return cls(data, times, **kwargs)
            
    @classmethod
    def _register_custom_msg_types(cls, custom_msg_types, custom_msg_paths, typestore):
        """
        Registers custom message types for ROS serialization.

        Args:
            custom_msg_types (str/List(str)): A list of custom message types used in the bag file data.
        """
        if isinstance(custom_msg_types, str):
            custom_msg_types = [custom_msg_types]
        if isinstance(custom_msg_paths, str):
            custom_msg_paths = [custom_msg_paths]
        add_types = {}
        for msg_type, msg_path in zip(custom_msg_types, custom_msg_paths):
            add_types.update(get_types_from_msg(Path(msg_path).read_text(), msg_type)) 
        typestore.register(add_types)
        return typestore
            
    def data(self, t):
        """
        Data at time t.

        Args:
            t (float): time

        Returns:
            any: data item at time t
        """
        return self.get_val(self._data, t)