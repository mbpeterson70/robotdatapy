import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader
from pathlib import Path
from robot_utils.robot_data.robot_data import RobotData

# ROS dependencies
try:
    import cv_bridge
except:
    print("Warning: import cv_bridge failed. Is ROS installed and sourced? " + 
          "Without cv_bridge, the ImgData class may fail.")    
        
class ImgData(RobotData):
    """
    Class for easy access to image data over time
    """
    
    def __init__(self, data_file, file_type, topic=None, time_tol=.1, t0=None, time_range=None, compressed=True): 
        """
        Class for easy access to image data over time

        Args:
            data_file (str): File path to data
            file_type (str): only 'bag' supported now
            topic (str, optional): ROS topic, necessary only for bag file_type. Defaults to None.
            time_tol (float, optional): Tolerance used when finding a pose at a specific time. If 
                no pose is available within tolerance, None is returned. Defaults to .1.
            t0 (float, optional): Local time at the first msg. If not set, uses global time from 
                the data_file. Defaults to None.
            time_range (list, shape=(2,), optional): Two element list indicating range of times
                (before being offset with t0) that should be stored within object
            compressed (bool, optional): True if data_file contains compressed images
        """        
        if file_type == 'bag':
            self._extract_bag_data(data_file, topic, time_range)
        else:
            assert False, "file_type not supported, please choose from: bag"

        self.compressed = compressed
        self.data_file = data_file
        self.file_type = file_type
        self.interp = False
        self.bridge = cv_bridge.CvBridge()
        super().__init__(time_tol=time_tol, t0=t0, interp=False)
            
    def _extract_bag_data(self, bag_file, topic, time_range=None):
        """
        Extracts pose data from ROS bag file. Assumes msg is of type PoseStamped.

        Args:
            bag_file (str): ROS bag file path
            topic (str): ROS pose topic
            time_range (list, shape=(2,), optional): Two element list indicating range of times
                that should be stored within object
        """
        if time_range is not None:
            assert time_range[0] < time_range[1], "time_range must be given in incrementing order"
        
        times = []
        img_msgs = []
        with AnyReader([Path(bag_file)]) as reader:
            connections = [x for x in reader.connections if x.topic == topic]
            for (connection, timestamp, rawdata) in reader.messages(connections=connections):
                if connection.topic != topic:
                    continue
                msg = reader.deserialize(rawdata, connection.msgtype)
                t = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
                if time_range is not None and t < time_range[0]:
                    continue
                elif time_range is not None and t > time_range[1]:
                    break

                times.append(t)
                img_msgs.append(msg)
        
        self.img_msgs = [msg for _, msg in sorted(zip(times, img_msgs), key=lambda zipped: zipped[0])]
        self.times = np.array(sorted(times))
    
    def extract_params(self, topic):
        """
        Get camera parameters

        Args:
            topic (str): ROS topic name containing cameraInfo msg

        Returns:
            np.array, shape=(3,3): camera intrinsic matrix K
        """
        if self.file_type == 'bag':
            self.K = None
            with AnyReader([Path(self.data_file)]) as reader:
                connections = [x for x in reader.connections if x.topic == topic]
                for (connection, timestamp, rawdata) in reader.messages(connections=connections):
                    if connection.topic == topic:
                        msg = reader.deserialize(rawdata, connection.msgtype)
                        self.K = np.array(msg.k).reshape((3,3))
        else:
            assert False, "file_type not supported, please choose from: bag"
        return self.K
        
    def img(self, t):
        """
        Image at time t.

        Args:
            t (float): time

        Returns:
            cv image
        """
        idx = self.idx(t)
        if idx is None:
            return None
        elif not self.compressed:
            img = self.bridge.imgmsg_to_cv2(self.img_msgs[idx], desired_encoding='bgr8')
        else:
            img = self.bridge.compressed_imgmsg_to_cv2(self.img_msgs[idx], desired_encoding='bgr8')
        return img