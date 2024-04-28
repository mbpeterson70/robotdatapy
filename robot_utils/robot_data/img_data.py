import numpy as np
import os
from rosbags.highlevel import AnyReader
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt

from robot_utils.robot_data.robot_data import RobotData
from robot_utils.exceptions import MsgNotFound
from robot_utils.camera import CameraParams
# TODO: support non-rvl compressed depth images

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
    
    def __init__(
            self, 
            data_file, 
            file_type, 
            topic=None, 
            time_tol=.1, 
            causal=False, 
            t0=None, 
            time_range=None, 
            compressed=True,
            compressed_encoding='passthrough',
            compressed_rvl=False
        ): 
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
        super().__init__(time_tol=time_tol, interp=False, causal=causal)
        data_file = os.path.expanduser(os.path.expandvars(data_file))
        if file_type == 'bag' or file_type == 'bag2':
            self._extract_bag_data(data_file, topic, time_range)
        else:
            assert False, "file_type not supported, please choose from: bag"

        self.compressed = compressed
        self.compressed_encoding = compressed_encoding
        self.compressed_rvl = compressed_rvl
        self.data_file = data_file
        self.file_type = file_type
        self.interp = False
        self.bridge = cv_bridge.CvBridge()
        if t0 is not None:
            self.set_t0(t0)
            
        self.camera_params = CameraParams()
            
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
            if len(connections) == 0:
                assert False, f"topic {topic} not found in bag file {bag_file}"
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
        self.set_times(np.array(sorted(times)))
    
    def extract_params(self, topic, ):
        """
        Get camera parameters

        Args:
            topic (str): ROS topic name containing cameraInfo msg

        Returns:
            np.array, shape=(3,3): camera intrinsic matrix K
        """
        if self.file_type == 'bag' or self.file_type == 'bag2':
            with AnyReader([Path(self.data_file)]) as reader:
                connections = [x for x in reader.connections if x.topic == topic]
                if len(connections) == 0:
                    assert False, f"topic {topic} not found in bag file {self.data_file}"
                for (connection, timestamp, rawdata) in reader.messages(connections=connections):
                    if connection.topic == topic:
                        msg = reader.deserialize(rawdata, connection.msgtype)
                        try:
                            K = np.array(msg.K).reshape((3,3))
                            D = np.array(msg.D)
                        except:
                            K = np.array(msg.k).reshape((3,3))
                            D = np.array(msg.d)
                        width = msg.width
                        height = msg.height
                        self.camera_params = CameraParams(K, D, width, height)
                        break
        else:
            assert False, "file_type not supported, please choose from: bag"
        return self.camera_params.K, self.camera_params.D
        
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
            img = self.bridge.imgmsg_to_cv2(self.img_msgs[idx], desired_encoding=self.compressed_encoding)
        elif self.compressed_rvl:
            from rvl import decompress_rvl
            assert self.width is not None and self.height is not None
            img = decompress_rvl(
                np.array(self.img_msgs[idx].data[20:]).astype(np.int8), 
                self.height*self.width).reshape((self.height, self.width))
        else:
            img = self.bridge.compressed_imgmsg_to_cv2(self.img_msgs[idx], desired_encoding=self.compressed_encoding)
        return img
    
    def show(self, t, ax=None):
        """
        Show image at time t.

        Args:
            t (float): time
        """
        img = self.img(t)
        if ax is None:
            _, ax = plt.subplots()
        if len(img.shape) == 3:
            ax.imshow(img[...,::-1])
        else:
            ax.imshow(img)
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        return ax
    
    @property
    def K(self):
        return self.camera_params.K
    
    @property
    def D(self):
        return self.camera_params.D
    
    @property
    def width(self):
        return self.camera_params.width
    
    @property
    def height(self):
        return self.camera_params.height
    
    @property
    def T(self):
        return self.camera_params.T
