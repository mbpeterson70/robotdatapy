import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import pandas as pd
from rosbags.highlevel import AnyReader
from pathlib import Path

# ROS dependencies
try:
    import cv_bridge
except:
    print("Warning: import cv_bridge failed. Is ROS installed and sourced? " + 
          "Without cv_bridge, the ImgData class may fail.")

class RobotData():
    """
    Parent class for easy access to robotics data over time
    """
    def __init__(self, time_tol=.1, t0=None):
        self.time_tol = time_tol
        if t0 is not None:
            self.times -= self.times[0] + t0
            
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
        
    
class PoseData(RobotData):
    """
    Class for easy access to object poses over time
    """
    
    def __init__(self, data_file, file_type, interp=False, topic=None, time_tol=.1, t0=None, csv_options=None): 
        """
        Class for easy access to object poses over time

        Args:
            data_file (str): File path to data
            file_type (str): 'csv' or 'bag'
            interp (bool): interpolate between closest times, else choose the closest time.
            topic (str, optional): ROS topic, necessary only for bag file_type. Defaults to None.
            time_tol (float, optional): Tolerance used when finding a pose at a specific time. If 
                no pose is available within tolerance, None is returned. Defaults to .1.
            t0 (float, optional): Local time at the first msg. If not set, uses global time from 
                the data_file. Defaults to None.
            csv_option (dict, optional): See _extract_csv_data for details. Defaults to None.
        """
        if file_type == 'csv':
            self._extract_csv_data(data_file, csv_options)
        elif file_type == 'bag':
            self._extract_bag_data(data_file, topic)
        else:
            assert False, "file_type not supported, please choose from: csv or bag2"
        self.interp = interp
        super().__init__(time_tol=time_tol, t0=t0)
        
    
    def _extract_csv_data(self, csv_file, csv_options):
        """
        Extracts pose data from csv file with 9 columns for time (sec/nanosec), position, and orientation

        Args:
            csv_file (str): CSV file path
            csv_options (dict): Can include dict of structure: dict['col'], dict['col_nums'] which 
                map to dicts containing keys of 'time', 'position', and 'orientation' and the 
                corresponding column names and numbers. csv_options['timescale'] can be given if 
                the time column is not in seconds.
        """
        if csv_options is None:
            pose_df = pd.read_csv(csv_file, usecols=['header.stamp.secs', 'header.stamp.nsecs', 'pose.position.x', 'pose.position.y', 'pose.position.z',
                'pose.orientation.x', 'pose.orientation.y', 'pose.orientation.z', 'pose.orientation.w'])
            self.positions = pd.DataFrame.to_numpy(pose_df.iloc[:, 2:5])
            self.orientations = pd.DataFrame.to_numpy(pose_df.iloc[:, 5:9])
            self.times =( pd.DataFrame.to_numpy(pose_df.iloc[:,0:1]) + pd.DataFrame.to_numpy(pose_df.iloc[:,1:2])*1e-9).reshape(-1)
        else:
            cols = csv_options['cols']
            pose_df = pd.read_csv(csv_file, usecols=cols['time'] + cols['position'] + cols['orientation'])
            
            if 'col_nums' in csv_options:
                t_cn = csv_options['col_nums']['time']
                pos_cn = csv_options['col_nums']['position']
                ori_cn = csv_options['col_nums']['orientation']
            else:
                t_cn, pos_cn, ori_cn = [0], [1,2,3], [4,5,6,7]
            self.positions = pd.DataFrame.to_numpy(pose_df.iloc[:, pos_cn])
            self.orientations = pd.DataFrame.to_numpy(pose_df.iloc[:, ori_cn])
            self.times = pd.DataFrame.to_numpy(pose_df.iloc[:,t_cn]).astype(np.float64).reshape(-1)

            if 'timescale' in csv_options:
                self.times *= csv_options['timescale']
        return
    
    def _extract_bag_data(self, bag_file, topic):
        """
        Extracts pose data from ROS bag file. Assumes msg is of type PoseStamped.

        Args:
            bag_file (str): ROS bag file path
            topic (str): ROS pose topic
        """
        times = []
        positions = []
        orientations = []
        with AnyReader([Path(bag_file)]) as reader:
            connections = [x for x in reader.connections if x.topic == topic]
            for (connection, timestamp, rawdata) in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                times.append(msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9)
                positions.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
                orientations.append([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        self.times = np.array(times)
        self.positions = np.array(positions)
        self.orientations = np.array(orientations)
                
    def position(self, t):
        """
        Position at time t.

        Args:
            t (float): time

        Returns:
            np.array, shape(3,): position in xyz
        """
        idx = self.idx(t)
        if idx is None:
            return None
        if self.interp:
            if np.allclose(*self.times[idx].tolist()):
                return self.positions[idx[0]]
            return self.positions[idx[0]] + \
                (self.positions[idx[1]] - self.positions[idx[0]]) * \
                (t - self.times[idx[0]]) / (self.times[idx[1]] - self.times[idx[0]])
        else:
            return self.positions[idx]
    
    def orientation(self, t):
        """
        Orientation at time t.

        Args:
            t (float): time

        Returns:
            np.array, shape(4,): orientation as a quaternion
        """
        idx = self.idx(t)
        if idx is None:
            return None
        
        if self.interp:
            if np.allclose(*self.times[idx].tolist()):
                return self.orientations[idx[0]]
            orientations = Rot.from_quat(self.orientations[idx])
            slerp = Slerp(self.times[idx], orientations)
            return slerp(t).as_quat()
        else:
            return self.orientations[idx]
    
    def T_WB(self, t):
        """
        Transform from world to body frame (or pose of body within world frame) at time t.

        Args:
            t (float): time

        Returns:
            np.array, shape(4,4): Rigid body transform
        """
        position = self.position(t)
        orientation = self.orientation(t)
        if position is None or orientation is None:
            return None
        T_WB = np.eye(4)
        T_WB[:3,:3] = Rot.from_quat(orientation).as_matrix()
        T_WB[:3,3] = position
        return T_WB
        
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
            self._extract_bag_data(data_file, topic, time_range, compressed)
        else:
            assert False, "file_type not supported, please choose from: bag"

        self.compressed = compressed
        self.data_file = data_file
        self.file_type = file_type
        self.interp = False
        self.bridge = cv_bridge.CvBridge()
        super().__init__(time_tol=time_tol, t0=t0)
            
    def _extract_bag_data(self, bag_file, topic, time_range):
        """
        Extracts pose data from ROS bag file. Assumes msg is of type PoseStamped.

        Args:
            bag_file (str): ROS bag file path
            topic (str): ROS pose topic
            time_range (list, shape=(2,), optional): Two element list indicating range of times
                that should be stored within object
        """
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
                if t < time_range[0]:
                    continue
                elif t > time_range[1]:
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