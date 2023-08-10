import numpy as np
from scipy.spatial.transform import Rotation as Rot
import pandas as pd
from rosbags.highlevel import AnyReader
from pathlib import Path

class PoseData():
    """
    Class for easy access to object poses over time
    """
    
    def __init__(self, data_file, file_type, topic=None, time_tol=.1, t0=None): 
        """
        Class for easy access to object poses over time

        Args:
            data_file (str): File path to data
            file_type (str): 'csv' or 'bag'
            topic (str, optional): ROS topic, necessary only for bag file_type. Defaults to None.
            time_tol (float, optional): Tolerance used when finding a pose at a specific time. If 
                no pose is available within tolerance, None is returned. Defaults to .1.
            t0 (float, optional): Local time at the first msg. If not set, uses global time from 
                the data_file. Defaults to None.
        """
        if file_type == 'csv':
            self._extract_csv_data(data_file)
        elif file_type == 'bag':
            self._extract_bag_data(data_file, topic)
        else:
            assert False, "file_type not supported, please choose from: csv or bag2"
        self.time_tol = time_tol
        if t0 is not None:
            self.times -= self.times[0] + t0
    
    def _extract_csv_data(self, csv_file):
        """
        Extracts pose data from csv file with 9 columns for time (sec/nanosec), position, and orientation

        Args:
            csv_file (str): CSV file path
        """
        pose_df = pd.read_csv(csv_file, usecols=['header.stamp.secs', 'header.stamp.nsecs', 'pose.position.x', 'pose.position.y', 'pose.position.z',
            'pose.orientation.x', 'pose.orientation.y', 'pose.orientation.z', 'pose.orientation.w'])
        self.positions = pd.DataFrame.to_numpy(pose_df.iloc[:, 2:5])
        self.orientations = pd.DataFrame.to_numpy(pose_df.iloc[:, 5:9])
        self.times = pd.DataFrame.to_numpy(pose_df.iloc[:,0:1]) + pd.DataFrame.to_numpy(pose_df.iloc[:,1:2])*1e-9
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
        return self.orientations[idx]
    
    def T_WB(self, t):
        """
        Transform from world to body frame (or pose of body within world frame) at time t.

        Args:
            t (float): time

        Returns:
            np.array, shape(4,4): Rigid body transform
        """
        idx = self.idx(t)
        if idx is None: 
            return None
        curr_position = self.positions[idx]
        curr_orientation = self.orientations[idx]
        T_WB = np.eye(4)
        T_WB[:3,:3] = Rot.from_quat(curr_orientation).as_matrix()
        T_WB[:3,3] = curr_position
        return T_WB
    
    def idx(self, t):
        """
        Finds the index of pose info closes to the desired time.

        Args:
            t (float): time

        Returns:
            int: Index of pose info closest to desired time or None if no time is available within 
                tolerance.
        """
        op1_exists = np.where(self.times >= t)[0].shape[0]
        op2_exists = np.where(self.times <= t)[0].shape[0]
        if not op1_exists and not op2_exists:
            idx = None
        if op1_exists:
            op1 = np.where(self.times >= t)[0][0]
        if op2_exists:
            op2 = np.where(self.times <= t)[0][-1]
            
        if not op1_exists: 
            idx = op2
        elif not op2_exists: 
            idx = op1
        elif abs(t - self.times[op1]) < abs(t - self.times[op2]): 
            idx = op1
        else: 
            idx = op2
            
        # check to make sure found time is close enough
        if (self.times[idx] - t) < self.time_tol: 
            return None
        else: 
            return idx