import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import pandas as pd
from rosbags.highlevel import AnyReader
from pathlib import Path
from robot_utils.robot_data.robot_data import RobotData
    
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