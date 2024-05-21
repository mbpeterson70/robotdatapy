import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import pandas as pd
import os
from rosbags.highlevel import AnyReader
from pathlib import Path
import matplotlib.pyplot as plt
import pykitti
from robot_utils.robot_data.robot_data import RobotData
import cv2
# TODO: maybe add a transform_pose function and a transform_by_pose function
    
class PoseData(RobotData):
    """
    Class for easy access to object poses over time
    """
    
    def __init__(self, data_file, file_type, 
                 kitti_sequence='00',
                 interp=False, causal=False, topic=None, 
                 time_tol=.1, t0=None, csv_options=None, 
                 T_recorded_body=None, T_premultiply=None, T_postmultiply=None): 
        """
        Class for easy access to object poses over time

        Args:
            data_file (str): File path to data
            file_type (str): 'csv' or 'bag' or 'kitti'
            interp (bool): interpolate between closest times, else choose the closest time.
            topic (str, optional): ROS topic, necessary only for bag file_type. Defaults to None.
            time_tol (float, optional): Tolerance used when finding a pose at a specific time. If 
                no pose is available within tolerance, None is returned. Defaults to .1.
            t0 (float, optional): Local time at the first msg. If not set, uses global time from 
                the data_file. Defaults to None.
            csv_option (dict, optional): See _extract_csv_data for details. Defaults to None.
            T_recorded_body (np.array, shape(4,4)): Rigid transform from body frame to the frame 
                the data was recorded in. 
        """
        super().__init__(time_tol=time_tol, interp=interp, causal=causal)
        data_file = os.path.expanduser(os.path.expandvars(data_file))
        if file_type == 'csv':
            self._extract_csv_data(data_file, csv_options)
        elif file_type == 'bag':
            self._extract_bag_data(data_file, topic)
        elif file_type == 'kitti':
            self.dataset = pykitti.odometry(data_file, kitti_sequence)
            self.poses = np.asarray(self.dataset.poses)
            self.positions = self.poses[:, 0:3, -1]
            self.orientations = np.asarray([Rot.as_quat(Rot.from_matrix(self.poses[i, :3, :3])) for i in range(self.poses.shape[0])])
            self.times = np.asarray([d.total_seconds() for d in self.dataset.timestamps])
            P2 = self.dataset.calib.P_rect_20.reshape((3, 4)) # Left RGB camera
            k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P2)
            T_recorded_body = np.vstack([np.hstack([r, t[:3]]), np.asarray([0, 0, 0, 1])])
        else:
            assert False, "file_type not supported, please choose from: csv or bag2"
        if t0 is not None:
            self.set_t0(t0)
        self.T_premultiply = T_premultiply
        self.T_postmultiply = T_postmultiply
        self.file_type = file_type

        if T_recorded_body is not None:
            assert self.T_postmultiply is None, "T_postmultiply not supported with T_recorded_body"
            self.T_postmultiply = T_recorded_body
        
    
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
            self.set_times((pd.DataFrame.to_numpy(pose_df.iloc[:,0:1]) + pd.DataFrame.to_numpy(pose_df.iloc[:,1:2])*1e-9).reshape(-1))
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
            self.set_times(pd.DataFrame.to_numpy(pose_df.iloc[:,t_cn]).astype(np.float64).reshape(-1))

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
            if len(connections) == 0:
                assert False, f"topic {topic} not found in bag file {bag_file}"
            for (connection, timestamp, rawdata) in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                times.append(msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9)
                if type(msg).__name__ == 'geometry_msgs__msg__PoseStamped':
                    pose = msg.pose
                elif type(msg).__name__ == 'nav_msgs__msg__Odometry':
                    pose = msg.pose.pose
                else:
                    assert False, "invalid msg type (not PoseStamped or Odometry)"
                positions.append([pose.position.x, pose.position.y, pose.position.z])
                orientations.append([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        self.set_times(np.array(times))
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
        if self.T_premultiply is not None or self.T_postmultiply is not None:
            return self.T_WB(t)[:3,3]
        else:
            return self._untransformed_position(t)
        
    def orientation(self, t):
        """
        Orientation at time t.

        Args:
            t (float): time

        Returns:
            np.array, shape(4,): orientation as a quaternion
        """
        if self.T_premultiply is not None or self.T_postmultiply is not None:
            return Rot.from_matrix(self.T_WB(t)[:3,:3]).as_quat()
        else:
            return self._untransformed_orientation(t)
                
    def _untransformed_position(self, t):
        """
        Position at time t.

        Args:
            t (float): time

        Returns:
            np.array, shape(3,): position in xyz
        """
        idx = self.idx(t)
        if self.interp:
            if idx[0] == idx[1]:
                position = self.positions[idx[0]]
            else:
                position = self.positions[idx[0]] + \
                    (self.positions[idx[1]] - self.positions[idx[0]]) * \
                    (t - self.times[idx[0]]) / (self.times[idx[1]] - self.times[idx[0]])
        else:
            position = self.positions[idx]
        return position
    
    def _untransformed_orientation(self, t):
        """
        Orientation at time t.

        Args:
            t (float): time

        Returns:
            np.array, shape(4,): orientation as a quaternion
        """
        idx = self.idx(t)        
        if self.interp:
            if idx[0] == idx[1]:
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
        position = self._untransformed_position(t)
        orientation = self._untransformed_orientation(t)
        if position is None or orientation is None:
            return None
        T_WB = np.eye(4)
        T_WB[:3,:3] = Rot.from_quat(orientation).as_matrix()
        T_WB[:3,3] = position
        if self.T_premultiply is not None:
            T_WB = self.T_premultiply @ T_WB
        if self.T_postmultiply is not None:
            T_WB = T_WB @ self.T_postmultiply
        return T_WB
    
    def clip(self, t0, tf):
        """
        Clips the data to be between t0 and tf

        Args:
            t0 (float): start time
            tf (float): end time
        """
        idx0 = self.idx(t0) if not self.interp else self.idx(t0)[1]
        idxf = self.idx(tf) if not self.interp else self.idx(tf)[0]
        self.set_times(self.times[idx0:idxf])
        self.positions = self.positions[idx0:idxf]
        self.orientations = self.orientations[idx0:idxf]

    # def plot2d(self, ax=None, dt=.1, t0=None, tf=None, axes='xy'):
    def plot2d(self, ax=None, dt=.1, t=None, t0=None, tf=None, axes='xy', pose=False, trajectory=True, axis_len=1.0):
        """
        Plots the position data in 2D

        Args:
            ax (matplotlib.axes._subplots.AxesSubplot): axis to plot on. Defaults to None.
            t0 (float, optional): start time. Defaults to self.t0.
            tf (float, optional): end time. Defaults to self.tf.
            axes (str, optional): axes to plot. Defaults to 'xy'.
        """
        assert t is None or (t0 is None and tf is None), "t and t0/tf cannot be given together"
        assert trajectory or pose, "Must request plotting trajectory and/or pose"
        
        if ax is None:
            ax = plt.gca()

        if t0 is None and t is None:
            t0 = self.t0
        if tf is None and t is None:
            tf = self.tf

        assert len(axes) == 2, "axes must be a string of length 2"
        ax_idx = []
        for i in range(2):
            if 'x' == axes[i]:
                ax_idx.append(0)
            elif 'y' == axes[i]:
                ax_idx.append(1)
            elif 'z' == axes[i]:
                ax_idx.append(2)
            else:
                assert False, "axes must be a string of x, y, or z"

        if t is None and trajectory:
            positions = np.array([self.position(t) for t in np.arange(t0, tf, dt)])
            ax.plot(positions[:,ax_idx[0]], positions[:,ax_idx[1]])
        if t is not None or pose:
            if t is not None:
                t = [t]
            else:
                t = np.arange(t0, tf, dt)
            for ti in t:
                for rob_ax, color in zip([0, 1, 2], ['red', 'green', 'blue']):
                    T_WB = self.T_WB(ti)
                    ax.plot([T_WB[ax_idx[0],3], T_WB[ax_idx[0],3] + axis_len*T_WB[ax_idx[0],rob_ax]], 
                            [T_WB[ax_idx[1],3], T_WB[ax_idx[1],3] + axis_len*T_WB[ax_idx[1],rob_ax]], color=color)
            
        ax.set_xlabel(axes[0])
        ax.set_ylabel(axes[1])
        ax.axis('equal')
        ax.grid(True)
        return ax