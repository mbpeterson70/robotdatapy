import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import pandas as pd
import os
from rosbags.highlevel import AnyReader
from pathlib import Path
import matplotlib.pyplot as plt
import pykitti
import cv2
import evo
import csv
import yaml

from robotdatapy.data.robot_data import RobotData

KIMERA_MULTI_GT_CSV_OPTIONS = {
    'cols': {
        'time': ["#timestamp_kf"],
        'position': ['x', 'y', 'z'],
        'orientation': ["qx", "qy", "qz", "qw"],
    },
    'col_nums': {
        'time': [0],
        'position': [1, 2, 3],
        'orientation': [5, 6, 7, 4]
    },
    'timescale': 1e-9
}

class PoseData(RobotData):
    """
    Class for easy access to object poses over time
    """
    
    def __init__(self, times, positions, orientations, interp=True, causal=False, time_tol=.1, t0=None, T_premultiply=None, T_postmultiply=None): 
        """
        Class for easy access to object poses over time

        Args:
            times (np.array, shape(n,)): times of the poses
            positions (np.array, shape(n,3)): xyz positions of the poses
            orientations (np.array, shape(n,4)): quaternions of the poses
            interp (bool): interpolate between closest times, else choose the closest time.
            time_tol (float, optional): Tolerance used when finding a pose at a specific time. If 
                no pose is available within tolerance, None is returned. Defaults to .1.
            t0 (float, optional): Local time at the first msg. If not set, uses global time from 
                the data_file. Defaults to None.
            T_premultiply (np.array, shape(4,4)): Rigid transform to premultiply to the pose.
            T_postmultiply (np.array, shape(4,4)): Rigid transform to postmultiply to the pose.
        """
        super().__init__(time_tol=time_tol, interp=interp, causal=causal)
        self.set_times(np.array(times))
        self.positions = np.array(positions)
        self.orientations = np.array(orientations)
        if t0 is not None:
            self.set_t0(t0)
        self.T_premultiply = T_premultiply
        self.T_postmultiply = T_postmultiply
    
    @classmethod
    def from_csv(cls, path, csv_options, interp=True, causal=False, time_tol=.1, t0=None, T_premultiply=None, T_postmultiply=None):
        """
        Extracts pose data from csv file with 9 columns for time (sec/nanosec), position, and orientation

        Args:
            path (str): CSV file path
            csv_options (dict): Can include dict of structure: dict['col'], dict['col_nums'] which 
                map to dicts containing keys of 'time', 'position', and 'orientation' and the 
                corresponding column names and numbers. csv_options['timescale'] can be given if 
                the time column is not in seconds.
            interp (bool): interpolate between closest times, else choose the closest time.
            time_tol (float, optional): Tolerance used when finding a pose at a specific time. If 
                no pose is available within tolerance, None is returned. Defaults to .1.
            t0 (float, optional): Local time at the first msg. If not set, uses global time from 
                the data_file. Defaults to None.
            T_premultiply (np.array, shape(4,4)): Rigid transform to premultiply to the pose.
            T_postmultiply (np.array, shape(4,4)): Rigid transform to postmultiply to the pose.
        """
        path = os.path.expanduser(os.path.expandvars(path))
        if csv_options is None:
            pose_df = pd.read_csv(path, usecols=['header.stamp.secs', 'header.stamp.nsecs', 'pose.position.x', 'pose.position.y', 'pose.position.z',
                'pose.orientation.x', 'pose.orientation.y', 'pose.orientation.z', 'pose.orientation.w'])
            positions = pd.DataFrame.to_numpy(pose_df.iloc[:, 2:5])
            orientations = pd.DataFrame.to_numpy(pose_df.iloc[:, 5:9])
            times = (pd.DataFrame.to_numpy(pose_df.iloc[:,0:1]) + pd.DataFrame.to_numpy(pose_df.iloc[:,1:2])*1e-9).reshape(-1)
        else:
            cols = csv_options['cols']
            pose_df = pd.read_csv(path, usecols=cols['time'] + cols['position'] + cols['orientation'])
            
            if 'col_nums' in csv_options:
                t_cn = csv_options['col_nums']['time']
                pos_cn = csv_options['col_nums']['position']
                ori_cn = csv_options['col_nums']['orientation']
            else:
                t_cn, pos_cn, ori_cn = [0], [1,2,3], [4,5,6,7]
            positions = pd.DataFrame.to_numpy(pose_df.iloc[:, pos_cn])
            orientations = pd.DataFrame.to_numpy(pose_df.iloc[:, ori_cn])
            times = pd.DataFrame.to_numpy(pose_df.iloc[:,t_cn]).astype(np.float64).reshape(-1)

        if 'timescale' in csv_options:
            times *= csv_options['timescale']

        return cls(times, positions, orientations, interp=interp, causal=causal, time_tol=time_tol,
                   t0=t0, T_premultiply=T_premultiply, T_postmultiply=T_postmultiply)
    
    @classmethod
    def from_kmd_gt_csv(cls, path, **kwargs):
        """
        Extracts pose data from a Kimera-Multi Data ground truth csv file. 
        The csv file should have columns 
        '#timestamp_kf', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'.

        Args:
            path (str): CSV file path
            kwargs: Additional arguments to pass to from_csv
        """
        csv_options = KIMERA_MULTI_GT_CSV_OPTIONS
        return cls.from_csv(path, csv_options, **kwargs)
    
    @classmethod
    def from_bag(
        cls, 
        path: str, 
        topic: str, 
        interp: bool = True, 
        causal: bool = False, 
        time_tol: float = .1, 
        t0: float = None, 
        T_premultiply: np.array = None, 
        T_postmultiply: np.array = None
    ):
        """
        Create a PoseData object from a ROS bag file. Supports msg types PoseStamped and Odometry.

        Args:
            path (str): ROS bag file path
            topic (str): ROS pose topic
            interp (bool): interpolate between closest times, else choose the closest time.
            causal (bool): if True, only use data that is available at the time requested.
            time_tol (float, optional): Tolerance used when finding a pose at a specific time. If 
                no pose is available within tolerance, None is returned. Defaults to .1.
            t0 (float, optional): Local time at the first msg. If not set, uses global time from 
                the data_file. Defaults to None.
            T_premultiply (np.array, shape(4,4)): Rigid transform to premultiply to the pose.
            T_postmultiply (np.array, shape(4,4)): Rigid transform to postmultiply to the pose.

        Returns:
            PoseData: PoseData object
        """
        path = os.path.expanduser(os.path.expandvars(path))
        times = []
        positions = []
        orientations = []
        with AnyReader([Path(path)]) as reader:
            connections = [x for x in reader.connections if x.topic == topic]
            if len(connections) == 0:
                assert False, f"topic {topic} not found in bag file {path}"
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

        return cls(times, positions, orientations, interp=interp, causal=causal, time_tol=time_tol, 
                   t0=t0, T_premultiply=T_premultiply, T_postmultiply=T_postmultiply)
        
    @classmethod
    def from_bag_tf(cls, path: str, parent_frame: str, child_frame: str, **kwargs):
        """
        Create a PoseData object from a ROS bag file using tf (/tf, /tf_static topic) messages.

        Args:
            path (str): ROS bag file path
            parent_frame (str): parent frame
            child_frame (str): child frame
            kwargs: Additional arguments to pass to from_bag
            
        Returns:
            PoseData: PoseData object
        """
        tf_tree = cls._tf_tree_from_bag(path)
        frame_chain = [child_frame]
        tf_types = []
        T_parent_child = []
        
        
        while frame_chain[-1] != parent_frame:
            if frame_chain[-1] not in tf_tree:
                assert False, f"parent_frame {parent_frame} not found in bag file {path}"
            new_parent = frame_chain[-1]
            frame_chain.append(tf_tree[new_parent][1])
            tf_types.append(tf_tree[new_parent][0])
            
        frame_chain = frame_chain[::-1]
        tf_types = tf_types[::-1]
            
        for joint_i in range(len(tf_types)):
            if tf_types[joint_i] == 'tf_static':
                T_parent_child.append(cls.static_tf_from_bag(path, 
                    frame_chain[joint_i], frame_chain[joint_i+1]))
            elif tf_types[joint_i] == 'tf':
                T_parent_child.append(cls._from_bag_single_tf_joint(path, 
                    frame_chain[joint_i], frame_chain[joint_i+1], **kwargs))
            else:
                assert False, "invalid tf type"
        
        times = []
        for joint_i in range(len(tf_types)-1, -1, -1):
            if tf_types[joint_i] == 'tf':
                times = T_parent_child[joint_i].times
                break
        assert len(times) > 0, "no times found"
        
        poses = []
        for t in times:
            T = np.eye(4)
            for joint_i in range(len(tf_types)):
                if tf_types[joint_i] == 'tf_static':
                    T = T @ T_parent_child[joint_i]
                elif tf_types[joint_i] == 'tf':
                    T = T @ T_parent_child[joint_i].pose(t)
                else:
                    assert False, "invalid tf type"
            poses.append(T)
            
        return cls.from_times_and_poses(times, poses, **kwargs)
        
    @classmethod
    def from_times_and_poses(cls, times, poses, **kwargs):
        """
        Create a PoseData object from times and poses.

        Args:
            times (np.array, shape(n,)): times of the poses
            poses (np.array, shape(n,4,4)): poses as rigid body transforms
            
        Returns:
            PoseData: PoseData object
        """
        positions = np.array([pose[:3,3] for pose in poses])
        orientations = np.array([Rot.as_quat(Rot.from_matrix(pose[:3,:3])) for pose in poses])
        return cls(times, positions, orientations, **kwargs)

    @classmethod
    def from_kitti(cls, path, kitti_sequence='00', interp=False, causal=False, time_tol=.1, t0=None, T_premultiply=None, T_postmultiply=None):
        """
        Create a PoseData object from a ROS bag file. Supports msg types PoseStamped and Odometry.

        Args:
            path (str): Path to directory that contains KITTI data.
            kitti_sequence (str): The KITTI sequence to use.
            interp (bool): interpolate between closest times, else choose the closest time.
            time_tol (float, optional): Tolerance used when finding a pose at a specific time. If 
                no pose is available within tolerance, None is returned. Defaults to .1.
            t0 (float, optional): Local time at the first msg. If not set, uses global time from 
                the data_file. Defaults to None.
            T_premultiply (np.array, shape(4,4)): Rigid transform to premultiply to the pose.
            T_postmultiply (np.array, shape(4,4)): Rigid transform to postmultiply to the pose.

        Returns:
            PoseData: PoseData object
        """
        data_file = os.path.expanduser(os.path.expandvars(path))
        dataset = pykitti.odometry(data_file, kitti_sequence)
        poses = np.asarray(dataset.poses)
        positions = poses[:, 0:3, -1]
        orientations = np.asarray([Rot.as_quat(Rot.from_matrix(poses[i, :3, :3])) for i in range(poses.shape[0])])
        times = np.asarray([d.total_seconds() for d in dataset.timestamps])
        P2 = dataset.calib.P_rect_20.reshape((3, 4)) # Left RGB camera
        k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P2)
        T_recorded_body = np.vstack([np.hstack([r, t[:3]]), np.asarray([0, 0, 0, 1])])
        T_postmultiply = T_recorded_body

        return cls(times, positions, orientations, interp=interp, causal=causal, time_tol=time_tol, 
                   t0=t0, T_premultiply=T_premultiply, T_postmultiply=T_postmultiply)

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(os.path.expanduser(yaml_path), 'r') as f:
            args = yaml.safe_load(f)
            if args['type'] == 'bag':
                return cls.from_bag(**{k: v for k, v in args.items() if k != 'type'})
            elif args['type'] == 'csv':
                return cls.from_csv(**{k: v for k, v in args.items() if k != 'type'})
            elif args['type'] == 'bag_tf':
                return cls.from_bag_tf(**{k: v for k, v in args.items() if k != 'type'})
            else:
                raise ValueError("Invalid pose data type")

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
            np.array, shape(4,): orientation as a quaternion (x, y, z, w)
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
    
    def pose(self, t):
        """
        Pose at time t.

        Args:
            t (float): time

        Returns:
            np.array, shape(4,4): Rigid body transform
        """
        return self.T_WB(t)     
    
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

    def plot2d(self, ax=None, dt=.1, t=None, t0=None, tf=None, axes='xy', pose=False, trajectory=True, axis_len=1.0, **kwargs):
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

        if trajectory:
            if t is None:
                positions = np.array([self.position(ti) for ti in np.arange(t0, tf, dt)])
            else:
                positions = np.array([self.position(ti) for ti in t])
            ax.plot(positions[:,ax_idx[0]], positions[:,ax_idx[1]], **kwargs)
        if pose:
            if t is not None:
                t = t #[t]
            else:
                t = np.arange(t0, tf, dt)
            for ti in t:
                for rob_ax, color in zip([0, 1, 2], ['red', 'green', 'blue']):
                    T_WB = self.T_WB(ti)
                    ax.plot([T_WB[ax_idx[0],3], T_WB[ax_idx[0],3] + axis_len*T_WB[ax_idx[0],rob_ax]], 
                            [T_WB[ax_idx[1],3], T_WB[ax_idx[1],3] + axis_len*T_WB[ax_idx[1],rob_ax]], color=color)
            
        ax.set_xlabel(axes[0])
        ax.set_ylabel(axes[1])
        ax.set_aspect('equal')
        ax.grid(True)
        return ax
    
    def to_evo(self):
        """
        Converts the PoseData object to an evo PoseTrajectory3D object.

        Returns:
            evo.core.trajectory.PoseTrajectory3D: evo PoseTrajectory3D object
        """
        if self.T_premultiply is not None or self.T_postmultiply is not None:
            assert False, "Cannot convert transformed poses to evo PoseTrajectory3D"
        quat_wxyz = self.orientations[:, [3,0,1,2]]
        return evo.core.trajectory.PoseTrajectory3D(self.positions, quat_wxyz, self.times)
    
    def to_csv(self, path, csv_options=KIMERA_MULTI_GT_CSV_OPTIONS):
        """
        Converts the PoseData object to a csv file.
        
        Args:
            path (str): Path to save the csv file
            csv_options (dict): Can include dict of structure: dict['col'], dict['col_nums'] which 
                map to dicts containing keys of 'time', 'position', and 'orientation' and the 
                corresponding column names and numbers. csv_options['timescale'] can be given if 
                the time column is not in seconds.
        """
        assert csv_options == KIMERA_MULTI_GT_CSV_OPTIONS, \
            "Only KIMERA_MULTI_GT_CSV_OPTIONS is supported"
        data = [csv_options['cols']['time'] + csv_options['cols']['position'] \
            + csv_options['cols']['orientation']]
        for t in self.times:
            data.append([int(t/csv_options['timescale'])] + self.position(t).tolist() + 
                        [self.orientation(t)[3]] + self.orientation(t)[:3].tolist())
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)      
    
    @classmethod
    def static_tf_from_bag(cls, path: str, parent_frame: str, child_frame: str):
        """
        Extracts a static transform from a ROS bag file. Transform is returned as T^parent_child,
        where T is a 4x4 rigid body transform and expresses the pose of the child in the parent frame, 
        which is equivalent to the transformation from the child frame to the parent frame.

        Args:
            parent_frame (str): parent frame
            child_frame (str): child frame

        Returns:
            np.array, shape(4,4): static transform
        """
        tf_tree = cls.static_tf_dict_from_bag(path)
                        
        if child_frame not in tf_tree:
            assert False, f"child_frame {child_frame} not found in bag file {path}"

        # compute transform by traversing up the tree from the child frame to the parent frame
        # T_parent_child = T_parent_child1 * T_child1_child2 * ... * T_childN_child
        T_chain = []
        child = child_frame
        while child != parent_frame:
            if child not in tf_tree:
                assert False, f"parent_frame {parent_frame} not found in bag file {path}"
            parent, transform = tf_tree[child]
            Ti = np.eye(4)
            Ti[:3,:3] = Rot.from_quat([transform.rotation.x, transform.rotation.y, 
                                      transform.rotation.z, transform.rotation.w]).as_matrix()
            Ti[:3,3] = [transform.translation.x, transform.translation.y, transform.translation.z]
            # transform msg is T_child_parent, the pose of the parent in the child frame or the 
            # transform from parent to child we want in the form T_parent_child so invert
            # actually it seems like this is not true (ROS documentation is a bit confusing)
            # Ti = np.linalg.inv(Ti)
            T_chain.insert(0, Ti)
            child = parent
            
        T = np.eye(4)
        for Ti in T_chain:
            T = T @ Ti
        
        return T
    
    @classmethod
    def static_tf_dict_from_bag(cls, path: str):
        """Returns a dictionary of static transforms from a ROS bag file. The dictionary maps 
        child_frame_id to a tuple of (parent_frame_id, transform_msg).

        Args:
            path (str): Path to ROS bag.

        Returns:
            dict: Static transform dictionary
        """
        tf_tree = {}
        with AnyReader([Path(os.path.expanduser(os.path.expandvars(path)))]) as reader:
            connections = [x for x in reader.connections if x.topic == '/tf_static']
            if len(connections) == 0:
                assert False, f"topic /tf_static not found in bag file {path}"
            for (connection, timestamp, rawdata) in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                if type(msg).__name__ == 'tf2_msgs__msg__TFMessage':
                    for transform_msg in msg.transforms:
                        tf_tree[transform_msg.child_frame_id] = (transform_msg.header.frame_id, transform_msg.transform)
        return tf_tree
    
    @classmethod
    def _tf_tree_from_bag(cls, path: str):
        """
        Returns a dict where each key is a child frame id and the value is a tuple including the parent frame id and 
        whether the tf is static or dynamic.

        Args:
            path (str): Path to ROS bag.
            
        Returns:
            dict: tf tree dictionary
        """
        tf_tree = {}
        with AnyReader([Path(os.path.expanduser(os.path.expandvars(path)))]) as reader:
            for tf_type in ['tf', 'tf_static']:
                connections = [x for x in reader.connections if x.topic == f"/{tf_type}"]
                for (connection, timestamp, rawdata) in reader.messages(connections=connections):
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    if type(msg).__name__ == 'tf2_msgs__msg__TFMessage':
                        for transform_msg in msg.transforms:
                            # if has appeared multiple times, check that the parent is the same
                            if transform_msg.child_frame_id in tf_tree:
                                assert tf_tree[transform_msg.child_frame_id][1] == transform_msg.header.frame_id, \
                                    f"child frame {transform_msg.child_frame_id} has multiple parents"
                                assert tf_tree[transform_msg.child_frame_id][0] == tf_type, \
                                    f"child frame {transform_msg.child_frame_id} has multiple tf types"
                            else:
                                tf_tree[transform_msg.child_frame_id] = (tf_type, transform_msg.header.frame_id)
        return tf_tree
    
    @classmethod
    def _from_bag_single_tf_joint(cls, path: str, parent_frame: str, child_frame: str, **kwargs):
        """
        Extracts a dynamic transform from a ROS bag file in the form of a PoseData object. 
        Transform is returned as T^parent_child, where T is a 4x4 rigid body transform and expresses 
        the pose of the child in the parent frame, which is equivalent to the transformation 
        from the child frame to the parent frame.

        Args:
            parent_frame (str): parent frame
            child_frame (str): child frame

        Returns:
            PoseData: PoseData object
        """
        times = []
        positions = []
        orientations = []
        
        with AnyReader([Path(os.path.expanduser(os.path.expandvars(path)))]) as reader:
            connections = [x for x in reader.connections if x.topic == '/tf']
            if len(connections) == 0:
                assert False, f"topic /tf not found in bag file {path}"
            for (connection, timestamp, rawdata) in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                if type(msg).__name__ == 'tf2_msgs__msg__TFMessage':
                    for transform_msg in msg.transforms:
                        if transform_msg.child_frame_id == child_frame and transform_msg.header.frame_id == parent_frame:
                            times.append(transform_msg.header.stamp.sec + transform_msg.header.stamp.nanosec*1e-9)
                            positions.append([transform_msg.transform.translation.x, 
                                              transform_msg.transform.translation.y, 
                                              transform_msg.transform.translation.z])
                            orientations.append([transform_msg.transform.rotation.x, 
                                                 transform_msg.transform.rotation.y, 
                                                 transform_msg.transform.rotation.z, 
                                                 transform_msg.transform.rotation.w])
        return cls(times, positions, orientations, **kwargs)