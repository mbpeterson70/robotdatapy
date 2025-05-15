import numpy as np
from scipy.spatial.transform import Rotation as Rot

def pose_msg_2_quat(pose_msg):
    return np.array([pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w])

def pose_msg_2_position(pose_msg):
    return np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])

def pose_msg_2_transform(pose_msg):
    quat = pose_msg_2_quat(pose_msg)
    position = pose_msg_2_position(pose_msg)
    T = np.eye(4)
    T[:3,:3] = Rot.from_quat(quat).as_matrix()
    T[:3,3] = position
    return T

def stamp_2_float(stamp):
    return stamp.sec + stamp.nanosec*1e-9

def stamp_2_nanosec(stamp):
    return int(stamp.sec*1e9 + stamp.nanosec)