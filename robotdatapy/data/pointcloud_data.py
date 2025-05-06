# Copyright 2008 Willow Garage, Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the Willow Garage, Inc. nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Serialization of sensor_msgs__msg__PointCloud2 messages.

rosbag1/rosbag2 compatible port by Qingyuan Li

--------
Adapted from:
Serialization of sensor_msgs.PointCloud2 messages.

Author: Tim Field
ROS 2 port by Sebastian Grans
File originally ported from:
https://github.com/ros/common_msgs/blob/f48b00d43cdb82ed9367e0956db332484f676598/
sensor_msgs/src/sensor_msgs/point_cloud2.py
--------
"""

import os, sys
import numpy as np
from rosbags.highlevel import AnyReader
from pathlib import Path

from robotdatapy.data.robot_data import RobotData

_DATATYPES = {
    'INT8': np.dtype(np.int8),
    'UINT8': np.dtype(np.uint8),
    'INT16': np.dtype(np.int16),
    'UINT16': np.dtype(np.uint16),
    'INT32': np.dtype(np.int32),
    'UINT32': np.dtype(np.uint32),
    'FLOAT32': np.dtype(np.float32),
    'FLOAT64': np.dtype(np.float64)
}

DUMMY_FIELD_PREFIX = 'unnamed_field'

def dtype_from_fields(fields, point_step=None) -> np.dtype:
    """
    Convert a Iterable of sensor_msgs.msg.PointField messages to a np.dtype.

    Args:
        fields: The point cloud fields.
        point_step: Point step size in bytes. Calculated from the given fields by default.
    Returns:
        NumPy datatype
    """
    # Create a lists containing the names, offsets and datatypes of all fields
    field_names = []
    field_offsets = []
    field_datatypes = []

    for i, field in enumerate(fields):
        # Datatype as numpy datatype
        datatype = None
        for k,v in field.__dict__.items():
            if k in _DATATYPES.keys() and v == field.datatype:
                datatype = _DATATYPES[k]
        assert datatype is not None, 'field has no recognized datatype'

        # Name field
        name = f'{DUMMY_FIELD_PREFIX}_{i}' if field.name == '' else field.name

        # Handle fields with count > 1 by creating subfields with a suffix consiting
        # of "_" followed by the subfield counter [0 -> (count - 1)]
        assert field.count > 0, "Can't process fields with count = 0."
        for a in range(field.count):
            # Add suffix if we have multiple subfields
            subfield_name = f'{name}_{a}' if field.count > 1 else name

            assert subfield_name not in field_names, 'Duplicate field names are not allowed!'

            field_names.append(subfield_name)

            # Create new offset that includes subfields
            field_offsets.append(field.offset + a * datatype.itemsize)
            field_datatypes.append(datatype.str)

    # Create dtype
    dtype_dict = {
            'names': field_names,
            'formats': field_datatypes,
            'offsets': field_offsets
    }
    if point_step is not None:
        dtype_dict['itemsize'] = point_step
    return np.dtype(dtype_dict)


class PointCloud:
    """
    Class to store & access point clouds in numpy
    """
    def __init__(self, header, height, width, fields, points):
        """
        Class to store & access point clouds in numpy

        Args:
            header: ROS message header
            height: height of point cloud
            width: width of point cloud
            fields: fields of each point
            points: point cloud stored as a structured numpy array
        """
        self.header = header
        self.height = height
        self.width = width
        self.fields = fields
        self.points = points

    @classmethod
    def from_msg(cls, cloud):
        """
        Extract point cloud from ROS PointCloud2 message
        """
        assert len(cloud.fields) > 0, "empty pointcloud"

        points = np.ndarray(
            shape=(cloud.width * cloud.height,),
            dtype=dtype_from_fields(cloud.fields, point_step=cloud.point_step),
            buffer=cloud.data
        )

        if bool(sys.byteorder != 'little') != bool(cloud.is_bigendian):
            points = points.byteswap(inplace=True)

        # Cast into 2D array if cloud is organized (multiple rows)
        if cloud.height > 1:
            points = points.reshape(cloud.width, cloud.height)

        fields = dict(points.dtype.fields)

        return cls(header=cloud.header, height=cloud.height, width=cloud.width, fields=fields, points=points)
    
    def extract_fields(self, fields):
        """
        Get unstructured numpy array of the requested fields. With $m$ fields, the shape will be
        $(n, m)$ if the point cloud is unstructured but $(w, h, m)$ if it is.
        """
        return np.array(self.points[fields].tolist())
    
    def get_xyz(self):
        """
        Get 'x', 'y', 'z' fields as numpy array
        """
        return self.extract_fields(['x', 'y', 'z'])
    
    def get_xy(self):
        """
        Get 'x', 'y' fields as numpy array
        """
        return self.extract_fields(['x', 'y'])


class PointCloudData(RobotData):
    """
    Class for easy access to point cloud data over time
    """
    
    def __init__(
            self, 
            times, 
            pointclouds,
            data_type='bag',
            data_path=None,
            causal=False, 
            time_tol=.1, 
            t0=None, 
            time_range=None
        ): 
        """
        Class for easy access to point cloud data over time

        Args:
            times (np.array, shape=(n,)): times of point clouds
            pointclouds (list, shape=(n,)): list of point clouds
            data_type (str): type of data file (only 'bag' supported for now)
            data_path (str): path to data file
            time_tol (float, optional): Tolerance used when finding a point cloud at a specific time. If 
                no point cloud is available within tolerance, None is returned. Defaults to .1.
            t0 (float, optional): Local time at the first msg. If not set, uses global time from 
                the data_path. Defaults to None.
            time_range (list, shape=(2,), optional): Two element list indicating range of times
                (before being offset with t0) that should be stored within object
        """        
        super().__init__(time_tol=time_tol, interp=False, causal=causal)

        if time_range is not None:
            assert time_range[0] < time_range[1], "time_range must be given in incrementing order"
            start_idx = np.where(np.array(times) >= time_range[0])[0][0]
            end_idx = np.where(np.array(times) <= time_range[1])[0][-1]
            times = times[start_idx:end_idx+1]
            pointclouds = pointclouds[start_idx:end_idx+1]
        
        self.set_times(times)
        self.pointclouds = pointclouds
        data_path = os.path.expanduser(os.path.expandvars(data_path)) if data_path is not None else None
        self.data_path = data_path
        self.data_type = data_type
        if t0 is not None:
            self.set_t0(t0)

        self.fields = None if len(self.pointclouds) == 0 else PointCloud.from_msg(self.pointclouds[0]).fields

    @classmethod
    def from_bag(cls, path, topic, causal=False, time_tol=.1, t0=None, time_range=None):
        """
        Creates PointCloudData object from ROS1/ROS2 bag file

        Args:
            path (str): ROS bag file path
            topic (str): ROS PointCloud2 topic
            time_tol (float, optional): Tolerance used when finding a pose at a specific time. If 
                no pose is available within tolerance, None is returned. Defaults to .1.
            t0 (float, optional): Local time at the first msg. If not set, uses global time from 
                the data_path. Defaults to None.
            time_range (list, shape=(2,), optional): Two element list indicating range of times
                that should be stored within object
        """
        if time_range is not None:
            assert time_range[0] < time_range[1], "time_range must be given in incrementing order"
        
        times = []
        pcds = []
        with AnyReader([Path(path)]) as reader:
            connections = [x for x in reader.connections if x.topic == topic]
            if len(connections) == 0:
                assert False, f"topic {topic} not found in bag file {path}"
                
            for (connection, timestamp, rawdata) in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                if connection.topic != topic:
                    continue
                t = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
                if time_range is not None and t < time_range[0]:
                    continue
                elif time_range is not None and t > time_range[1]:
                    break
                times.append(t)

                pcds.append(msg)

        pcds = [msg for _, msg in sorted(zip(times, pcds), key=lambda zipped: zipped[0])]
        times = sorted(times)

        return cls(times=times, pointclouds=pcds, data_type='bag', data_path=path, 
                    causal=causal, time_tol=time_tol, t0=t0, time_range=time_range)
    
    def pointcloud(self, t: float):
        """
        Point cloud at time t.

        Args:
            t (float): time

        Returns:
            PointCloud
        """
        idx = self.idx(t)
        return PointCloud.from_msg(self.pointclouds[idx])     
    
    def clip(self, t0: float, tf: float):
        """
        Clips point cloud data from between t0 and tf

        Args:
            t0 (float): start clip time
            tf (float): end clip time
        """
        idx0 = self.idx(t0, force_single=True)
        idxf = self.idx(tf, force_single=True)
        self.set_times(self.times[idx0:idxf+1])
        self.pointclouds = self.pointclouds[idx0:idxf+1]
        return
    
    def msg_header(self, t: float):
        """
        Header of point cloud at time t.

        Args:
            t (float): time

        Returns:
            ROS message header
        """
        assert self.data_type == 'bag', "must be from a rosbag to have message header"
        idx = self.idx(t)
        return self.pointclouds[idx].header
    
    @property
    def field_names(self):
        """
        Get names of fields of each point
        """
        return np.array(list(self.fields.keys()))