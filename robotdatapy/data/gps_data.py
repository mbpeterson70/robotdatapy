###########################################################
#
# gps_data.py
#
# Interface for GPS data. Currently supports data 
# from ROS bags.
#
# Authors: Lucas Jia, Mason Peterson
#
# October 28, 2025
#
###########################################################

import numpy as np
import pandas as pd
import os
from rosbags.highlevel import AnyReader
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass
import utm

from robotdatapy.data.robot_data import RobotData
from robotdatapy.ros_msg_convert import stamp_2_float



def _haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance on WGS-84 sphere (meters), ignoring altitude."""
    R = 6371000.0  # mean Earth radius in meters
    # radians
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def _distance_3d_m(p1, p2, use_altitude=True):
    """Distance between two GPS fixes (lat,lon,alt)."""
    d_horiz = _haversine_m(p1[0], p1[1], p2[0], p2[1])
    if use_altitude and len(p1) >= 3 and len(p2) >= 3 and not (np.isnan(p1[2]) or np.isnan(p2[2])):
        dz = float(p2[2]) - float(p1[2])
        return float(np.hypot(d_horiz, dz))
    return float(d_horiz)

class GPSData(RobotData):
    def __init__(self, times, lat_lon_alts, covariances=None, interp=True, causal=False, time_tol=1.0, t0=None): 
        """
        Class for easy access to object poses over time

        Args:
            times (np.array, shape(n,)): times of the poses
            lat_lon_alts (np.array, shape(n,3)): gps readings (lat lon alt)
            covariances (np.array, shape(n,3,3), optional): position covariance matrices. 
                Defaults to None.
            interp (bool): interpolate between closest times, else choose the closest time.
            causal (bool): if True, only finds data at or before requested time.
            time_tol (float, optional): Tolerance used when finding a pose at a specific time. If 
                no pose is available within tolerance, None is returned. Defaults to 1.0 (higher than
                other RobotData defaults since GPS data is typically lower rate).
            t0 (float, optional): Local time at the first msg. If not set, uses global time from 
                the data_file. Defaults to None.
        """
        super().__init__(time_tol=time_tol, interp=interp, causal=causal)
        self.set_times(np.array(times))
        self.lat_lon_alts = np.array(lat_lon_alts)
        self.covariances = np.array(covariances)
        if t0 is not None:
            self.set_t0(t0)

    @classmethod
    def from_bag(
        cls, 
        path: str, 
        topic: str, 
        interp: bool = True, 
        causal: bool = False, 
        time_tol: float = 1.0, 
        t0: float = None, 
    ):
        path = os.path.expanduser(os.path.expandvars(path))
        times = []
        lat_lon_alts = []
        covariances = []
        with AnyReader([Path(path)]) as reader:
            connections = [x for x in reader.connections if x.topic == topic]
            if len(connections) == 0:
                assert False, f"topic {topic} not found in bag file {path}"

            for (connection, timestamp, rawdata) in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                if type(msg).__name__ == 'sensor_msgs__msg__NavSatFix':
                    latitude = msg.latitude
                    longitude = msg.longitude
                    altitude = msg.altitude
                else:
                    assert False, "invalid msg type (not PoseStamped or Odometry)"
                times.append(stamp_2_float(msg.header.stamp))
                lat_lon_alts.append([latitude, longitude, altitude])
                covariances.append(np.array(msg.position_covariance).reshape((3,3)))

        return cls(times, lat_lon_alts, covariances=covariances, interp=interp, causal=causal, time_tol=time_tol, t0=t0)

    def latitude(self, t):
        lat, lon, alt = self.lat_lon_alt(t)
        return lat
    
    def longitude(self, t):
        lat, lon, alt = self.lat_lon_alt(t)
        return lon

    def altitude(self, t):
        lat, lon, alt = self.lat_lon_alt(t)
        return alt
    
    def utm(self, t):
        """
        Return the (easting, northing, zone_number, zone_letter) at time t.
        If self.interp is True, linearly interpolates each component between the two surrounding samples.
        Otherwise, returns the closest sample.
        """
        lat, lon, alt = self.lat_lon_alt(t)
        if np.any(np.isnan([lat, lon])):
            return (np.nan, np.nan, np.nan, np.nan)
        easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
        return (easting, northing, zone_number, zone_letter)

    def lat_lon_alt(self, t):
        """
        Return the (lat, lon, alt) at time t.
        If self.interp is True, linearly interpolates each component between the two surrounding samples.
        Otherwise, returns the closest sample.
        """
        idx = self.idx(t)  # RobotData gives either (i0, i1) or a single index
        if self.interp:
            i0, i1 = idx
            if i0 == i1:
                return self.lat_lon_alts[i0]
            t0, t1 = self.times[i0], self.times[i1]
            if t1 == t0:
                return self.lat_lon_alts[i0]
            alpha = (t - t0) / (t1 - t0)
            p0 = self.lat_lon_alts[i0]
            p1 = self.lat_lon_alts[i1]
            return p0 + alpha * (p1 - p0)
        else:
            if isinstance(idx, (tuple, list)) and len(idx) == 2:
                i0, i1 = idx
                i = i0 if abs(self.times[i0] - t) <= abs(self.times[i1] - t) else i1
                return self.lat_lon_alts[i]
            # Otherwise assume a single index
            return self.lat_lon_alts[idx]
        
    def covariance(self, t):
        """
        Return the covariance matrix at the time nearest t.
        """
        idx = self.idx(t, force_single=True)
        return self.covariances[idx]
    
    def rm_nans(self):
        """
        Removes any entries with NaNs in lat_lon_alts.
        """
        nan_idx = np.any(np.isnan(self.lat_lon_alts), axis=1)
        self.times = self.times[~nan_idx]
        self.lat_lon_alts = self.lat_lon_alts[~nan_idx]
        self.covariances = self.covariances[~nan_idx]
        return
    
    def path_length(self, t0=None, tf=None, use_altitude=True) -> float:
        """
        Compute total path length (meters) between t0 and tf.

        Args:
            t0 (float, optional): start time (seconds). If None, use first timestamp.
            tf (float, optional): end time (seconds). If None, use last timestamp.
            use_altitude (bool): if True, compute 3D distance using altitude; else use 2D.

        Returns:
            float: path length in meters.
        """
        if self.lat_lon_alts.shape[0] < 2:
            return 0.0

        # Default to full time range
        if t0 is None:
            t0 = self.times[0]
        if tf is None:
            tf = self.times[-1]

        # Ensure valid ordering
        if tf <= t0:
            return 0.0

        # Mask points in time window
        mask = (self.times >= t0) & (self.times <= tf)
        pts = self.lat_lon_alts[mask]
        if pts.shape[0] < 2:
            return 0.0

        # Compute cumulative distance
        total = 0.0
        for i in range(1, pts.shape[0]):
            total += _distance_3d_m(pts[i-1], pts[i], use_altitude=use_altitude)

        return float(total)
    
    def plot2d(self, ax=None, dt=0.25, t=None, t0=None, tf=None, axes='xy', utm=False, **kwargs):
        """
        Plots the position data in 2D

        Args:
            ax (matplotlib.axes._subplots.AxesSubplot): axis to plot on. Defaults to None.
            dt(float, optional): time step for plotting trajectory. Defaults to 0.25s.
            t(np.array, optional): specific times to plot. Defaults to None.
            t0 (float, optional): start time. Defaults to self.t0.
            tf (float, optional): end time. Defaults to self.tf.
            axes (str, optional): axes to plot. 'x' corresponds to 'latitude', 'y' corresponds to 
                'longitude', and 'z' corresponds to 'altitude'. Defaults to 'xy'.
            utm (bool, optional): if True, plots in UTM coordinates instead of lat/lon. Defaults to False.
        """
        assert t is None or (t0 is None and tf is None), "t and t0/tf cannot be given together"
        
        if ax is None:
            ax = plt.gca()

        assert axes == 'xy', "Only 'xy' axes supported for GPS data."
        # assert len(axes) == 2, "axes must be a string of length 2"
        # ax_idx = []
        # for i in range(2):
        #     if 'x' == axes[i]:
        #         ax_idx.append(0)
        #     elif 'y' == axes[i]:
        #         ax_idx.append(1)
        #     elif 'z' == axes[i]:
        #         ax_idx.append(2)
        #     else:
        #         assert False, "axes must be a string of x, y, or z"

        t = self._get_time_array(t=t, dt=dt, t0=t0, tf=tf)
        
        if utm:
            positions = np.array([self.utm(ti)[:2] for ti in t])
        else:
            positions = np.array([self.lat_lon_alt(ti) for ti in t])
            
        ax.plot(positions[:,0], positions[:,1], **kwargs)
        ax.set_aspect('equal')
        ax.grid(True)
        return ax