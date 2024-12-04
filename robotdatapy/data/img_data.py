import numpy as np
import os
from rosbags.highlevel import AnyReader
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
import concurrent

from robotdatapy.data.robot_data import RobotData
import cv2
import pykitti
from robotdatapy.camera import CameraParams
from robotdatapy.exceptions import MsgNotFound

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
            times, 
            imgs,
            data_type,
            data_path=None,
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
            times (np.array, shape=(n,)): times of images
            imgs (list, shape=(n,)): list of images or image messages
            data_type (str): type of data file: 'bag', 'kitti', 'raw'
            data_path (str): path to data file
            time_tol (float, optional): Tolerance used when finding a pose at a specific time. If 
                no pose is available within tolerance, None is returned. Defaults to .1.
            t0 (float, optional): Local time at the first msg. If not set, uses global time from 
                the data_path. Defaults to None.
            time_range (list, shape=(2,), optional): Two element list indicating range of times
                (before being offset with t0) that should be stored within object
            compressed (bool, optional): True if data_path contains compressed images
        """        
        super().__init__(time_tol=time_tol, interp=False, causal=causal)

        if time_range is not None:
            assert time_range[0] < time_range[1], "time_range must be given in incrementing order"
            start_idx = np.where(np.array(times) >= time_range[0])[0][0]
            end_idx = np.where(np.array(times) <= time_range[1])[0][-1]
            times = times[start_idx:end_idx+1]
            imgs = imgs[start_idx:end_idx+1]
        
        self.set_times(times)
        self.imgs = imgs
        
        data_path = os.path.expanduser(os.path.expandvars(data_path)) if data_path is not None else None
        self.compressed = compressed
        self.compressed_encoding = compressed_encoding
        self.compressed_rvl = compressed_rvl
        self.data_path = data_path
        self.data_type = data_type
        self.interp = False
        self.bridge = cv_bridge.CvBridge()
        if t0 is not None:
            self.set_t0(t0)
            
        self.camera_params = CameraParams()
            
    @classmethod
    def from_bag(cls, path, topic, time_range=None, time_tol=.1, causal=False, 
                 t0=None, compressed=True, compressed_encoding='passthrough', compressed_rvl=False):
        """
        Creates ImgData object from bag file

        Args:
            path (str): ROS bag file path
            topic (str): ROS image topic
            time_range (list, shape=(2,), optional): Two element list indicating range of times
                that should be stored within object
            time_tol (float, optional): Tolerance used when finding a pose at a specific time. If 
                no pose is available within tolerance, None is returned. Defaults to .1.
            t0 (float, optional): Local time at the first msg. If not set, uses global time from 
                the data_path. Defaults to None.
            time_range (list, shape=(2,), optional): Two element list indicating range of times
                (before being offset with t0) that should be stored within object
            compressed (bool, optional): True if data_path contains compressed images
        """
        if time_range is not None:
            assert time_range[0] < time_range[1], "time_range must be given in incrementing order"
        
        times = []
        img_msgs = []
        with AnyReader([Path(path)]) as reader:
            connections = [x for x in reader.connections if x.topic == topic]
            if len(connections) == 0:
                raise MsgNotFound(topic, path)
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
        
        img_msgs = [msg for _, msg in sorted(zip(times, img_msgs), key=lambda zipped: zipped[0])]
        times = sorted(times)

        return cls(times=times, imgs=img_msgs, data_type='bag',  data_path=path, 
                   time_tol=time_tol, causal=causal, t0=t0, compressed=compressed, 
                   compressed_encoding=compressed_encoding, compressed_rvl=compressed_rvl)


    @classmethod
    def from_kitti(cls, path, kitti_type, kitti_sequence='00', time_range=None, time_tol=.1, causal=False, 
                 t0=None, compressed=True, compressed_encoding='passthrough', compressed_rvl=False):
        """
        Creates ImgData object from bag file

        Args:
            path (str): Path to directory that contains KITTI data.
            kitti_sequence (str): The KITTI sequence to use.
            kitti_type (str): 'rgb' or 'depth'
            time_range (list, shape=(2,), optional): Two element list indicating range of times
                that should be stored within object
            time_tol (float, optional): Tolerance used when finding a pose at a specific time. If 
                no pose is available within tolerance, None is returned. Defaults to .1.
            t0 (float, optional): Local time at the first msg. If not set, uses global time from 
                the data_path. Defaults to None.
            time_range (list, shape=(2,), optional): Two element list indicating range of times
                (before being offset with t0) that should be stored within object
            compressed (bool, optional): True if data_path contains compressed images
        """
        assert kitti_type == 'rgb' or kitti_type == 'depth'
        data_file = os.path.expanduser(os.path.expandvars(path))

        if kitti_type == 'rgb':
            dataset = pykitti.odometry(data_file, kitti_sequence)
            times = np.asarray([d.total_seconds() for d in dataset.timestamps])
        elif kitti_type == 'depth':
            dataset = pykitti.odometry(data_file, kitti_sequence)
            times = np.asarray([d.total_seconds() for d in dataset.timestamps])

        img_data = cls(times=times, imgs=dataset, data_type='kitti',  data_path=path,
                       time_tol=time_tol, causal=causal, t0=t0, compressed=compressed, 
                       compressed_encoding=compressed_encoding, compressed_rvl=compressed_rvl)
        img_data.kitti_type = kitti_type
        img_data.kitti_depth_img_path = data_file + '/sequences/' + kitti_sequence  + '/depth/'
        return img_data
    
    @classmethod
    def from_zip(cls, path, **kwargs):
        """
        Load image data from zip file
        """
        times = []
        imgs = []
        directory_path = path.replace('.zip', '')
        directory_path = os.path.expanduser(os.path.expandvars(directory_path))
        
        os.system(f"cd {os.path.dirname(path)} && unzip {path}")
        with open(os.path.join(directory_path, 'metadata.txt'), 'r') as f:
            for line in f:
                idx, secs, nsecs = line.strip().split()
                times.append(float(secs) + float(nsecs) * 1e-9)
                imgs.append(cv2.imread(os.path.join(directory_path, f"{idx}.png")))
        os.system(f"rm -r {directory_path}")
        return cls(times=times, imgs=imgs, data_type='raw', **kwargs)
        
    
    def to_zip(self, path):
        """
        Save image data to zip file
        """
        directory_path = path.replace('.zip', '')
        os.makedirs(os.path.expanduser(os.path.expandvars(directory_path)), exist_ok=False)
        with open(os.path.join(directory_path, 'metadata.txt'), 'w') as f:
            for i, t in enumerate(self.times):
                f.write(f"{i} {int(t)} {int((t % 1) * 1e9)}\n")
        for i, t in enumerate(self.times):
            cv2.imwrite(os.path.join(directory_path, f"{i}.png"), self.img(t))
        # TODO: speedup?
        # write_img = lambda i: cv2.imwrite(os.path.join(directory_path, f"{i}.png"), self.img(self.times[i]))
        # executor = concurrent.futures.ProcessPoolExecutor(10)
        # futures = [executor.submit(write_img, i) for i in range(len(self.times))]
        # concurrent.futures.wait(futures)

        prev_dir = os.getcwd()
        os.chdir(os.path.dirname(directory_path))
        os.system(f"zip -r {os.path.basename(path)} {os.path.basename(directory_path)}")
        os.system(f"rm -r {os.path.basename(directory_path)}")
        os.chdir(prev_dir)
        
        
    
    def extract_params(self, topic=None):
        """
        Get camera parameters

        Args:
            topic (str): ROS topic name containing cameraInfo msg

        Returns:
            np.array, shape=(3,3): camera intrinsic matrix K
        """
        if self.data_type == 'bag' or self.data_type == 'bag2':
            self.camera_params = CameraParams.from_bag(self.data_path, topic)
        elif self.data_type == 'kitti':
            P2 = self.imgs.calib.P_rect_20.reshape((3, 4))
            k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P2)
            self.camera_params = CameraParams(k, np.zeros(4), 1241, 376) # Hard coding for now.... TODO: improve this
        else:
            assert False, "data_type not supported, please choose from: bag, bag2, kitti"
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
        if self.data_type == 'bag' or self.data_type == 'bag2':
            if not self.compressed:
                img = self.bridge.imgmsg_to_cv2(self.imgs[idx], desired_encoding=self.compressed_encoding)
            elif self.compressed_rvl:
                from rvl import decompress_rvl
                assert self.width is not None and self.height is not None
                img = decompress_rvl(
                    np.array(self.imgs[idx].data[20:]).astype(np.int8), 
                    self.height*self.width).reshape((self.height, self.width))
            else:
                img = self.bridge.compressed_imgmsg_to_cv2(self.imgs[idx], desired_encoding=self.compressed_encoding)
                
        elif self.data_type == 'kitti' and self.kitti_type == 'rgb':
            pil_image = self.imgs.get_cam2(idx)
            img = np.array(pil_image)
            # Convert RGB to BGR
            img = img[:, :, ::-1].copy()

        elif self.data_type == 'kitti' and self.kitti_type == 'depth':
            # img = np.load(self.data_file + '/sequences/' + self.kitti_sequence  + '/depth/' + str(idx).zfill(6) + '.npy')
            img = np.load(self.kitti_depth_img_path + str(idx).zfill(6) + '.npy')
        
        elif self.data_type == 'raw':
            img = self.imgs[idx]
            
        else:
            raise ValueError("data_type not supported, please choose from: raw, bag, kitti")
            
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
