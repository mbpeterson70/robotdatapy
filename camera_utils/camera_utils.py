import numpy as np

def pixel_depth_2_xyz(x, y, depth, K):
    """
    Converts a camera pixel and depth to 3D coordinates in optical RDF frame

    Args:
        x (float): pixel x coordinate
        y (float): pixel y coordinate
        depth (float): depth (distance along z-axis)
        K ((3,3) np.array): camera intrinsic matrix

    Returns:
        (3,) np.array: x, y, z 3D point coordinates in camera RDF coordinates
    """
    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    
    x_rdf = depth * (x - cx) / fx
    y_rdf = depth * (y - cy) / fy
    z_rdf = depth
    
    return np.array([x_rdf, y_rdf, z_rdf])

def xyz_2_pixel(xyz, K, axis=0):
    """
    Converts xyz point array to pixel coordinate array

    Args:
        xyz (np.array, shape=(3,n) or (n,3)): 3D coordinates in RDF camera coordinates
        K (np.array, shape=(3,3)): camera intrinsic calibration matrix
        axis (int, optional): 0 or 1, axis along which xyz coordinates are stacked. Defaults to 0.

    Returns:
        np.array, shape=(2,n) or (n,2): Pixel coordinates (x,y) in RDF camera coordinates
    """
    if axis == 0:
        xyz_shaped = np.array(xyz).reshape((-1,3)).T
    elif axis == 1:
        xyz_shaped = np.array(xyz).reshape((3,-1))
    else:
        assert False, "only axis 0 or 1 supported"
        
    pixel = K @ xyz_shaped / xyz_shaped[2,:]
    pixel = pixel[:2,:]
    if axis == 0:
        pixel = pixel.T
    return pixel
    