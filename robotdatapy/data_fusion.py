import numpy as np
import gtsam

from robotdatapy.data.robot_data import RobotData
from robotdatapy.data.pose_data import PoseData
from robotdatapy.data.gps_data import GPSData
from robotdatapy.transform import transform_to_gtsam, aruns


def fuse_gps_and_local_pose_estimates(
    gps_data: GPSData, 
    local_pose_estimate: PoseData, 
    gps_position: np.ndarray = np.zeros(3),
    estimate_rot_sig_deg: float = 0.5, 
    estimate_tran_sig_m: float = 0.1,
    gps_position_rot_sig_deg: float = 1.0,
    gps_position_tran_sig_m: float = 0.01,
    max_gps_sigma: float = np.inf,
    rot_init_time_window: float = 100.0,
    rot_init_num_samples: int = 10
):
    """
    Fuses GPS data with local pose estimates to create PoseData that is globally consistent
    with GPS data but with high rate pose estimation and rotation estimates from the augmenting
    local pose estimate data.

    Rotation initialization must be performed carefully. For this, positions from GPS and local
    pose estimates are sampled over a time window containing each pose time, and GPS positions and
    local positions are aligned using Arun's method, which enables expressing the local rotation
    estimation in the global frame.

    Args:
        gps_data (GPSData): GPS data
        local_pose_estimate (PoseData): Local pose estimates
        gps_position: (np.ndarray, shape(3,)): Position of the GPS antenna in the 
            base link frame.
        estimate_rot_sig_deg (float): Standard deviation of rotation noise (degrees) for
            relative pose factors from local pose estimates.
        estimate_tran_sig_m (float): Standard deviation of translation noise (meters) for
            relative pose factors from local pose estimates.
        gps_position_rot_sig_deg (float): Standard deviation of rotation noise (degrees)
            for the GPS antenna position calibration factors.
        gps_position_tran_sig_m (float): Standard deviation of translation noise (meters)
            for the GPS antenna position calibration factors.
        max_gps_sigma (float): Maximum allowed standard deviation (meters) for GPS measurements.
            Measurements with higher standard deviations will be ignored.
        rot_init_time_window (float): Time window (seconds) across which points are sampled for
            rotation intialization.
        rot_init_num_samples (int): Number of points to sample within the time window for rotation
            initialization.

    Returns:
        PoseData: Global PoseData object
    """
    # Factor graph form:
    #   A0---A0i---A1---A2---A3---A3i---A4
    #         |                    | 
    #        B0i                  B3i
    #         |                    |
    #       GPS0i                GPS3i
    #
    # A# are the local pose estimate variables
    # A#i are the interpolated local pose estimate variables at GPS times
    # B#i are the poses of the GPS
    # Factors between A# and A# are from local pose estimates or interpolations
    # Factors between A# and B# are the pose of the GPS antenna in the base link frame
    # GPS#i are the GPS measurements (3D prior factors) connected to the B#i GPS pose variables
    estimate_rot_sig_rad = np.deg2rad(estimate_rot_sig_deg)
    estimate_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([estimate_rot_sig_rad, estimate_rot_sig_rad, estimate_rot_sig_rad, 
                    estimate_tran_sig_m, estimate_tran_sig_m, estimate_tran_sig_m]))
    graph = gtsam.NonlinearFactorGraph()

    T_d_iminus1 = local_pose_estimate.pose(local_pose_estimate.t0)
    all_times = set(local_pose_estimate.times).union(set([
        t for t in gps_data.times if t > local_pose_estimate.t0 and t < local_pose_estimate.tf
    ]))
    time_idx_data = RobotData(time_tol=np.inf, interp=False, causal=False)
    time_idx_data.set_times(np.array(sorted(all_times)))

    # For all factors, add between factors from local pose estimates
    # (will automatically interpolate for times that are only in gps data)
    for i, t_i in enumerate(time_idx_data.times):
        if i == 0:
            continue
        T_d_i = local_pose_estimate.pose(time_idx_data.times[i])
        T_iminus1_i = np.linalg.inv(T_d_iminus1) @ T_d_i
        T_iminus1_i_gtsam = transform_to_gtsam(T_iminus1_i)
        graph.add(gtsam.BetweenFactorPose3(i-1, i, T_iminus1_i_gtsam, estimate_noise))
        T_d_iminus1 = T_d_i

    # Add between factors for GPS pose variables to local pose estimate variables
    gps_var_idx0 = len(time_idx_data.times)
    T_baselink_gps = np.eye(4)
    T_baselink_gps[:3,3] = gps_position
    gps_position_rot_sig_rad = np.deg2rad(gps_position_rot_sig_deg)
    calibration_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
        gps_position_rot_sig_rad, gps_position_rot_sig_rad, gps_position_rot_sig_rad, 
        gps_position_tran_sig_m, gps_position_tran_sig_m, gps_position_tran_sig_m
    ]))
    for i, t_i in enumerate(gps_data.times):
        pose_idx_gtsam = time_idx_data.idx(t_i, force_single=True)
        gps_idx_gtsam = gps_var_idx0 + i
        graph.add(gtsam.BetweenFactorPose3(
            pose_idx_gtsam, 
            gps_idx_gtsam, 
            transform_to_gtsam(T_baselink_gps), 
            calibration_noise
        ))
        

    # Add prior 3D factors for variables with GPS measurements
    for j, t_j in enumerate(gps_data.times):
        easting, northing, _, _ = gps_data.utm(t_j)
        altitude = gps_data.altitude(t_j)
        if np.any(np.isnan([easting, northing, altitude])):
            continue
        idx_gtsam = gps_data.idx(t_j, force_single=True) + gps_var_idx0
        covariance = gps_data.covariance(t_j)

        max_sigma = np.sqrt(np.max(np.diag(covariance)))
        if max_sigma > max_gps_sigma:
            continue
        gps_noise = gtsam.noiseModel.Gaussian.Covariance(covariance)
        graph.add(gtsam.PoseTranslationPrior3D(idx_gtsam,
            np.array([easting, northing, altitude]), gps_noise))


    # Compute initial guess for variable rotations by attempting to align GPS positions
    # with local pose estimate positions over a time window
    orig_gps_time_tol = gps_data.time_tol
    gps_data.time_tol = np.inf  # guarantees a value is returned for initial guess
    init_guess_poses = []
    for t in time_idx_data.times:
        sample_t0 = np.max([t - rot_init_time_window/2, local_pose_estimate.t0, gps_data.t0])
        sample_tf = np.min([t + sample_t0, local_pose_estimate.tf, gps_data.tf])
        sample_times = np.linspace(sample_t0, sample_tf, rot_init_num_samples)
        gps_points = [np.concatenate([gps_data.utm(st)[:2], [gps_data.altitude(st)]]) 
                        for st in sample_times]
        local_points = [local_pose_estimate.position(st)[:3] for st in sample_times]
        R_gps_odom = aruns(np.array(gps_points), np.array(local_points))[:3, :3]
        R_odom_baselink = local_pose_estimate.pose(t)[:3,:3]
        new_pose_init_guess = np.eye(4)
        new_pose_init_guess[:3,:3] = R_gps_odom[:3,:3] @ R_odom_baselink
        new_pose_init_guess[:2, 3] = gps_data.utm(t)[:2]
        new_pose_init_guess[2, 3] = gps_data.altitude(t)
        init_guess_poses.append(new_pose_init_guess)
    gps_data.time_tol = orig_gps_time_tol # restore time tol
    init_guess_pose_data = PoseData.from_times_and_poses(time_idx_data.times, init_guess_poses,
                                                    time_tol=np.inf, interp=False)
    
    # Create initial estimate for all variables
    initial_estimate = gtsam.Values()

    # Start with initial guesses for GPS pose variables
    for i, t_i in enumerate(gps_data.times):
        gps_idx_gtsam = gps_var_idx0 + i
        initial_estimate.insert(gps_idx_gtsam, transform_to_gtsam(init_guess_pose_data.pose(t_i)))

    # Next add initial guesses for local pose estimate variables
    # Post multiply the intial guess by T_gps_baselink
    init_guess_pose_data.T_postmultiply = np.linalg.inv(T_baselink_gps)
    for i, t_i in enumerate(time_idx_data.times):
        initial_estimate.insert(i, transform_to_gtsam(init_guess_pose_data.pose(t_i)))

    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    result = optimizer.optimize()
    result_pd = PoseData.from_times_and_poses(
        time_idx_data.times, 
        [result.atPose3(i).matrix() for i in range(len(time_idx_data.times))], 
        time_tol=local_pose_estimate.time_tol,
        interp=local_pose_estimate.interp,
        causal=local_pose_estimate.causal,
    )
    return result_pd