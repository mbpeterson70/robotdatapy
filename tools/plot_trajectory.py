import matplotlib.pyplot as plt
import argparse
from dataclasses import dataclass

from robotdatapy.data import PoseData

@dataclass
class PlotTrajectoryParams:

    time_tol: float = 10.0
    trajectory_dt: float = 1.0
    pose_dt: float = 10.0

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Quick trajectory plotter tool.")
    parser.add_argument("-c", "--csv", type=str, 
                        help="Path to a CSV file containing the trajectory data.")
    parser.add_argument("-b", "--bag", type=str, nargs=2, metavar=('BAG_FILE', 'TOPIC_NAME'),
                        help="Path to a ROS bag file and topic name containing the trajectory data.")
    parser.add_argument("-t", "--bag-tf", type=str, nargs=3, 
                        metavar=('BAG_FILE', 'PARENT_FRAME', 'CHILD_FRAME'),
                        help="Path to a ROS bag file and frame names for TF transformation.")
    parser.add_argument("-k", "--kitti", type=str, nargs=2, metavar=('KITTI_DATASET_PATH', 'SEQUENCE_ID'),
                        help="Path to a KITTI dataset and sequence ID to extract trajectory data.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Path to save the trajectory plot image (optional).")
    
    parser.add_argument("--axes", type=str, default="xy", choices=["xy", "xz", "yz"],
                        help="Axes to plot (default: xy).")
    
    args = parser.parse_args()
    params = PlotTrajectoryParams()
    
    # Validate input arguments
    if not args.csv and not args.bag and not args.bag_tf and not args.kitti:
        parser.error("At least one of --csv or --bag or --bag-tf or --kitti must be provided.")
    if int(bool(args.csv)) + int(bool(args.bag)) + int(bool(args.bag_tf)) + int(bool(args.kitti)) > 1:
        parser.error("Only one of --csv or --bag or --bag-tf or --kitti can be provided.")
    if args.bag and len(args.bag) != 2:
        parser.error("Invalid number of arguments for --bag. "
                     "Please provide both bag file path and topic name.")
    if args.bag_tf and len(args.bag_tf) != 3:
        parser.error("Invalid number of arguments for --bag-tf. "
                     "Please provide bag file path, parent frame, and child frame.")
    if args.kitti and len(args.kitti) != 2:
        parser.error("Invalid number of arguments for --kitti. "
                     "Please provide both KITTI dataset path and sequence ID.")

    # Load the pose data
    if args.csv:
        pose_data = PoseData.from_csv(args.csv, time_tol=params.time_tol)
    elif args.bag:
        bag_file, topic_name = args.bag
        pose_data = PoseData.from_bag(bag_file, topic_name, time_tol=params.time_tol)
    elif args.bag_tf:
        bag_file, parent_frame, child_frame = args.bag_tf
        pose_data = PoseData.from_bag_tf(bag_file, parent_frame, child_frame, time_tol=params.time_tol)
    elif args.kitti:
        dataset_path, sequence_id = args.kitti
        pose_data = PoseData.from_kitti(dataset_path, sequence_id, time_tol=params.time_tol)
    else:
        raise ValueError("No valid input source provided.")

    # Plot the trajectory
    ax = pose_data.plot2d(dt=params.trajectory_dt, axes=args.axes)
    pose_data.plot2d(ax=ax, dt=params.pose_dt, pose=True, trajectory=False, axes=args.axes)

    # Save the plot if an output path is provided
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()