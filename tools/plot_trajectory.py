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
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Path to save the trajectory plot image (optional).")
    
    args = parser.parse_args()
    params = PlotTrajectoryParams()

    # Load the pose data
    pose_data = PoseData.from_csv(args.csv, time_tol=params.time_tol)

    # Plot the trajectory
    ax = pose_data.plot2d(dt=params.trajectory_dt)
    pose_data.plot2d(ax=ax, dt=params.pose_dt, pose=True, trajectory=False)

    # Save the plot if an output path is provided
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()