#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
import argparse
import argcomplete
import ast
import cv2
from dataclasses import dataclass

from robotdatapy.data import ImgData

@dataclass
class ImgDataToMp4Params:

    fps: int = 30

def main():

    parser = argparse.ArgumentParser(description="Quick ImgData to mp4 converter tool.")
    parser.add_argument("-b", "--bag", type=str, nargs=2, metavar=('BAG_FILE', 'TOPIC_NAME'),
                        help="Path to a ROS bag file and image topic name.")
    parser.add_argument("-k", "--kitti", type=str, nargs=2, metavar=('KITTI_DATASET_PATH', 'KITTI_TYPE'),
                        help="Path to a KITTI dataset and kitti_type (e.g. 'cam2').")
    parser.add_argument("-z", "--zip", type=str, metavar='ZIP_FILE',
                        help="Path to a zip file containing image data.")
    parser.add_argument("-n", "--npz", type=str, metavar='NPZ_FILE',
                        help="Path to an npz file containing image data.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path to save the output mp4 file.")
    parser.add_argument("--fps", type=int, default=ImgDataToMp4Params.fps,
                        help=f"Frames per second for the output mp4 (default: {ImgDataToMp4Params.fps}).")
    parser.add_argument("--kwargs", type=str, default=None,
                        help="Extra kwargs forwarded to the ImgData loader, as a Python dict literal "
                             "string (e.g. \"{'compressed': False, 'stride': 2}\").")
    parser.add_argument("--colormap", type=str, default=None,
                        help="OpenCV colormap name (e.g. 'jet', 'turbo', 'viridis') for "
                             "visualizing single-channel/depth streams as color video.")
    parser.add_argument("--norm-range", type=float, nargs=2, metavar=('MIN', 'MAX'), default=None,
                        help="Fixed (min, max) for colormap normalization in the source "
                             "image's native units (e.g. mm for L515 depth). Without this "
                             "flag, each frame is normalized independently.")

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # Validate input arguments
    if not args.bag and not args.kitti and not args.zip and not args.npz:
        parser.error("At least one of --bag or --kitti or --zip or --npz must be provided.")
    if int(bool(args.bag)) + int(bool(args.kitti)) + int(bool(args.zip)) + int(bool(args.npz)) > 1:
        parser.error("Only one of --bag or --kitti or --zip or --npz can be provided.")
    if args.bag and len(args.bag) != 2:
        parser.error("Invalid number of arguments for --bag. "
                     "Please provide both bag file path and topic name.")
    if args.kitti and len(args.kitti) != 2:
        parser.error("Invalid number of arguments for --kitti. "
                     "Please provide both KITTI dataset path and kitti_type.")

    # Parse extra kwargs
    extra_kwargs = {}
    if args.kwargs:
        parsed = ast.literal_eval(args.kwargs)
        if not isinstance(parsed, dict):
            parser.error("--kwargs must be a dict literal, e.g. \"{'stride': 2}\".")
        extra_kwargs = parsed

    # Load the image data
    if args.bag:
        bag_file, topic_name = args.bag
        img_data = ImgData.from_bag(bag_file, topic_name, **extra_kwargs)
    elif args.kitti:
        dataset_path, kitti_type = args.kitti
        img_data = ImgData.from_kitti(dataset_path, kitti_type, **extra_kwargs)
    elif args.zip:
        img_data = ImgData.from_zip(args.zip, **extra_kwargs)
    elif args.npz:
        img_data = ImgData.from_npz(args.npz, **extra_kwargs)
    else:
        raise ValueError("No valid input source provided.")

    # Resolve colormap name to cv2 constant
    colormap = None
    if args.colormap:
        attr = f"COLORMAP_{args.colormap.upper()}"
        if not hasattr(cv2, attr):
            parser.error(f"Unknown colormap '{args.colormap}'. Try 'jet', 'turbo', 'viridis', etc.")
        colormap = getattr(cv2, attr)

    norm_range = tuple(args.norm_range) if args.norm_range else None

    # Write the mp4
    img_data.to_mp4(args.output, fps=args.fps, colormap=colormap, norm_range=norm_range)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
