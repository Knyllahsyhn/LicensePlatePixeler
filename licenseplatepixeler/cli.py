import argparse
import os
import glob
from processor import VideoProcessor
from licenseplatepixeler.detector import LicensePlateDetector
from utils import setup_logger, get_output_video_path, hardware_acceleration_supported

def process_files(
    input_files,
    tracker_type="KCF",
    detection_interval = 30,
    conf_threshold = 0.45,
    iou_threshold = 0.45,
    shrink_factor =0.1,
    verbose=False
    ):
    logger = setup_logger(verbose=verbose)

    detector = LicensePlateDetector(conf_threshold=conf_threshold,iou_threshold=iou_threshold)  # or specify custom model
    processor = VideoProcessor(
        detector=detector,
        tracker_type=tracker_type,
        tracking_enabled=True,
        detection_interval=detection_interval,
        iou_threshold=iou_threshold,
        shrink_factor=shrink_factor
        )

    def progress_callback(frame_index, total_frames):
        if total_frames is None:
            print(f"Processing frame {frame_index} ...", end='\r')
        else:
            percent = (frame_index / total_frames) * 100
            print(f"Processed frame {frame_index}/{total_frames} ({percent:.2f}%)", end='\r')

    for in_file in input_files:
        if not os.path.isfile(in_file):
            logger.warning(f"File {in_file} does not exist. Skipping.")
            continue
        out_file = get_output_video_path(in_file)
        logger.warning(f"Processing {in_file} -> {out_file}")
        processor.process_video(in_file, out_file, progress_callback=progress_callback)
        print()  # line break after each video

def run_cli():
    parser = argparse.ArgumentParser(description="Automatically blur license plates in videos.")
    parser.add_argument("paths", nargs="+", help="Paths to video files or folders to process.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--tracker", default="KCF", choices=["CSRT","KCF","MIL","MOSSE"],help="OpenCV tracker type to use (default KCF)")
    parser.add_argument("--interval", type=int, default=15,
                        help="Interval (in frames) for YOLO re-detection.")
    parser.add_argument("--conf-threshold", type=float, default=0.5,
                        help="Confidence threshold for YOLO. Default=0.5.")
    parser.add_argument("--iou-threshold", type=float, default=0.45,
                        help="IoU threshold for YOLO NMS. Default=0.45.")
    parser.add_argument("--shrink-factor", type=float, default=0.1,
                        help="Shrink bounding boxes by this factor (0.1=10%). Default=0.1.")
    
    args = parser.parse_args()

    # Collect input_files from the given paths
    input_files = []
    for path in args.paths:
        if os.path.isdir(path):
            # Gather all common video extensions in that folder
            for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
                input_files.extend(glob.glob(os.path.join(path, ext)))
        elif os.path.isfile(path):
            input_files.append(path)
        else:
            print(f"Invalid path: {path}")

    # Process them
    if input_files:
        process_files(
            input_files,
            verbose=args.verbose,
            detection_interval=args.interval,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            shrink_factor=args.shrink_factor
        )
    else:
        print("No valid video files to process.")
