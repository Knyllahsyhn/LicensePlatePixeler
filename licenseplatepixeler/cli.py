import os
import sys
import glob
import tqdm
from processor import process_video
from utils import setup_logging

def run_cli(input_path, output_path, verbose=False):
    setup_logging(verbose)

    # Check if input is a directory or single file
    if os.path.isdir(input_path):
        video_files = glob.glob(os.path.join(input_path, "*.mp4"))
    else:
        video_files = [input_path]

    if not video_files:
        print("No video files found.")
        sys.exit(1)

    print(f"Processing {len(video_files)} video(s)...")

    for video in tqdm.tqdm(video_files, desc="Processing Videos", unit="video"):
        output_file = os.path.join(output_path, os.path.basename(video).replace(".mp4", "_blurred.mp4"))
        process_video(video, output_file)

    print("Processing complete!")

