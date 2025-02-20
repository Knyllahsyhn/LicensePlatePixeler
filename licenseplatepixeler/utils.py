import logging
import sys
import os

def setup_logger(verbose=False):
    """
    Sets up logging.
    If verbose is False, set the logging level to WARNING (or ERROR) 
    to suppress most messages from YOLO, PyAV, etc.
    If verbose is True, set the logging to DEBUG.
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format='[%(levelname)s] %(name)s: %(message)s'
    )
    logger = logging.getLogger(__name__)
    return logger


def get_output_video_path(input_path):
    """
    Returns the output path by appending '_blurred' before the file extension.
    e.g. /path/to/video.mp4 -> /path/to/video_blurred.mp4
    """
    base, ext = os.path.splitext(input_path)
    return f"{base}_blurred{ext}"


def hardware_acceleration_supported():
    """
    Check if hardware acceleration (e.g. GPU decoding/encoding) is available.
    This is highly system-dependent and typically requires particular 
    builds of ffmpeg, drivers, etc.
    Return True if available, else False.
    """
    # Simplified example. Real checks can be more involved.
    # For example, checking for nvcodec, vaapi, or videotoolbox, etc.
    return False
