import av
import cv2
import numpy as np
from detection import detect_license_plates
from utils import add_pixelation

def process_video(input_video, output_video, progress_signal=None):
    container = av.open(input_video)
    output_container = av.open(output_video, mode='w')

    # Copy video/audio stream format
    stream = container.streams.video[0]
    output_stream = output_container.add_stream('h264', rate=stream.rate)
    output_stream.width = stream.width
    output_stream.height = stream.height

    # Process frames
    total_frames = stream.frames
    for frame_no, frame in enumerate(container.decode(video=0)):
        img = frame.to_ndarray(format='bgr24')
        plates = detect_license_plates(img)
        img = add_pixelation(img, plates)

        # Convert back to AV frame and write to output
        new_frame = av.VideoFrame.from_ndarray(img, format='bgr24')
        output_container.mux(output_stream.encode(new_frame))

        if progress_signal:
            progress_signal.emit(int((frame_no / total_frames) * 100))

    # Copy audio stream
    for packet in container.demux(audio=0):
        output_container.mux(packet)

    output_container.close()
    container.close()
