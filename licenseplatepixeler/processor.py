import logging
import av
import numpy as np
import cv2
from detector import LicensePlateDetector
from tqdm import tqdm

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, detector=None, use_hw_accel=False):
        """
        Initialize the processor with an optional Detector 
        and hardware acceleration flag.
        """
        self.detector = detector if detector else LicensePlateDetector()
        self.use_hw_accel = use_hw_accel

    def blur_bboxes(self, frame, bboxes, blur_kernel=(15,15)):
        """
        Blur the regions specified by bboxes in the frame.
        """
        for (x1, y1, x2, y2) in bboxes:
            roi = frame[y1:y2, x1:x2]
            roi_blurred = cv2.blur(roi, blur_kernel)
            frame[y1:y2, x1:x2] = roi_blurred
        return frame

    def process_video(self, input_path, output_path, progress_callback=None):
        """
        Process a single video:
        - Read frames with PyAV
        - Detect plates
        - Blur them
        - Write to output (with audio)
        
        progress_callback: a callable for reporting progress (frame index, total frames).
        """
        logger.debug(f"Opening input video {input_path}")
        
        # Open input container
        input_container = av.open(input_path)

        # Retrieve input video stream info
        input_video_stream = input_container.streams.video[0]
        total_frames = input_video_stream.frames if input_video_stream.frames > 0 else None

        # For hardware acceleration (if supported), you can set 'hwaccel' or 'codec' in the Container/Stream options.
        # This is highly dependent on your ffmpeg build and GPU. 
        # See PyAV docs for how to pass these options: https://pyav.org/docs/stable/ 
        # Example (not guaranteed to work on your system):
        # if self.use_hw_accel:
        #     input_video_stream.codec_context.hwaccel = 'cuda'
        #     # or something similar, plus you need to specify the device, etc.

        # Create output container
        logger.debug(f"Creating output container {output_path}")
        output_container = av.open(output_path, mode='w')
        
        # Add video stream to output container
        output_video_stream = output_container.add_stream('h264',rate=input_video_stream.base_rate)
        # Add audio streams if any
        input_audio_streams = [s for s in input_container.streams if s.type == 'audio']
        output_audio_streams = []
        for ias in input_audio_streams:
            oas = output_container.add_stream("aac")
            output_audio_streams.append(oas)

        frame_index = 0
        
        # Use tqdm for a local progress bar if no callback. Otherwise, rely on the callback.
        if progress_callback is None and total_frames is not None:
            progress_bar = tqdm(total=total_frames, desc="Processing video")

        for packet in input_container.demux():
            # If this is a video packet
            if packet.stream.type == 'video':
                for frame in packet.decode():
                    frame_index += 1
                    # Convert PyAV frame to numpy array (BGR)
                    img = frame.to_ndarray(format='bgr24')
                    
                    # Detect license plates
                    bboxes = self.detector.detect_plates(img)
                    # Blur them
                    img = self.blur_bboxes(img, bboxes)

                    # Convert back to PyAV VideoFrame
                    new_frame = av.VideoFrame.from_ndarray(img, format='bgr24')
                    new_frame.pts = frame.pts
                    new_frame.time_base = frame.time_base

                    packet_out = output_video_stream.encode(new_frame)
                    if packet_out:
                        output_container.mux(packet_out)

                    # Update progress
                    if progress_callback:
                        # e.g. progress_callback(frame_index, total_frames)
                        progress_callback(frame_index, total_frames)
                    else:
                        if total_frames is not None:
                            progress_bar.update(1)

            # If this is an audio packet
            elif packet.stream.type == 'audio':
                for audio_frame in packet.decode():
                    packet_out = output_audio_streams[packet.stream.index - 1].encode(audio_frame)
                    if packet_out:
                        output_container.mux(packet_out)

        # Flush the streams
        for stream in output_container.streams:
            packet_out = stream.encode(None)
            if packet_out:
                output_container.mux(packet_out)

        # Close containers
        input_container.close()
        output_container.close()

        logger.debug("Processing complete.")
