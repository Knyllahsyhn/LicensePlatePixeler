# modules/processor.py
import logging
import av
import cv2
from tqdm import tqdm

from detector import LicensePlateDetector
from tracker import PlateTracker

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(
        self,
        detector=None,
        tracking_enabled=True,
        detection_interval=30,
        tracker_type="CSRT",
        shrink_factor= 0.1
    ):
        """
        :param detector: LicensePlateDetector (YOLO) 
        :param tracking_enabled: If True, we use the PlateTracker
        :param detection_interval: Run YOLO every N frames 
        :param tracker_type: Which OpenCV tracker to use ("CSRT", "KCF", "MIL", etc.)
        """
        self.detector = detector if detector else LicensePlateDetector()
        self.tracking_enabled = tracking_enabled
        self.detection_interval = detection_interval
        self.tracker_type = tracker_type
        self.tracker = None
        self.shrink_factor = shrink_factor
    
    def shrink_bbox(self,x1, y1, x2, y2, shrink_factor=0.1):
        """Shrink each dimension of the bounding box by shrink_factor."""
        width = x2 - x1
        height = y2 - y1
        # Calculate new corners
        x1n = x1 + int(width * shrink_factor)
        y1n = y1 + int(height * shrink_factor)
        x2n = x2 - int(width * shrink_factor)
        y2n = y2 - int(height * shrink_factor)
        # Ensure we don’t invert the box
        if x2n < x1n: x2n = x1n
        if y2n < y1n: y2n = y1n
        return [x1n, y1n, x2n, y2n]

    def blur_bboxes(self, frame, bboxes, blur_kernel=(15,15)):
        for (x1, y1, x2, y2) in bboxes:
            # Ensure bounding box is within image
            x1c = max(x1, 0); y1c = max(y1, 0)
            x2c = min(x2, frame.shape[1]); y2c = min(y2, frame.shape[0])
            roi = frame[y1c:y2c, x1c:x2c]
            roi_blurred = cv2.blur(roi, blur_kernel)
            frame[y1c:y2c, x1c:x2c] = roi_blurred
        return frame

    def process_video(self, input_path, output_path, progress_callback=None):
        logger.debug(f"Opening input video {input_path}")
        
        input_container = av.open(input_path)
        input_video_stream = input_container.streams.video[0]
        total_frames = input_video_stream.frames if input_video_stream.frames > 0 else None

        logger.debug(f"Creating output container {output_path}")
        output_container = av.open(output_path, mode='w')
        output_video_stream = output_container.add_stream(codec_name='h264',rate=input_video_stream.average_rate)

        # Copy any audio
        input_audio_streams = [s for s in input_container.streams if s.type == 'audio']
        output_audio_streams = []
        for ias in input_audio_streams:
            oas = output_container.add_stream('aac')
            output_audio_streams.append(oas)

        # Initialize our PlateTracker if needed
        if self.tracking_enabled:
            self.tracker = PlateTracker(tracker_type=self.tracker_type)

        frame_index = 0
        if progress_callback is None and total_frames is not None:
            progress_bar = tqdm(total=total_frames, desc="Processing video")

        for packet in input_container.demux():
            if packet.stream.type == 'video':
                for frame in packet.decode():
                    frame_index += 1
                    img = frame.to_ndarray(format='bgr24')

                    # Decide whether to run YOLO detection or track
                    if not self.tracking_enabled:
                        # Always run YOLO if tracking is disabled
                        bboxes = self.detector.detect_plates(img)
                    else:
                        # If first frame or detection_interval has passed, run YOLO
                        if frame_index == 1 or (frame_index % self.detection_interval == 0):
                            bboxes = self.detector.detect_plates(img)
                            # Initialize or re-initialize the tracker
                            self.tracker.initialize(img, bboxes)
                        else:
                            # Update the existing trackers
                            bboxes = self.tracker.update(img)
                    
                    # Shrink bounding boxes if shrink_factor > 0
                    if self.shrink_factor > 0:
                        bboxes = [self.shrink_bbox(*bbox, self.shrink_factor) for bbox in bboxes]


                    # Blur the bounding boxes
                    self.blur_bboxes(img, bboxes)

                    # Re-encode the frame
                    new_frame = av.VideoFrame.from_ndarray(img, format='bgr24')
                    new_frame.pts = frame.pts
                    new_frame.time_base = frame.time_base
                    packet_out = output_video_stream.encode(new_frame)
                    if packet_out:
                        output_container.mux(packet_out)

                    # Progress callback or tqdm update
                    if progress_callback:
                        progress_callback(frame_index, total_frames)
                    else:
                        if total_frames:
                            progress_bar.update(1)

            elif packet.stream.type == 'audio':
                # Copy audio frames
                for audio_frame in packet.decode():
                    packet_out = output_audio_streams[packet.stream.index - 1].encode(audio_frame)
                    if packet_out:
                        output_container.mux(packet_out)

        # Flush
        for stream in output_container.streams:
            packet_out = stream.encode(None)
            if packet_out:
                output_container.mux(packet_out)

        input_container.close()
        output_container.close()
        logger.debug("Processing complete.")
