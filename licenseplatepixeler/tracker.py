# modules/tracker.py
import cv2


def create_single_tracker(tracker_type="KCF"):
    """
    Create an OpenCV tracker instance for the specified type.
    Common options:
      - "CSRT": cv2.TrackerCSRT_create()
      - "KCF": cv2.TrackerKCF_create()
      - "MIL": cv2.TrackerMIL_create()
      - "MOSSE": cv2.TrackerMOSSE_create()
      - etc.
    """
    if tracker_type == "CSRT":
        return cv2.TrackerCSRT_create()
    elif tracker_type == "KCF":
        return cv2.TrackerKCF_create()
    elif tracker_type == "MIL":
        return cv2.TrackerMIL_create()
    elif tracker_type == "MOSSE":
        return cv2.TrackerMOSSE_create()
    else:
        raise ValueError(f"Unsupported tracker type: {tracker_type}")


class PlateTracker:
    """
    A manual multi-tracker that uses individual OpenCV trackers for each bounding box.
    """
    def __init__(self, tracker_type="CSRT"):
        """
        :param tracker_type: One of "CSRT", "KCF", "MIL", "MOSSE", etc.
        """
        self.tracker_type = tracker_type
        self.trackers = []  # will hold (tracker, last_bbox)
    
    def initialize(self, frame, bboxes):
        """
        Clear existing trackers and create new ones based on the provided bboxes.
        
        :param frame: the current video frame as a NumPy array (BGR).
        :param bboxes: list of bounding boxes [x1, y1, x2, y2].
        """
        self.trackers.clear()
        for (x1, y1, x2, y2) in bboxes:
            w = x2 - x1
            h = y2 - y1
            # Create a new single-object tracker
            single_tracker = create_single_tracker(self.tracker_type)
            # Initialize it with the region of interest
            single_tracker.init(frame, (x1, y1, w, h))
            # Store it
            self.trackers.append((single_tracker, (x1, y1, w, h)))

    def update(self, frame):
        """
        Update each tracker on the new frame.
        :return: list of [x1, y1, x2, y2] bounding boxes (updated positions).
        """
        updated_bboxes = []
        new_trackers = []

        for (tracker, last_bbox) in self.trackers:
            success, bbox = tracker.update(frame)
            if success:
                # bbox is (x, y, w, h)
                x, y, w, h = bbox
                x1 = int(x)
                y1 = int(y)
                x2 = int(x + w)
                y2 = int(y + h)
                updated_bboxes.append([x1, y1, x2, y2])
                new_trackers.append((tracker, (x1, y1, w, h)))
            else:
                # If tracking fails, we won't keep this tracker
                pass
        
        # Update the internal trackers list with only successful trackers
        self.trackers = new_trackers
        return updated_bboxes
