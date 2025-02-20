import logging
import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)
logging.getLogger("ultralytics").setLevel(logging.WARNING)


class LicensePlateDetector:
    def __init__(self, model_path='yolo11n.pt', conf_threshold=0.5,iou_threshold=0.45):
        """
        :param model_path: Path to YOLO model
        :param conf_threshold: Confidence threshold for detection
        :param iou_threshold: IoU threshold for NMS
        """
        logger.debug(f"Loading YOLO model from {model_path} "
                     f"with conf threshold {conf_threshold}, iou {iou_threshold}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect_plates(self, frame):
        """
        Detect license plates in a given frame. 
        Returns a list of bounding boxes [x1, y1, x2, y2].
        """
        # YOLO expects: BGR or RGB image in numpy
        # The ultralytics YOLO automatically handles transformations
        results = self.model.predict(
        frame,
        conf=self.conf_threshold,
        iou = self.iou_threshold)
        
        bboxes = []
        # YOLOv8 returns results in results[0].boxes
        # Each box has xyxy format plus confidence, class, etc.
        for box in results[0].boxes:
            cls_id = box.cls.cpu().numpy()[0]
            conf = box.conf.cpu().numpy()[0]
            # Filter by confidence or by specific class if model has a "license plate" class
            if conf >= self.conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                bboxes.append([x1, y1, x2, y2])
        
        return bboxes
