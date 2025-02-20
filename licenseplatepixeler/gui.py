import os
import glob
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QProgressBar,
    QCheckBox, QSpinBox, QDoubleSpinBox, QHBoxLayout, QFormLayout, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from processor import VideoProcessor
from detector import LicensePlateDetector
from utils import setup_logger, get_output_video_path

class ProcessingThread(QThread):
    progress_signal = pyqtSignal(str)  # For text updates
    progress_bar_signal = pyqtSignal(int, int)  # For progress bar
    
    def __init__(self, files, tracker_type, detection_interval = 30, conf_threshold = 0.5, iou_threshold = 0.45, shrink_factor = 0.1, verbose=False):
        super().__init__()
        self.files = files
        self.verbose = verbose
        self.tracker_type= tracker_type
        self.detection_interval = detection_interval
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.shrink_factor = shrink_factor

    def run(self):
        logger = setup_logger(verbose=self.verbose)
        detector = LicensePlateDetector(
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold)
        processor = VideoProcessor(
            detector=detector,
            tracker_type=self.tracker_type,
            detection_interval=self.detection_interval,
            shrink_factor=self.shrink_factor
            )

        def progress_callback(frame_index, total_frames):
            if total_frames:
                self.progress_bar_signal.emit(frame_index, total_frames)
            self.progress_signal.emit(f"Processing frame {frame_index}/{total_frames or '?'}")

        for idx, file in enumerate(self.files):
            self.progress_signal.emit(f"Processing file {idx+1}/{len(self.files)}: {file}")
            out_file = get_output_video_path(file)
            processor.process_video(file, out_file, progress_callback=progress_callback)

class BlurApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        #self.tracker_type="KCF"
        self.advanced_mode = False

    def init_ui(self):
        self.setWindowTitle("License Plate Blurrer")

        main_layout = QVBoxLayout()
        

        self.select_file_btn = QPushButton("Select Video File")
        self.select_file_btn.clicked.connect(self.select_file)

        self.select_folder_btn = QPushButton("Select Folder")
        self.select_folder_btn.clicked.connect(self.select_folder)
        
        self.tracker_combo= QComboBox()
        self.tracker_combo.addItems([ "KCF", "CSRT", "MIL", "MOSSE"])
        self.tracker_combo.currentTextChanged.connect(self.on_tracker_changed)
        
        # Advanced mode checkbox
        self.advanced_checkbox = QCheckBox("Advanced Mode")
        self.advanced_checkbox.stateChanged.connect(self.toggle_advanced_mode)

        self.start_btn = QPushButton("Start Processing")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setEnabled(False)

        self.status_label = QLabel("Status: Idle")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        # Advanced options (hidden by default)
        self.advanced_layout = QFormLayout()
        # Detection interval
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 999)
        self.interval_spin.setValue(30)
        self.advanced_layout.addRow("Detection Interval (frames)", self.interval_spin)

        # Confidence threshold
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.5)
        self.advanced_layout.addRow("Confidence Threshold", self.conf_spin)

        # IoU threshold
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        self.advanced_layout.addRow("IoU Threshold", self.iou_spin)

        # Shrink factor
        self.shrink_spin = QDoubleSpinBox()
        self.shrink_spin.setRange(0.0, 1.0)
        self.shrink_spin.setSingleStep(0.05)
        self.shrink_spin.setValue(0.1)
        self.advanced_layout.addRow("Box Shrink Factor", self.shrink_spin)

        # Hide advanced layout initially
        advanced_container = QWidget()
        advanced_container.setLayout(self.advanced_layout)
        advanced_container.setVisible(False)
        self.advanced_container = advanced_container  # store reference

        #load widgets
        widgets = [
            self.select_file_btn,
            self.select_folder_btn,
            self.start_btn,
            self.advanced_container,
            self.advanced_checkbox,
            self.status_label,
            self.progress_bar,
            self.tracker_combo
        ]
        
        for w in widgets:
            main_layout.addWidget(w)
        

        self.setLayout(main_layout)

        self.files_to_process = []

    def toggle_advanced_mode(self, state:bool):
        self.advanced_container.setVisible(state)
   
   
    def select_file(self):
        file_dialog = QFileDialog(self, "Select Video File")
        file_dialog.setNameFilters(["Video files (*.mp4 *.mov *.avi *.mkv)"])
        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            selected_files = file_dialog.selectedFiles()
            self.files_to_process.extend(selected_files)
            self.update_status(f"Added {len(selected_files)} file(s).")
            self.start_btn.setEnabled(True)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            # Gather all video files
            video_extensions = ("*.mp4", "*.mov", "*.avi", "*.mkv")
            for ext in video_extensions:
                self.files_to_process.extend(glob.glob(os.path.join(folder, ext)))
            self.update_status(f"Added {len(self.files_to_process)} file(s) from folder.")
            self.start_btn.setEnabled(True)

    def start_processing(self):
        detection_interval = self.interval_spin.value()
        conf_threshold = self.conf_spin.value()
        iou_threshold = self.iou_spin.value()
        shrink_factor = self.shrink_spin.value()
        tracker_type = self.tracker_combo.currentText()
        
        self.processing_thread = ProcessingThread(
            self.files_to_process,
            tracker_type=tracker_type,
            detection_interval=detection_interval,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            shrink_factor=shrink_factor,
            verbose=False)
        self.processing_thread.progress_signal.connect(self.update_status)
        self.processing_thread.progress_bar_signal.connect(self.update_progress_bar)
        self.processing_thread.finished.connect(self.processing_done)
        self.processing_thread.start()
        self.start_btn.setEnabled(False)
    
    def on_tracker_changed(self, text):
        self.tracker_type = text
        self.update_status(f"Tracker changed to: {text}")

    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    def update_progress_bar(self, frame_index, total_frames):
        percent = int((frame_index / total_frames) * 100)
        self.progress_bar.setValue(percent)

    def processing_done(self):
        self.update_status("Processing complete.")
        self.start_btn.setEnabled(True)
        self.files_to_process = []

def run_gui():
    app = QApplication([])
    window = BlurApp()
    window.show()
    app.exec()
