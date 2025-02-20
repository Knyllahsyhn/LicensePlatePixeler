import os
import glob
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QProgressBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from processor import VideoProcessor
from detector import LicensePlateDetector
from utils import setup_logger, get_output_video_path, hardware_acceleration_supported

class ProcessingThread(QThread):
    progress_signal = pyqtSignal(str)  # For text updates
    progress_bar_signal = pyqtSignal(int, int)  # For progress bar
    
    def __init__(self, files, hw_accel=False, verbose=False):
        super().__init__()
        self.files = files
        self.hw_accel = hw_accel
        self.verbose = verbose

    def run(self):
        logger = setup_logger(verbose=self.verbose)
        detector = LicensePlateDetector()
        processor = VideoProcessor(detector=detector, use_hw_accel=self.hw_accel and hardware_acceleration_supported())

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

    def init_ui(self):
        self.setWindowTitle("License Plate Blurrer")

        layout = QVBoxLayout()

        self.select_file_btn = QPushButton("Select Video File")
        self.select_file_btn.clicked.connect(self.select_file)

        self.select_folder_btn = QPushButton("Select Folder")
        self.select_folder_btn.clicked.connect(self.select_folder)

        self.start_btn = QPushButton("Start Processing")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setEnabled(False)

        self.status_label = QLabel("Status: Idle")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        layout.addWidget(self.select_file_btn)
        layout.addWidget(self.select_folder_btn)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

        self.files_to_process = []

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
        self.processing_thread = ProcessingThread(self.files_to_process, hw_accel=False, verbose=False)
        self.processing_thread.progress_signal.connect(self.update_status)
        self.processing_thread.progress_bar_signal.connect(self.update_progress_bar)
        self.processing_thread.finished.connect(self.processing_done)
        self.processing_thread.start()
        self.start_btn.setEnabled(False)

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
