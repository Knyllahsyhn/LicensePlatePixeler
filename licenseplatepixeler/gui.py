import sys
import os
from PyQt6.QtWidgets import QApplication, QFileDialog, QVBoxLayout, QPushButton, QLabel, QWidget, QProgressBar
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from processor import process_video

class ProcessingThread(QThread):
    progress_signal = pyqtSignal(int)

    def __init__(self, input_path, output_path):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path

    def run(self):
        process_video(self.input_path, self.output_path, self.progress_signal)

class VideoBlurGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("License Plate Pixelation Tool")
        self.setGeometry(200, 200, 400, 200)

        layout = QVBoxLayout()

        self.label = QLabel("Select a video file or folder:")
        layout.addWidget(self.label)

        self.select_button = QPushButton("Select File")
        self.select_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_button)

        self.start_button = QPushButton("Start Processing")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_processing)
        layout.addWidget(self.start_button)

        self.progress = QProgressBar(self)
        self.progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress)

        self.setLayout(layout)

    def select_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Videos (*.mp4 *.avi)", options=options)
        if file_name:
            self.input_path = file_name
            self.output_path = file_name.replace(".mp4", "_blurred.mp4")
            self.start_button.setEnabled(True)

    def start_processing(self):
        self.thread = ProcessingThread(self.input_path, self.output_path)
        self.thread.progress_signal.connect(self.progress.setValue)
        self.thread.start()

def run_gui():
    app = QApplication(sys.argv)
    gui = VideoBlurGUI()
    gui.show()
    sys.exit(app.exec())

