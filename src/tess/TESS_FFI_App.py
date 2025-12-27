import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QProgressBar
from PyQt5.QtCore import Qt
import FFIDownloader as dffi
import FFICalibrate as cffi
import FFIStarFinder as sffi
# FFILcCreator was removed - use LightcurveBuilder instead
# import FFILcCreator as lffi
import os

print(f"Current working directory: {os.getcwd()}")

class TESS_FFI_App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TESS FFI Data Processing")
        self.setGeometry(100, 100, 400, 500)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.init_ui()

        self.fits_files = []
        self.calibrated_data = None
        self.star_data = None

    def init_ui(self):
        # Input fields
        input_layout = QVBoxLayout()
        self.entries = {}
        for label in ["Sector", "Year", "Day", "Camera", "CCD"]:
            row_layout = QHBoxLayout()
            row_layout.addWidget(QLabel(f"{label}:"))
            self.entries[label.lower()] = QLineEdit()
            row_layout.addWidget(self.entries[label.lower()])
            input_layout.addLayout(row_layout)
        self.layout.addLayout(input_layout)

        # Buttons
        self.download_button = QPushButton("Download FFIs")
        self.download_button.clicked.connect(self.download_ffis)
        self.layout.addWidget(self.download_button)

        self.calibrate_button = QPushButton("Calibrate FFIs")
        self.calibrate_button.clicked.connect(self.calibrate_ffis)
        self.calibrate_button.setEnabled(False)
        self.layout.addWidget(self.calibrate_button)

        self.find_stars_button = QPushButton("Find Stars")
        self.find_stars_button.clicked.connect(self.find_stars)
        self.find_stars_button.setEnabled(False)
        self.layout.addWidget(self.find_stars_button)

        # RA/Dec inputs
        self.ra_dec_layout = QHBoxLayout()
        self.ra_label = QLabel("RA:")
        self.ra_entry = QLineEdit()
        self.dec_label = QLabel("Dec:")
        self.dec_entry = QLineEdit()
        self.ra_dec_layout.addWidget(self.ra_label)
        self.ra_dec_layout.addWidget(self.ra_entry)
        self.ra_dec_layout.addWidget(self.dec_label)
        self.ra_dec_layout.addWidget(self.dec_entry)
        self.layout.addLayout(self.ra_dec_layout)
        self.ra_label.hide()
        self.ra_entry.hide()
        self.dec_label.hide()
        self.dec_entry.hide()

        self.create_lc_button = QPushButton("Create Lightcurve")
        self.create_lc_button.clicked.connect(self.create_lightcurve)
        self.create_lc_button.setEnabled(False)
        self.layout.addWidget(self.create_lc_button)

        # Status label and progress bar
        self.status_label = QLabel("Enter parameters and click 'Download FFIs'")
        self.layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)

    def download_ffis(self):
        inputs = {key: entry.text() for key, entry in self.entries.items()}
        try:
            self.fits_files = dffi.download_fits(inputs)
            if self.fits_files:
                QMessageBox.information(self, "Success", "FFIs downloaded successfully!")
                self.status_label.setText("FFIs downloaded. You can now calibrate them.")
                self.calibrate_button.setEnabled(True)
            else:
                QMessageBox.critical(self, "Error", "Failed to download FFIs.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during download: {str(e)}")

    def calibrate_ffis(self):
        try:
            self.calibrated_data = cffi.calibrate_background(self.fits_files)
            if self.calibrated_data is not None:
                QMessageBox.information(self, "Success", "FFIs calibrated successfully!")
                self.status_label.setText("FFIs calibrated. You can now find stars.")
                self.find_stars_button.setEnabled(True)
            else:
                QMessageBox.critical(self, "Error", "Calibration failed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during calibration: {str(e)}")

    def find_stars(self):
        try:
            self.star_data = sffi.find_stars(self.calibrated_data)
            if self.star_data is not None:
                QMessageBox.information(self, "Success", "Stars found successfully!")
                self.status_label.setText("Stars found. You can now create a lightcurve.")
                self.create_lc_button.setEnabled(True)
                self.ra_label.show()
                self.ra_entry.show()
                self.dec_label.show()
                self.dec_entry.show()
            else:
                QMessageBox.critical(self, "Error", "Star finding failed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during star finding: {str(e)}")

    def create_lightcurve(self):
        # Legacy GUI - lightcurve creation moved to CLI
        # Use: tess-ffi lightcurves --star STAR_XXXXXX
        QMessageBox.information(
            self,
            "Feature Moved",
            "Lightcurve creation is now available via CLI:\n\n"
            "tess-ffi lightcurves --star STAR_000123\n\n"
            "Or use the Python API:\n"
            "from tess.LightcurveBuilder import LightcurveCollection"
        )
        self.status_label.setText("Use CLI for lightcurves: tess-ffi lightcurves")

def main():
    app = QApplication(sys.argv)
    window = TESS_FFI_App()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()