from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QTimer
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton

import sys
import os
from model import BulkExcelLoader

import logging

# Basic logger setup without a file handler
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

class InfoDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(InfoDialog, self).__init__(parent)
        uic.loadUi('info_dialog.ui', self)  # Load the .ui file for the dialog
        self.setWindowTitle("Information")
        
        self.pushButton_Continuar.clicked.connect(self.close)
        self.label.setText("")
        
    def set_text(self, valid, total):
        self.label.setText(f"Archivos con extensión válida: {valid}/{total}")

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('uibasic.ui', self) # Load the .ui file
        self.setWindowTitle("Inecon: calculador de VAN")
        self.show() # Show the GUI
        
        self.pushButton_Ejecutar.setEnabled(False) #can only be pressed after a table is generated
                
        self.pushButton_Cargar.clicked.connect(self.cargar_carpeta)
        self.pushButton_Ejecutar.clicked.connect(self.ejecutar_algoritmo)
        
        self.model = BulkExcelLoader()
        self.logger = logging.getLogger(__name__)
        
    def openInfoDialog(self, valid, total):
        dialog = InfoDialog(self)
        dialog.set_text(valid, total)
        dialog.exec_()  # Show the dialog
        
    def cargar_carpeta(self):
        self.folder_route = QFileDialog.getExistingDirectory()
        
        log_file_path = os.path.join(self.folder_route, 'app.log')
        # Create a file handler and set it to the logger
        file_handler = logging.FileHandler(log_file_path, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        try:
            self.model.load_filesList_by_folderName(self.folder_route)
            self.label_filename.setText(self.model.folderName)
            valid, total = self.model.get_count_valid_and_total_filenames()
            self.logger.info('\n' + self.model.print_log_filtering())
            self.openInfoDialog(valid, total)
            self.label_progress.setText(f"0/{valid}")
            self.pushButton_Ejecutar.setEnabled(True)
                   
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))  # Show error message in a dialog box
            print(str(e))
                         
    def ejecutar_algoritmo(self):
        self.pushButton_Ejecutar.setEnabled(False)
        self.worker = Worker(self.model, self.logger)
        self.thread = QThread()  # Create a QThread object
        self.worker.moveToThread(self.thread)  # Move worker to the thread

        # Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.progress_signal.connect(self.update_tiempoRestante)
        self.worker.completed_signal.connect(self.on_task_completed)
        
        # Set up and start the QTimer
        self.loading_timer = QTimer(self)
        self.loading_timer.timeout.connect(self.update_loading_label)
        self.loading_timer.start(500)  # Change the label every 500 milliseconds

        # Start the thread
        self.thread.start()
        
    def update_tiempoRestante(self, actual, total):
        self.label_progress.setText(f"{actual}/{total}")
        
    def update_loading_label(self):
        current_text = self.label_loading.text()
        if current_text.count('.') < 3:
            self.label_loading.setText("Cargando" + "." * (current_text.count('.') + 1))
        else:
            self.label_loading.setText("Cargando")
            
    def clean_loading_label(self):
        self.label_loading.setText("")
        
    def on_task_completed(self):
        self.clean_up()
        self.clean_loading_label()
        QMessageBox.information(self, "Completado", f"Completado exitosamente!")
        
        
    def clean_up(self):
        self.loading_timer.stop()
        self.thread.quit()  # Stop the thread
        self.thread.wait()  # Wait for the thread to finish
        self.thread.deleteLater()  # Schedule the thread for deletion
        self.worker.deleteLater()  # Schedule the worker for deletion
        
class Worker(QObject):
    progress_signal = pyqtSignal(int, int)  # Signal to update progress
    completed_signal = pyqtSignal()

    def __init__(self, model, logger):
        super().__init__()
        self.logger = logger
        self.model = model

    def run(self):
        output_str = ""
        filename_list = self.model.get_valid_filenames()
        valid, total = self.model.get_count_valid_and_total_filenames()
        output_str+="\n=============\nALGORITMO"
        for i, filename in enumerate(filename_list):
            self.progress_signal.emit(i+1, valid)
            try:
                self.model.run_algorithm_on_filename(filename)
                output_str+=(f"\n{filename}: OK.")
            except:
                output_str+=(f"\n{filename}: Error. formato de entrada de datos inválido.")
        self.logger.info(output_str)
        self.completed_signal.emit()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)  # Create an instance of QtWidgets.QApplication
    window = Ui()  # Create an instance of our class
    app.exec_()  # Start the application
