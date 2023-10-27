# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 21:06:49 2023

@author: Aaron
"""

# from PyQt5 import QtWidgets
# from PyQt5.QtWidgets import QApplication, QMainWindow
# import sys

# def window():
#     app = QApplication(sys.argv) # config setup based on OS
#     win = QMainWindow()
#     win.setGeometry(10,10,1920-20,1080-20)
#     win.setWindowTitle("TestRun")
    
#     label = QtWidgets.QLabel(win)
#     label.setText("my first label!")
#     label.move(50,50)
    
#     win.show()
#     sys.exit(app.exec_())
    
# window()

import sys
import os
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap

class ImageDisplayApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Display")
        self.setGeometry(100, 100, 600, 400)

        # Create a central widget to hold the image and buttons
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setPixmap(QPixmap('aoa_distribution.png'))  # Initial image

        # Buttons to navigate through images
        self.prev_button = QPushButton("Previous", self)
        self.next_button = QPushButton("Next", self)

        # Create a list of image file paths
        self.image_files = [
            'aero_force.png',
            'drag_distribution.png',
            'lift_distribution',
            'Project_InterfaceChart.png'
        ]

        self.current_image_index = 0  # Index of the currently displayed image

        # Connect button clicks to functions
        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)

        # Create a vertical layout for the central widget
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.prev_button)
        layout.addWidget(self.next_button)
        central_widget.setLayout(layout)

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image()

    def show_next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_image()

    def show_image(self):
        image_path = self.image_files[self.current_image_index]
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap)

def main():
    # app = QApplication(sys.argv)
    # window = ImageDisplayApp()
    # window.show()
    # sys.exit(app.exec_())
    # app.setQuitOnLastWindowClosed(True)
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    main = ImageDisplayApp()
    main.show()

    return main

if __name__ == '__main__':
    m = main()