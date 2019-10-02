from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget)
from PyQt5.QtGui import QImage, QPixmap
import sys


class FPMGUI(QDialog):
    def __init__(self, parent=None):
        super(FPMGUI, self).__init__(parent)

        self.originalPalette = QApplication.palette()

        styleComboBox = QComboBox()
        styleComboBox.addItems(['Reconstruct', 'Collect'])

        styleLabel = QLabel("&Mode:")
        styleLabel.setBuddy(styleComboBox)

        self.RightGroup()
        self.LeftGroup()
        #self.createProgressBar()

        topLayout = QHBoxLayout()
        topLayout.addWidget(styleLabel)
        topLayout.addWidget(styleComboBox)
        topLayout.addStretch(1)

        Layout = QGridLayout()
        Layout.addLayout(topLayout, 0, 0, 1, 2)
        #mainLayout.addWidget(self.topLeftGroupBox, 1, 0)
        Layout.addWidget(self.RightGridWidget, 1, 1)
        Layout.addWidget(self.LeftGridWidget, 1, 0)
        #mainLayout.addWidget(self.progressBar, 3, 0, 1, 2)
        Layout.setRowStretch(1, 1)
        Layout.setRowStretch(2, 1)
        Layout.setColumnStretch(0, 1)
        Layout.setColumnStretch(1, 1)
        self.setLayout(Layout)

        self.setWindowTitle("FPM")
        QApplication.setStyle(QStyleFactory.create('Fusion'))
        QApplication.setPalette(self.originalPalette)

    def advanceProgressBar(self):
        curVal = self.progressBar.value()
        maxVal = self.progressBar.maximum()
        self.progressBar.setValue(curVal + (maxVal - curVal) / 100)


    def RightGroup(self):
        self.RightGridWidget = QGroupBox("Process")

        image = QImage('./sample1.jpg')
        pic = QLabel(self)
        pic.setPixmap(QPixmap(image))

        self.sRGroup()
        self.createProgressBar()

        Layout = QGridLayout()
        Layout.addWidget(pic, 0, 0)
        Layout.addWidget(self.sRGWidget, 1, 0)
        Layout.addWidget(self.progressBar, 2, 0)

        self.RightGridWidget.setLayout(Layout)


    def sRGroup(self):
        self.sRGWidget = QGroupBox()
        startButton = QPushButton("Start")
        startButton.setDefault(False)
        saveButton = QPushButton("Save")
        saveButton.setDefault(False)

        Layout = QGridLayout()
        Layout.addWidget(startButton, 0, 0)
        Layout.addWidget(saveButton, 0, 1)

        self.sRGWidget.setLayout(Layout)


    def LeftGroup(self):
        self.LeftGridWidget = QGroupBox("Settings")

        self.sLGroup()

        Layout = QGridLayout()
        Layout.addWidget(self.sLGroupWid, 0, 0)
        Layout.addWidget(self.sLGroupWid1, 1, 0)
        Layout.addWidget(self.sLGroupWid2, 2, 0)

        self.LeftGridWidget.setLayout(Layout)


    def sLGroup(self):
        self.sLGroupWid = QGroupBox('System Params')

        paramEntry = QLineEdit('File')
        paramButton = QPushButton("Select")

        Layout = QGridLayout()
        Layout.addWidget(paramEntry, 0, 0)
        Layout.addWidget(paramButton, 0, 1)
        self.sLGroupWid.setLayout(Layout)

        self.sLGroupWid1 = QGroupBox('Images')

        imgEntry = QLineEdit('File')
        imgButton = QPushButton("Select")

        Layout = QGridLayout()
        Layout.addWidget(imgEntry, 0, 0)
        Layout.addWidget(imgButton, 0, 1)
        self.sLGroupWid1.setLayout(Layout)

        self.sLGroupWid2 = QGroupBox('Process Options')

        Callibrate = QRadioButton('Callibrate')
        FPM = QRadioButton('FPM')
        full = QRadioButton('Full Process')
        procButton = QPushButton("Select")

        Layout = QGridLayout()
        Layout.addWidget(Callibrate, 0, 0)
        Layout.addWidget(FPM, 1, 0)
        Layout.addWidget(full, 2, 0)
        Layout.addWidget(procButton, 2, 1)
        self.sLGroupWid2.setLayout(Layout)


    def createProgressBar(self):
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 10000)
        self.progressBar.setValue(0)

        timer = QTimer(self)
        #prog = Signal()
        if __name__ == '__main__':
            timer.timeout.connect(self.advanceProgressBar)
            timer.start(1000)
        else:
            prog.connect(self.advanceProgressBar())



if __name__ == '__main__':
    app = QApplication(sys.argv)
    gallery = FPMGUI()
    gallery.show()
    sys.exit(app.exec_())