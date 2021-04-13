from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import time

#from Distances import radial_distance
from gesture_classifier import *


class Ui_window1(object):
    def setupUi(self, window1):
        window1.setObjectName("window1")
        window1.resize(185, 130) # 820 530
        self.centralwidget = QtWidgets.QWidget(window1)
        self.centralwidget.setObjectName("centralwidget")
        window1.setCentralWidget(self.centralwidget)

        self.groupBox_2 = QtWidgets.QGroupBox("Your Gesture is", self.centralwidget)

        self.output_rd = QtWidgets.QTextBrowser(self.groupBox_2)
        self.output_rd.setGeometry(QtCore.QRect(10, 40, 150, 81))
        self.output_rd.setObjectName("output_rd")
        time.sleep(2)
        s=classify_from_webcam(clf)
        if not (s is None):
            self.output_rd.append(s)


        

        self.retranslateUi(window1)

        QtCore.QMetaObject.connectSlotsByName(window1)        

    def retranslateUi(self, window1):
            _translate = QtCore.QCoreApplication.translate
            window1.setWindowTitle(_translate("window1", "Gesture Detector"))


    
if __name__ == "__main__":
    global clf
    yes_fname = "data/gestures/yes_seqs.pkl"
    no_fname = "data/gestures/no_seqs.pkl"
    other_fname = "data/gestures/other_seqs.pkl"
    metric = M3
    delta = 10
    eps = 5
    n_neighbors = 5
    clf = KNNGestureClassifier(yes_fname, no_fname, other_fname, metric, delta, eps, n_neighbors)
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window1 = QtWidgets.QMainWindow()
    ui = Ui_window1()
    ui.setupUi(window1)                                                       # +

    window1.show()
    sys.exit(app.exec_())