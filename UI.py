import numpy as np
from gesture_classifier import *
from PyQt5 import QtCore, QtGui, QtWidgets
import time


if __name__ == '__main__':
    import sys
    global clf
    yes_fname = "data/gestures/yes_seqs.pkl"
    no_fname = "data/gestures/no_seqs.pkl"
    other_fname = "data/gestures/other_seqs.pkl"
    metric = M3
    delta = 10
    eps = 5
    n_neighbors = 5
    clf = KNNGestureClassifier(yes_fname, no_fname, other_fname, metric, delta, eps, n_neighbors)
    
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Gesture Classifier")
    w = QtWidgets.QTextBrowser(maximumWidth=200, maximumHeight=120)
#     w.setStyleSheet('color: blue') 
    w.setFont(QtGui.QFont("Monospace"))
    w.setWordWrapMode(QtGui.QTextOption.NoWrap)
    w.show()
    s=classify_from_webcam(clf)
    w.setText(s)
    app.processEvents()
    for i in range(3):
        app.processEvents()
        s=classify_from_webcam(clf)
        w.setText(s)
    

    sys.exit(app.exec_())
