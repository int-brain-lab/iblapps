from PyQt5 import QtCore, QtGui


class CustomEventWidget(QtGui.QWidget):

    keyPressed = QtCore.pyqtSignal(int)

    def __init__(self):
        super(CustomEventWidget, self).__init__()

    def keyPressEvent(self, event):
        self.keyPressed.emit(event.key())
