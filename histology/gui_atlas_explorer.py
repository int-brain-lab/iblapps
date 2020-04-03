import sys  # We need sys so that we can pass argv to QApplication

import numpy as np
from PyQt5 import QtWidgets, QtCore, uic
import pyqtgraph as pg

import ibllib.atlas as atlas
from iblapps import qt

ba = atlas.AllenAtlas(25)


class WinAtlas(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(WinAtlas, self).__init__(*args, **kwargs)
        uic.loadUi('gui_atlas_explorer.ui', self)
        self.plotItem_atlas = self.GraphicsLayoutWidget_atlas.addPlot()
        self.plotItem_atlas.setAspectLocked(False)
        self.plotItem_atlas.invertY()

        self.imageItem_atlas = pg.ImageItem(border=(1, 1, 1))
        self.plotItem_atlas.addItem(self.imageItem_atlas)

        self.update()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_PageUp:
            self.gain_up()
        if event.key() == QtCore.Qt.Key_PageDown:
            self.gain_down()

    def mouseMoveEvent(self, event):
        print('move')
        pass

    def update(self, im=None, sr=0.002):
        if im is None:
            im = ba.image[ba.bc.y2i(0), :, :]
        self.imageItem_atlas.setImage(im)
        self.plotItem_atlas.setLimits(xMin=0, xMax=im.shape[0], yMin=0, yMax=im.shape[1])
        self.plotItem_atlas.getAxis('left').setScale(0.002)

    def gain_up(self):
        self.imageItem_atlas.setLevels(self.imageItem_atlas.levels / np.sqrt(2))

    def gain_down(self):
        self.imageItem_atlas.setLevels(self.imageItem_atlas.levels * np.sqrt(2))


def viewatlas(w=None, title=None):
    app = qt.create_app()
    eqc = WinAtlas()
    eqc.setWindowTitle(title)
    eqc.show()
    if w is not None:
        eqc.update(w)
    return eqc


if __name__ == "__main__":
    w = viewatlas(title="Allen Atlas - IBL")
    qt.run_app()