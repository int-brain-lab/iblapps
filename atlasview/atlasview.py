import sys  # We need sys so that we can pass argv to QApplication
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtGui import QTransform
import pyqtgraph as pg

from ibllib.atlas import AllenAtlas
import qt


class AtlasView(QtWidgets.QMainWindow):

    @staticmethod
    def _instances():
        app = QtWidgets.QApplication.instance()
        return [w for w in app.topLevelWidgets() if isinstance(w, AtlasView)]

    @staticmethod
    def _get_or_create(title=None):
        av = next(filter(lambda e: e.isVisible() and e.windowTitle() == title,
                      AtlasView._instances()), None)
        if av is None:
            av = AtlasView()
            av.setWindowTitle(title)
        return av

    def __init__(self, *args, **kwargs):
        super(AtlasView, self).__init__(*args, **kwargs)
        self.ctrl = Controller(self)
        uic.loadUi(Path(__file__).parent.joinpath('atlasview.ui'), self)
        # init the seismic density display
        self.plotItem_slice.setAspectLocked(True)
        # self.plotItem_slice.invertY()
        self.imageItem_slice = pg.ImageItem()
        self.plotItem_slice.addItem(self.imageItem_slice)
        # connect signals and slots
        s = self.plotItem_slice.getViewBox().scene()
        self.proxy = pg.SignalProxy(s.sigMouseMoved, rateLimit=60, slot=self.mouseMoveEvent)
        s.sigMouseClicked.connect(self.mouseClick)
        # self.comboBox_layer.activated[str].connect(self.ctrl.set_layer)
    """
    View Methods
    """
    def closeEvent(self, event):
        self.destroy()

    def keyPressEvent(self, e):
        """
        """
        pass

    def mouseClick(self, event):
        if not event.double():
            return
        qxy = self.imageItem_seismic.mapFromScene(event.scenePos())
        tr, s = (qxy.x(), qxy.y())
        print(tr, s)

    def mouseMoveEvent(self, scenepos):
        if isinstance(scenepos, tuple):
            scenepos = scenepos[0]
        else:
            return
        qpoint = self.imageItem_slice.mapFromScene(scenepos)
        iw, _, v, w, h = self.ctrl.cursor2timetraceamp(qpoint)
        self.label_x.setText(f"{iw:.0f}")
        self.label_t.setText(f"{h:.4f}")
        self.label_amp.setText(f"{v:2.2E}")
        self.label_h.setText(f"{w:.4f}")


class Controller:

    def __init__(self, view, res=25):
        self.view = view
        self.atlas = AllenAtlas(res)
        self.transform = None  # affine transform image indices 2 data domain

    def cursor2timetraceamp(self, qpoint):
        """Used for the mouse hover function over image display"""
        iw, ih = self.cursor2ind(qpoint)
        v = self.im[iw, ih]
        w, h, _ = np.matmul(self.transform, np.array([iw, ih, 1]))
        return iw, ih, v, w, h

    def cursor2ind(self, qpoint):
        """ image coordinates over the image display"""
        iw = np.max((0, np.min((int(np.floor(qpoint.x())), self.nw - 1))))
        ih = np.max((0, np.min((int(np.round(qpoint.y())), self.nh - 1))))
        return iw, ih

    def set_slice(self, ap=0):
        """
        data is a 2d array [ntr, nsamples]
        if 3d the first dimensions are merged in ntr and the last is nsamples
        update_data(self, data=None, h=0.002, gain=None)
        """
        ## coronal slice
        daxis = 1  # depth axis is ap
        waxis = 0  # width axis is ml
        haxis = 2  # height axis is dv
        self.im = self.atlas.slice(ap, axis=daxis)
        self.nw, self.nh = self.im.shape
        self.view.imageItem_slice.setImage(self.im)

        # construct the transform matrix image 2 ibl coordinates
        dw = self.atlas.bc.dxyz[waxis]
        dh = self.atlas.bc.dxyz[haxis]
        wl = self.atlas.bc.lim(waxis) - dw / 2
        hl = self.atlas.bc.lim(haxis) - dh / 2

        transform = [dw, 0., 0., 0., dh, 0., wl[0], hl[0], 1.]
        self.transform = np.array(transform).reshape((3, 3)).T
        self.view.imageItem_slice.setTransform(QTransform(*transform))
        # self.view.plotItem_slice.setLimits(xMin=wl[0], xMax=wl[1], yMin=hl[0], yMax=hl[1])


def viewatlas(res=25, title=None):
    """
    """
    qt.create_app()
    av = AtlasView._get_or_create(title=title)
    if av is not None:
        av.ctrl.set_slice()
    av.show()
    return av

