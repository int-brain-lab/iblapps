from PyQt5 import QtWidgets, QtCore, QtGui, uic

from easyqc.gui import viewseis
from iblapps import qt
from pathlib import Path

import numpy as np

import pyqtgraph as pg


def viewephys(data, fs, channels=None, br=None, title='ephys'):
    """

    :param data: [nc, ns]
    :param fs:
    :param channels:
    :param br:
    :param title:
    :return:
    """
    width = 40
    height = 800
    nc, ns = data.shape
    # ih = np.linspace(0, nc - 1, height).astype(np.int32)
    # image = br.rgb[channels['ibr'][ih]].astype(np.uint8)
    # image = np.tile(image[:, np.newaxis, :], (1, width, 1))
    # image = np.tile(image[np.newaxis, :, :], (width, 1, 1))
    from ibllib.ephys.neuropixel import trace_header
    if channels is None:
        channels = trace_header(version = 1)
        eqc = viewseis(data.T, si=1 / fs * 1e3, h=channels, title=title, taxis=0)
        return eqc
    else:
        image = br.rgb[channels['ibr']].astype(np.uint8)
        image = image[np.newaxis, :, :]


    eqc = viewseis(data.T, si=1 / fs * 1e3, h=channels, title=title, taxis=0)
    imitem = pg.ImageItem(image)
    eqc.plotItem_header_v.addItem(imitem)
    transform = [1, 0, 0, 0, 1, 0, -0.5, 0, 1.]
    imitem.setTransform(QtGui.QTransform(*transform))
    eqc.plotItem_header_v.setLimits(xMin=-.5, xMax=.5)
    # eqc.comboBox_header.setVisible(False)

    return eqc


COLOR_PLOTS = (pg.mkColor((31, 119, 180)),)


def view2p(tiff_file, title=None):
    qt.create_app()
    v2p = View2p._get_or_create(title=title)
    v2p.update_tiff(tiff_file)
    v2p.show()
    return v2p


class View2p(QtWidgets.QMainWindow):
    """
    This is the view in the MVC approach
    """
    layers = None  # used for additional scatter layers

    @staticmethod
    def _instances():
        app = QtWidgets.QApplication.instance()
        return [w for w in app.topLevelWidgets() if isinstance(w, View2p)]

    @staticmethod
    def _get_or_create(title=None):
        v2p = next(filter(lambda e: e.isVisible() and e.windowTitle() == title,
                          View2p._instances()), None)
        if v2p is None:
            v2p = View2p()
            v2p.setWindowTitle(title)
        return v2p

    def __init__(self, *args, **kwargs):
        super(View2p, self).__init__(*args, **kwargs)
        # wave by Diana Militano from the Noun Projectp
        uic.loadUi(Path(__file__).parent.joinpath('view2p.ui'), self)

    def update_tiff(self, tiff_file):
        pass