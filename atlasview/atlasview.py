"""
TopView is the main Widget with the related Controller Class
There are several SliceView windows (sagittal, coronal, possibly tilted etc...) that each have
a SliceController object
The underlying data model object is an ibllib.atlas.AllenAtlas object
"""
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QTransform
import pyqtgraph as pg
import matplotlib

from ibllib.atlas import AllenAtlas
import qt


class TopView(QtWidgets.QMainWindow):
    """
    Main Window of the application.
    This is a top view of the brain with 2 movable lines allowing to select sagittal and coronal
    slices.
    """
    @staticmethod
    def _instances():
        app = QtWidgets.QApplication.instance()
        return [w for w in app.topLevelWidgets() if isinstance(w, TopView)]

    @staticmethod
    def _get_or_create(title=None):
        av = next(filter(lambda e: e.isVisible() and e.windowTitle() == title,
                         TopView._instances()), None)
        if av is None:
            av = TopView()
            av.setWindowTitle(title)
        return av

    def __init__(self):
        super(TopView, self).__init__()
        self.ctrl = Controller(self)
        uic.loadUi(Path(__file__).parent.joinpath('topview.ui'), self)
        self.plotItem_topview.setAspectLocked(True)
        self.imageItem = pg.ImageItem()
        self.plotItem_topview.addItem(self.imageItem)
        # setup one horizontal and one vertical line that can be moved
        line_kwargs = {'movable': True, 'pen': pg.mkPen((0, 255, 0), width=3)}
        self.line_coronal = pg.InfiniteLine(angle=0, pos=0, **line_kwargs)
        self.line_sagittal = pg.InfiniteLine(angle=90, pos=0, **line_kwargs)
        self.line_coronal.sigDragged.connect(self.coronal_line_moved)  # sigPositionChangeFinished
        self.line_sagittal.sigDragged.connect(self.sagittal_line_moved)
        self.plotItem_topview.addItem(self.line_coronal)
        self.plotItem_topview.addItem(self.line_sagittal)
        # connect signals and slots
        s = self.plotItem_topview.getViewBox().scene()
        self.proxy = pg.SignalProxy(s.sigMouseMoved, rateLimit=60, slot=self.mouseMoveEvent)
        self.ctrl.set_top()

    def coronal_line_moved(self):
        self.ctrl.set_slice(self.ctrl.fig_coronal, self.line_coronal.value())

    def sagittal_line_moved(self):
        self.ctrl.set_slice(self.ctrl.fig_sagittal, self.line_sagittal.value())
        # self.ctrl.set_scatter(self.ctrl.fig_coronal, self.line_coronal.value())

    def mouseMoveEvent(self, scenepos):
        if isinstance(scenepos, tuple):
            scenepos = scenepos[0]
        else:
            return
        pass
        # qpoint = self.imageItem.mapFromScene(scenepos)

    def add_scatter_feature(self, data):
        self.ctrl.scatter_data = data / 1e6
        self.ctrl.scatter_data_ind = self.ctrl.atlas.bc.xyz2i(self.ctrl.scatter_data)
        self.ctrl.fig_coronal.add_scatter()
        self.ctrl.fig_sagittal.add_scatter()
        self.line_coronal.sigDragged.connect(
            lambda: self.ctrl.set_scatter(self.ctrl.fig_coronal, self.line_coronal.value()))
        self.line_sagittal.sigDragged.connect(
            lambda: self.ctrl.set_scatter(self.ctrl.fig_sagittal, self.line_sagittal.value()))
        self.ctrl.set_scatter(self.ctrl.fig_coronal)
        self.ctrl.set_scatter(self.ctrl.fig_sagittal)

    def add_image_feature(self, values, cmap, levels=None):
        self.ctrl.values = values
        # creat cmap look up table
        colormap = matplotlib.cm.get_cmap(cmap)
        colormap._init()
        self.ctrl.lut = (colormap._lut * 255).view(np.ndarray)
        self.ctrl.lut = np.insert(self.ctrl.lut, 0, [0, 0, 0, 0], axis=0)
        if levels is None:
            self.ctrl.levels = [np.min(values[np.nonzero(values)]), np.max(values)]
        else:
            self.ctrl.levels = [levels[0], levels[1]]

        self.ctrl.fig_coronal.add_image()
        self.ctrl.fig_sagittal.add_image()
        self.line_coronal.sigDragged.connect(
            lambda: self.ctrl.set_slice_with_value(self.ctrl.fig_coronal, self.line_coronal.value()))
        self.line_sagittal.sigDragged.connect(
            lambda: self.ctrl.set_slice_with_value(self.ctrl.fig_sagittal, self.line_sagittal.value()))
        self.ctrl.set_slice_with_value(self.ctrl.fig_coronal)
        self.ctrl.set_slice_with_value(self.ctrl.fig_sagittal)



class SliceView(QtWidgets.QWidget):
    """
    Window containing a volume slice
    """

    def __init__(self, topview: TopView, waxis, haxis, daxis):
        super(SliceView, self).__init__()
        self.topview = topview
        self.ctrl = SliceController(self, waxis, haxis, daxis)
        uic.loadUi(Path(__file__).parent.joinpath('sliceview.ui'), self)
        # init the image display
        self.plotItem_slice.setAspectLocked(True)
        self.imageItem = pg.ImageItem()
        self.imageItem2 = None
        self.plotItem_slice.addItem(self.imageItem)
        # connect signals and slots
        s = self.plotItem_slice.getViewBox().scene()
        self.proxy = pg.SignalProxy(s.sigMouseMoved, rateLimit=60, slot=self.mouseMoveEvent)
        s.sigMouseClicked.connect(self.mouseClick)
        # self.comboBox_layer.activated[str].connect(self.ctrl.set_layer)

    def closeEvent(self, event):
        self.destroy()

    def keyPressEvent(self, e):
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
        qpoint = self.imageItem.mapFromScene(scenepos)
        iw, ih, w, h, v, region = self.ctrl.cursor2xyamp(qpoint)
        self.label_x.setText(f"{w:.4f}")
        self.label_y.setText(f"{h:.4f}")
        self.label_ix.setText(f"{iw:.0f}")
        self.label_iy.setText(f"{ih:.0f}")
        self.label_v.setText(f"{v:.4f}")
        if region is None:
            self.label_region.setText("")
            self.label_acronym.setText("")
        else:
            self.label_region.setText(region['name'][0])
            self.label_acronym.setText(region['acronym'][0])

    def add_scatter(self):
        self.scatterItem = pg.ScatterPlotItem()
        self.plotItem_slice.addItem(self.scatterItem)

    def add_image(self):
        if self.imageItem2:
            self.plotItem_slice.removeItem(self.imageItem2)
        self.imageItem2 = pg.ImageItem()
        self.plotItem_slice.addItem(self.imageItem2)

class PgImageController:
    """
    Abstract class that implements mapping from axes to voxels for any window.
    Not instantiated directly.
    """
    def __init__(self, win, res=25):
        self.qwidget = win
        self.transform = None  # affine transform image indices 2 data domain

    def cursor2xyamp(self, qpoint):
        """Used for the mouse hover function over image display"""
        iw, ih = self.cursor2ind(qpoint)
        v = self.im[iw, ih]
        w, h, _ = np.matmul(self.transform, np.array([iw, ih, 1]))
        return iw, ih, w, h, v

    def cursor2ind(self, qpoint):
        """ image coordinates over the image display"""
        iw = np.max((0, np.min((int(np.floor(qpoint.x())), self.nw - 1))))
        ih = np.max((0, np.min((int(np.round(qpoint.y())), self.nh - 1))))
        return iw, ih

    def set_image(self, imItem, im, dw, dh, w0, h0, value=False, lut=None, levels=None):
        """
        :param im:
        :param dw:
        :param dh:
        :param w0:
        :param h0:
        :return:
        """
        self.im = im
        self.nw, self.nh = self.im.shape[0:2]
        if not value:
            imItem.setImage(self.im)
        else:
            imItem.setImage(self.im)
            imItem.setLookupTable(lut)
            imItem.setLevels((levels[0], levels[1]))
        transform = [dw, 0., 0., 0., dh, 0., w0, h0, 1.]
        self.transform = np.array(transform).reshape((3, 3)).T
        imItem.setTransform(QTransform(*transform))
        # self.view.plotItem.setLimits(xMin=wl[0], xMax=wl[1], yMin=hl[0], yMax=hl[1])

    def set_points(self, x=None, y=None):
        self.qwidget.scatterItem.setData(x=x, y=y, brush='b', size=5)


class Controller(PgImageController):
    """
    TopView Controller
    """
    def __init__(self, qmain: TopView, res=25):
        super(Controller, self).__init__(qmain)
        self.atlas = AllenAtlas(res)
        self.fig_top = self.qwidget = qmain
        # Setup Coronal slice: width: ml, height: dv, depth: ap
        self.fig_coronal = SliceView(qmain, waxis=0, haxis=2, daxis=1)
        self.fig_coronal.setWindowTitle('Coronal Slice')
        self.set_slice(self.fig_coronal)
        self.fig_coronal.show()
        # Setup Sagittal slice: width: ap, height: dv, depth: ml
        self.fig_sagittal = SliceView(qmain, waxis=1, haxis=2, daxis=0)
        self.fig_sagittal.setWindowTitle('Sagittal Slice')
        self.set_slice(self.fig_sagittal)
        self.fig_sagittal.show()

    def set_slice(self, fig, coord=0):
        waxis, haxis, daxis = (fig.ctrl.waxis, fig.ctrl.haxis, fig.ctrl.daxis)
        # construct the transform matrix image 2 ibl coordinates
        dw = self.atlas.bc.dxyz[waxis]
        dh = self.atlas.bc.dxyz[haxis]
        wl = self.atlas.bc.lim(waxis) - dw / 2
        hl = self.atlas.bc.lim(haxis) - dh / 2
        fig.ctrl.set_image(fig.imageItem,
                           self.atlas.slice(coord, axis=daxis, mode='clip'), dw, dh, wl[0], hl[0])
        fig.ctrl.slice_coord = coord

    def set_top(self):
        img = self.atlas.top.transpose()
        img[np.isnan(img)] = np.nanmin(img)  # img has dims ml, ap
        dw, dh = (self.atlas.bc.dxyz[0], self.atlas.bc.dxyz[1])
        wl, hl = (self.atlas.bc.xlim, self.atlas.bc.ylim)
        self.set_image(self.fig_top.imageItem, img, dw, dh, wl[0], hl[0])
        # self.qwidget.line_coronal.setData(x=wl, y=wl * 0, pen=pg.mkPen((0, 255, 0), width=3))
        # self.qwidget.line_sagittal.setData(x=hl * 0, y=hl, pen=pg.mkPen((0, 255, 0), width=3))

    def set_scatter(self, fig, coord=0):
        waxis = fig.ctrl.waxis
        # dealing with coronal slice
        if waxis == 0:
            idx = np.where(self.scatter_data_ind[:, 1] == self.atlas.bc.y2i(coord))[0]
            x = self.scatter_data[idx, 0]
            y = self.scatter_data[idx, 2]
        else:
            idx = np.where(self.scatter_data_ind[:, 0] == self.atlas.bc.x2i(coord))[0]
            x = self.scatter_data[idx, 1]
            y = self.scatter_data[idx, 2]

        fig.ctrl.set_points(x, y)

    def set_slice_with_value(self, fig, coord=0):
        waxis, haxis, daxis = (fig.ctrl.waxis, fig.ctrl.haxis, fig.ctrl.daxis)
        # construct the transform matrix image 2 ibl coordinates
        dw = self.atlas.bc.dxyz[waxis]
        dh = self.atlas.bc.dxyz[haxis]
        wl = self.atlas.bc.lim(waxis) - dw / 2
        hl = self.atlas.bc.lim(haxis) - dh / 2
        fig.ctrl.set_image(fig.imageItem2,
                           self.atlas.slice(coord, axis=daxis, volume='value', mode='clip',
                                            region_values=self.values), dw, dh, wl[0], hl[0],
                           value=True, lut=self.lut, levels=self.levels)
        fig.ctrl.slice_coord = coord


class SliceController(PgImageController):

    def __init__(self, fig, waxis=None, haxis=None, daxis=None):
        """
        :param waxis: brain atlas axis corresponding to display abscissa (coronal: 0, sagittal: 1)
        :param haxis: brain atlas axis corresponding to display ordinate (coronal: 2, sagittal: 2)
        :param daxis: brain atlas axis corresponding to display abscissa (coronal: 1, sagittal: 0)
        """
        super(SliceController, self).__init__(fig)
        self.waxis = waxis
        self.haxis = haxis
        self.daxis = daxis

    def cursor2xyamp(self, qpoint):
        """
        Extends the superclass method to also get the brain region from the model
        :param qpoint:
        :return:
        """
        iw, ih, w, h, v = super(SliceController, self).cursor2xyamp(qpoint)
        ba = self.qwidget.topview.ctrl.atlas
        xyz = np.zeros(3)
        xyz[np.array([self.waxis, self.haxis, self.daxis])] = [w, h, self.slice_coord]
        try:
            region = ba.regions.get(ba.get_labels(xyz))
        except ValueError:
            region = None
        return iw, ih, w, h, v, region


def view(res=25, title=None):
    """
    """
    qt.create_app()
    av = TopView._get_or_create(title=title)
    av.show()
    return av
