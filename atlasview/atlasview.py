"""
TopView is the main Widget with the related ControllerTopView Class
There are several SliceView windows (sagittal, coronal, possibly tilted etc...) that each have
a SliceController object
The underlying data model object is an ibllib.atlas.AllenAtlas object

    TopView(QMainWindow)
    ControllerTopView(PgImageController)

    SliceView(QWidget)
    SliceController(PgImageController)

"""
from dataclasses import dataclass, field
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
    def _get_or_create(title=None, **kwargs):
        av = next(filter(lambda e: e.isVisible() and e.windowTitle() == title,
                         TopView._instances()), None)
        if av is None:
            av = TopView(**kwargs)
            av.setWindowTitle(title)
        return av

    def __init__(self, **kwargs):
        super(TopView, self).__init__()
        self.ctrl = ControllerTopView(self, **kwargs)
        self.ctrl.image_layers = [ImageLayer()]
        uic.loadUi(Path(__file__).parent.joinpath('topview.ui'), self)
        self.plotItem_topview.setAspectLocked(True)
        self.plotItem_topview.addItem(self.ctrl.imageItem)
        # setup one horizontal and one vertical line that can be moved
        line_kwargs = {'movable': True, 'pen': pg.mkPen((0, 255, 0), width=3)}
        self.line_coronal = pg.InfiniteLine(angle=0, pos=0, **line_kwargs)
        self.line_sagittal = pg.InfiniteLine(angle=90, pos=0, **line_kwargs)
        self.line_coronal.sigDragged.connect(self._refresh_coronal)  # sigPositionChangeFinished
        self.line_sagittal.sigDragged.connect(self._refresh_sagittal)
        self.plotItem_topview.addItem(self.line_coronal)
        self.plotItem_topview.addItem(self.line_sagittal)
        # connect signals and slots: mouse moved
        s = self.plotItem_topview.getViewBox().scene()
        self.proxy = pg.SignalProxy(s.sigMouseMoved, rateLimit=60, slot=self.mouseMoveEvent)
        # combobox for the atlas remapping choices
        self.comboBox_mappings.addItems(self.ctrl.atlas.regions.mappings.keys())
        self.comboBox_mappings.currentIndexChanged.connect(self._refresh)
        # slider for transparency between image and labels
        self.slider_alpha.sliderMoved.connect(self.slider_alpha_move)
        self.ctrl.set_top()

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

    def add_image_layer(self, **kwargs):
        """
        :param pg_kwargs: pyqtgraph setImage arguments: {'levels': None, 'lut': None,
        'opacity': 1.0}
        :param slice_kwargs: ibllib.atlas.slice arguments: {'volume': 'image', 'mode': 'clip'}
        :return:
        """
        self.ctrl.fig_sagittal.add_image_layer(**kwargs)
        self.ctrl.fig_coronal.add_image_layer(**kwargs)

    def add_regions_feature(self, values, cmap, opacity=1.0):
        self.ctrl.values = values
        # creat cmap look up table
        colormap = matplotlib.cm.get_cmap(cmap)
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        lut = np.insert(lut, 0, [0, 0, 0, 0], axis=0)
        self.add_image_layer(pg_kwargs={'lut': lut, 'opacity': opacity}, slice_kwargs={
            'volume': 'value', 'region_values': values, 'mode': 'clip'})
        self._refresh()

    def slider_alpha_move(self):
        annotation_alpha = self.slider_alpha.value() / 100
        self.ctrl.fig_coronal.ctrl.image_layers[0].pg_kwargs['opacity'] = 1 - annotation_alpha
        self.ctrl.fig_sagittal.ctrl.image_layers[0].pg_kwargs['opacity'] = 1 - annotation_alpha
        self.ctrl.fig_coronal.ctrl.image_layers[1].pg_kwargs['opacity'] = annotation_alpha
        self.ctrl.fig_sagittal.ctrl.image_layers[1].pg_kwargs['opacity'] = annotation_alpha
        self._refresh()

    def mouseMoveEvent(self, scenepos):
        if isinstance(scenepos, tuple):
            scenepos = scenepos[0]
        else:
            return
        pass
        # qpoint = self.imageItem.mapFromScene(scenepos)

    def _refresh(self):
        self._refresh_sagittal()
        self._refresh_coronal()

    def _refresh_coronal(self):
        self.ctrl.set_slice(self.ctrl.fig_coronal, self.line_coronal.value(),
                            mapping=self.comboBox_mappings.currentText())

    def _refresh_sagittal(self):
        self.ctrl.set_slice(self.ctrl.fig_sagittal, self.line_sagittal.value(),
                            mapping=self.comboBox_mappings.currentText())


class SliceView(QtWidgets.QWidget):
    """
    Window containing a volume slice
    """

    def __init__(self, topview: TopView, waxis, haxis, daxis):
        super(SliceView, self).__init__()
        self.topview = topview
        self.ctrl = SliceController(self, waxis, haxis, daxis)
        uic.loadUi(Path(__file__).parent.joinpath('sliceview.ui'), self)
        self.add_image_layer(slice_kwargs={'volume': 'image', 'mode': 'clip'},
                             pg_kwargs={'opacity': 0.8})
        self.add_image_layer(slice_kwargs={'volume': 'annotation', 'mode': 'clip'},
                             pg_kwargs={'opacity': 0.2})
        # init the image display
        self.plotItem_slice.setAspectLocked(True)
        # connect signals and slots
        s = self.plotItem_slice.getViewBox().scene()
        self.proxy = pg.SignalProxy(s.sigMouseMoved, rateLimit=60, slot=self.mouseMoveEvent)
        s.sigMouseClicked.connect(self.mouseClick)

    def add_scatter(self):
        self.scatterItem = pg.ScatterPlotItem()
        self.plotItem_slice.addItem(self.scatterItem)

    def add_image_layer(self, **kwargs):
        """
        :param pg_kwargs: pyqtgraph setImage arguments: {'levels': None, 'lut': None,
        'opacity': 1.0}
        :param slice_kwargs: ibllib.atlas.slice arguments: {'volume': 'image', 'mode': 'clip'}
        :return:
        """
        il = ImageLayer(**kwargs)
        self.ctrl.image_layers.append(il)
        self.plotItem_slice.addItem(il.image_item)

    def closeEvent(self, event):
        self.destroy()

    def keyPressEvent(self, e):
        pass

    def mouseClick(self, event):
        if not event.double():
            return

    def mouseMoveEvent(self, scenepos):
        if isinstance(scenepos, tuple):
            scenepos = scenepos[0]
        else:
            return
        qpoint = self.ctrl.image_layers[0].image_item.mapFromScene(scenepos)
        iw, ih, w, h, v, region = self.ctrl.cursor2xyamp(qpoint)
        self.label_x.setText(f"{w:.4f}")
        self.label_y.setText(f"{h:.4f}")
        self.label_ix.setText(f"{iw:.0f}")
        self.label_iy.setText(f"{ih:.0f}")
        if isinstance(v, np.ndarray):
            self.label_v.setText(str(v))
        else:
            self.label_v.setText(f"{v:.4f}")
        if region is None:
            self.label_region.setText("")
            self.label_acronym.setText("")
        else:
            self.label_region.setText(region['name'][0])
            self.label_acronym.setText(region['acronym'][0])

    def replace_image_layer(self, index, **kwargs):
        if index and len(self.imageItem) >= index:
            il = self.image_layers.pop(index)
            self.plotItem_slice.removeItem(il.image_item)
        self.add_image_layer(**kwargs)


class PgImageController:
    """
    Abstract class that implements mapping fr`om axes to voxels for any window.
    Not instantiated directly.
    """
    def __init__(self, win, res=25):
        self.qwidget = win
        self.transform = None  # affine transform image indices 2 data domain
        self.image_layers = []

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

    @property
    def imageItem(self):
        """returns the first image item"""
        return self.image_layers[0].image_item

    def set_image(self, pg_image_item, im, dw, dh, w0, h0, **pg_kwargs):
        """
        :param im:
        :param dw:
        :param dh:
        :param w0:
        :param h0:
        :param pgkwargs: og.ImageItem.setImage() parameters: level=None, lut=None, opacity=1
        :return:
        """
        self.im = im
        self.nw, self.nh = self.im.shape[0:2]
        pg_image_item.setImage(self.im, **pg_kwargs)
        transform = [dw, 0., 0., 0., dh, 0., w0, h0, 1.]
        self.transform = np.array(transform).reshape((3, 3)).T
        pg_image_item.setTransform(QTransform(*transform))

    def set_points(self, x=None, y=None):
        # at the moment brush and size are fixed! These need to be arguments
        # For the colour need to convert the colour to QtGui.QColor
        self.qwidget.scatterItem.setData(x=x, y=y, brush='b', size=5)


class ControllerTopView(PgImageController):
    """
    TopView ControllerTopView
    """
    def __init__(self, qmain: TopView, res: int = 25, volume='image', brainmap='Allen'):
        super(ControllerTopView, self).__init__(qmain)
        self.volume = volume
        self.atlas = AllenAtlas(res, brainmap=brainmap)
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

    def set_slice(self, fig, coord=0, mapping="Allen"):
        waxis, haxis, daxis = (fig.ctrl.waxis, fig.ctrl.haxis, fig.ctrl.daxis)
        # construct the transform matrix image 2 ibl coordinates
        dw = self.atlas.bc.dxyz[waxis]
        dh = self.atlas.bc.dxyz[haxis]
        wl = self.atlas.bc.lim(waxis) - dw / 2
        hl = self.atlas.bc.lim(haxis) - dh / 2
        # the ImageLayer object carries slice kwargs and pyqtgraph ImageSet kwargs
        # reversed order so the self.im is set with the base layer
        for layer in reversed(fig.ctrl.image_layers):
            _slice = self.atlas.slice(coord, axis=daxis, mapping=mapping, **layer.slice_kwargs)
            fig.ctrl.set_image(layer.image_item, _slice, dw, dh, wl[0], hl[0], **layer.pg_kwargs)
        fig.ctrl.slice_coord = coord

    def set_top(self):
        img = self.atlas.top.transpose()
        img[np.isnan(img)] = np.nanmin(img)  # img has dims ml, ap
        dw, dh = (self.atlas.bc.dxyz[0], self.atlas.bc.dxyz[1])
        wl, hl = (self.atlas.bc.xlim, self.atlas.bc.ylim)
        self.set_image(self.image_layers[0].image_item, img, dw, dh, wl[0], hl[0])

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

    def set_volume(self, volume):
        self.volume = volume


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
        mapping = self.qwidget.topview.comboBox_mappings.currentText()
        try:
            region = ba.regions.get(ba.get_labels(xyz, mapping=mapping))
        except ValueError:
            region = None
        return iw, ih, w, h, v, region


@dataclass
class ImageLayer:
    """
    Class for keeping track of image layers.
    :param image_item
    :param pg_kwargs: pyqtgraph setImage arguments: {'levels': None, 'lut': None, 'opacity': 1.0}
    :param slice_kwargs: ibllib.atlas.slice arguments: {'volume': 'image', 'mode': 'clip'}
    :param
    """
    image_item: pg.ImageItem = field(default_factory=pg.ImageItem)
    pg_kwargs: dict = field(default_factory=lambda: {})
    slice_kwargs: dict = field(default_factory=lambda: {'volume': 'image', 'mode': 'clip'})


def view(res=25, title=None, brainmap='Allen'):
    """
    """
    qt.create_app()
    av = TopView._get_or_create(title=title, res=res, brainmap=brainmap)
    av.show()
    return av
