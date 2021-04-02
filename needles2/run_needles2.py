from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QTransform
import pyqtgraph as pg
import matplotlib

from ibllib.atlas import AllenAtlas, regions
import qt

from PyQt5 import Qt
import vedo
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from oneibl.one import ONE
#one = ONE()
from ibllib.pipes.histology import coverage
from needles2.needles_viewer import NeedlesViewer

class MainWindow(QtWidgets.QMainWindow):

    @staticmethod
    def _instances():
        app = QtWidgets.QApplication.instance()
        return [w for w in app.topLevelWidgets() if isinstance(w, MainWindow)]

    @staticmethod
    def _get_or_create(title=None, **kwargs):
        av = next(filter(lambda e: e.isVisible() and e.windowTitle() == title,
                         MainWindow._instances()), None)
        if av is None:
            av = MainWindow(**kwargs)
            av.setWindowTitle(title)
        return av

    def __init__(self):
        super(MainWindow, self).__init__()

        self.resize(1600, 800)
        self.setWindowTitle('Needles2')
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)

        self.atlas = AllenAtlas(25)
        main_layout = QtWidgets.QGridLayout()
        self.frame = Qt.QFrame()
        self.vl = Qt.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)
        self.plt = vedo.Plotter(qtWidget=self.vtkWidget, N=1)
        self.la = NeedlesViewer()
        self.la.initialize(self.plt)
        print(self.la)
        self.la2 = NeedlesViewer()
        self.la2.initialize(self.plt)
        self.frame.setLayout(self.vl)


        self.coronal = SliceView(self, self.atlas, waxis=0, haxis=2, daxis=1)
        self.sagittal = SliceView(self, self.atlas, waxis=1, haxis=2, daxis=0)
        self.top = TopView(self, self.atlas)

        menu_bar = QtWidgets.QMenuBar(self)
        menu_bar.setNativeMenuBar(False)
        self.setMenuBar(menu_bar)

        # Add menu bar for mappings
        self.map_menu = menu_bar.addMenu('Mappings')
        self.map_group = QtGui.QActionGroup(self.map_menu)
        # Only allow one to plot to be selected at any one time
        self.map_group.setExclusive(True)
        self.add_menu_bar(self.map_menu, self.map_group, list(self.atlas.regions.mappings.keys()),
                          callback=self.change_mapping)
#
        # Add menu bar for base image
        self.img_menu = menu_bar.addMenu('Images')
        self.img_group = QtGui.QActionGroup(self.img_menu)
        self.img_group.setExclusive(True)
        images = ['Image', 'Annotation']
        self.add_menu_bar(self.img_menu, self.img_group, images, callback=self.change_image)

        # Add menu bar for coverage
        self.coverage_menu = menu_bar.addMenu('Coverage')
        self.coverage_group = QtGui.QActionGroup(self.coverage_menu)
        self.coverage_group.setExclusive(True)
        self.add_menu_bar(self.img_menu, self.img_group, ['Coverage'], callback=self.change_image)

        self.region_list = QtGui.QStandardItemModel()
        self.region_combobox = QtWidgets.QComboBox()
        # Add line edit and completer to be able to search for subject
        self.region_combobox.setLineEdit(QtWidgets.QLineEdit())
        region_completer = QtWidgets.QCompleter()
        region_completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.region_combobox.setCompleter(region_completer)
        self.region_combobox.setModel(self.region_list)
        self.region_combobox.completer().setModel(self.region_list)
        self.region_combobox.activated.connect(self.on_region_selected)
        self.change_mapping()


        offset = self.atlas.bc.lim(0)[0]
        self.la.update_slicer(self.la.nx_slicer, self.top.line_sagittal.value()*1e6 + offset*1e6)

        main_layout.addWidget(self.region_combobox)
        main_layout.addWidget(self.top)
        main_layout.addWidget(self.coronal)
        main_layout.addWidget(self.sagittal)
        main_layout.addWidget(self.frame)
        main_widget.setLayout(main_layout)

        # initialise
        self.set_slice(self.coronal)
        self.set_slice(self.sagittal)


    def populate_lists(self, data, list_name, combobox):
        """
        Populate drop down lists with subject/session/alignment options
        :param data: list of options to add to widget
        :type data: 1D array of strings
        :param list_name: widget object to which to add data to
        :type list_name: QtGui.QStandardItemModel
        :param combobox: combobox object to which to add data to
        :type combobox: QtWidgets.QComboBox
        """
        list_name.clear()
        for dat in data:
            item = QtGui.QStandardItem(dat)
            item.setEditable(False)
            list_name.appendRow(item)

        # This makes sure the drop down menu is wide enough to show full length of string
        min_width = combobox.fontMetrics().width(max(data, key=len))
        min_width += combobox.view().autoScrollMargin()
        min_width += combobox.style().pixelMetric(QtGui.QStyle.PM_ScrollBarExtent)
        combobox.view().setMinimumWidth(min_width)

        # Set the default to be the first option
        combobox.setCurrentIndex(0)

    def change_mapping(self):
        data = np.unique(self.atlas.regions.name[self.atlas._get_mapping(
            mapping=self.get_mapping())])
        self.populate_lists(data, self.region_list, self.region_combobox)
        self._refresh()

    def get_mapping(self):
        return self.map_group.checkedAction().text()

    def change_image(self):
        self.coronal.ctrl.change_base_layer(self.coronal.fig_slice,
                                         self.img_group.checkedAction().text())
        self.sagittal.ctrl.change_base_layer(self.sagittal.fig_slice,
                                          self.img_group.checkedAction().text())
        self._refresh()

    def add_coverage(self):
        traj = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track')
        self.coverage = coverage(traj, self.atlas)
        self.coronal.ctrl.add_volume_layer(self.coronal.fig_slice, self.coverage)
        self.sagittal.ctrl.add_volume_layer(self.sagittal.fig_slice, self.coverage)
        self._refresh()


    def add_menu_bar(self, menu, group, items, callback=None, default=None):
        if not default:
            default = items[0]
        for item in items:
            if item == default:
                _item = QtGui.QAction(item, self, checkable=True, checked=True)
            else:
                _item = QtGui.QAction(item, self, checkable=True, checked=False)
            if callback:
                _item.triggered.connect(callback)
            menu.addAction(_item)
            group.addAction(_item)

        return menu, group

    def on_region_selected(self, idx):
        region_name = self.region_list.item(idx).text()
        region_idx = np.where(self.atlas.regions.name == region_name)[0][0]
        self._refresh_locked_region(self.atlas.regions.id[region_idx])

    def set_selected_region(self, region):
        region_idx = np.where(self.atlas.regions.id == region)[0][0]
        region_name = self.atlas.regions.name[region_idx]
        item = self.region_list.findItems(region_name)
        self.region_combobox.setCurrentIndex(self.region_list.indexFromItem(item[0]).row())

    def set_slice(self, fig, coord=0):
        waxis, haxis, daxis = (fig.ctrl.waxis, fig.ctrl.haxis, fig.ctrl.daxis)
        # construct the transform matrix image 2 ibl coordinates
        dw = self.atlas.bc.dxyz[waxis]
        dh = self.atlas.bc.dxyz[haxis]
        wl = self.atlas.bc.lim(waxis) - dw / 2
        hl = self.atlas.bc.lim(haxis) - dh / 2
        # the ImageLayer object carries slice kwargs and pyqtgraph ImageSet kwargs
        # reversed order so the self.im is set with the base layer
        #
        for layer in reversed(fig.ctrl.image_layers):
            _slice = self.atlas.slice(coord, axis=daxis, mapping=self.get_mapping(),
                                      **layer.slice_kwargs)
            fig.ctrl.set_image(layer.image_item, _slice, dw, dh, wl[0], hl[0], **layer.pg_kwargs)
        fig.ctrl.slice_coord = coord

    def _refresh(self):
        self._refresh_sagittal()
        self._refresh_coronal()

    def _refresh_coronal(self):
        self.set_slice(self.coronal, self.top.line_coronal.value())
        offset = self.atlas.bc.lim(1)[1]
        self.la.update_slicer(self.la.ny_slicer,  (self.top.line_coronal.value()*1e6 + offset*16))

    def _refresh_sagittal(self):
        self.set_slice(self.sagittal, self.top.line_sagittal.value())
        offset = self.atlas.bc.lim(0)[0]
        self.la.update_slicer(self.la.nx_slicer, self.top.line_sagittal.value()*1e6 + offset*1e6)

    def _refresh_locked_region(self, region):
        region_values = np.zeros_like(self.atlas.regions.id)
        # Set the coronal and sagittal mask layers to empty
        image = self.coronal.ctrl.get_image_layer(name='non_locked')
        image.slice_kwargs['region_values'] = region_values
        image = self.sagittal.ctrl.get_image_layer(name='non_locked')
        image.slice_kwargs['region_values'] = region_values

        region_idx = np.where(self.atlas.regions.id == region)[0]
        region_values[region_idx] = 100
        image = self.coronal.ctrl.get_image_layer(name='locked')
        image.slice_kwargs['region_values'] = region_values
        image = self.sagittal.ctrl.get_image_layer(name='locked')
        image.slice_kwargs['region_values'] = region_values
        self._refresh()
        self.set_selected_region(region)
        # need to set the drop down to the selected region
        self.la.reveal_regions(region_idx)

    def _reset_region(self, name):
        region_values = np.zeros_like(self.atlas.regions.id)
        image = self.coronal.ctrl.get_image_layer(name=name)
        image.slice_kwargs['region_values'] = region_values
        image = self.sagittal.ctrl.get_image_layer(name=name)
        image.slice_kwargs['region_values'] = region_values
        self._refresh()

    def _refresh_non_locked_region(self, region):
        region_values = np.zeros_like(self.atlas.regions.id)
        region_idx = np.where(self.atlas.regions.id == region)[0]
        region_values[region_idx] = 100
        image = self.coronal.ctrl.get_image_layer(name='non_locked')
        image.slice_kwargs['region_values'] = region_values
        image = self.sagittal.ctrl.get_image_layer(name='non_locked')
        image.slice_kwargs['region_values'] = region_values
        self._refresh()



class TopView(QtWidgets.QWidget):
    def __init__(self, qmain: MainWindow, atlas: AllenAtlas, **kwargs):
        self.qmain = qmain
        self.atlas = atlas
        super(TopView, self).__init__()
        self.ctrl = TopController(self, qmain, atlas, **kwargs)
        main_layout = QtWidgets.QGridLayout()

        self.fig_top = pg.PlotWidget()
        self.fig_top.setAspectLocked(True)
        self.ctrl.add_image_layer(self.fig_top, name='top')
        # self.fig_top.addItem(self.ctrl.imageItem)

        # setup one horizontal and one vertical line that can be moved
        line_kwargs = {'movable': True, 'pen': pg.mkPen((0, 255, 0), width=3)}
        self.line_coronal = pg.InfiniteLine(angle=0, pos=0, **line_kwargs)
        self.line_sagittal = pg.InfiniteLine(angle=90, pos=0, **line_kwargs)
        self.line_coronal.sigDragged.connect(self.qmain._refresh_coronal)  # sigPositionChangeFinished
        self.line_sagittal.sigDragged.connect(self.qmain._refresh_sagittal)
        self.fig_top.addItem(self.line_coronal)
        self.fig_top.addItem(self.line_sagittal)

        main_layout.addWidget(self.fig_top)
        self.setLayout(main_layout)

        self.ctrl.set_top()


class SliceView(QtWidgets.QWidget):
    """
    Window containing a volume slice
    """
    def __init__(self, qmain: MainWindow, atlas: AllenAtlas, waxis, haxis, daxis):
        super(SliceView, self).__init__()
        self.qmain = qmain
        self.atlas = atlas

        self.ctrl = SliceController(self, qmain, atlas, waxis, haxis, daxis)
        self.fig_slice = pg.PlotWidget()
        self.fig_slice.setAspectLocked(True)

        # self.ctrl.add_base_layer(self.fig_slice)
        self.ctrl.add_image_layer(self.fig_slice, slice_kwargs={'volume': 'image', 'mode': 'clip'},
                                  pg_kwargs={'opacity': 0.8})
        self.ctrl.add_mask_layer(self.fig_slice, name='locked', cmap='Blues')
        self.ctrl.add_mask_layer(self.fig_slice, name='non_locked', cmap='Greens')
        #self.ctrl.add_base_layer(self.fig_slice)

        main_layout = QtWidgets.QGridLayout()
        self.label_x = QtWidgets.QLabel('x')
        self.label_y = QtWidgets.QLabel('y')
        self.label_ix = QtWidgets.QLabel('ix')
        self.label_iy = QtWidgets.QLabel('iy')
        self.label_v = QtWidgets.QLabel('v')
        self.label_region = QtWidgets.QLabel('region')
        self.label_acronym = QtWidgets.QLabel('acronym')

        label_group = QtWidgets.QGroupBox()
        label_layout = QtWidgets.QVBoxLayout()
        label_layout.addWidget(self.label_x)
        label_layout.addWidget(self.label_y)
        label_layout.addWidget(self.label_ix)
        label_layout.addWidget(self.label_iy)
        label_layout.addWidget(self.label_v)
        #label_layout.addWidget(self.label_region)
        label_layout.addWidget(self.label_acronym)
        label_group.setLayout(label_layout)


        main_layout.addWidget(self.fig_slice, 0, 0)
        main_layout.addWidget(label_group, 0, 1)
        self.setLayout(main_layout)

        # init the image display
        # connect signals and slots
        s = self.fig_slice.scene()
        self.proxy = pg.SignalProxy(s.sigMouseMoved, rateLimit=60, slot=self.mouseMoveEvent)
        s.sigMouseClicked.connect(self.mouseClick)

    def closeEvent(self, event):
        self.destroy()

    def keyPressEvent(self, e):
        pass

    def mouseClick(self, event):
        if event.double():
            self.qmain._reset_region('locked')
        else:
            qpoint = self.ctrl.image_layers[0].image_item.mapFromScene(event.scenePos())
            iw, ih, w, h, v, region = self.ctrl.cursor2xyamp(qpoint)
            if region and region['acronym'][0] != 'void':
                self.qmain._refresh_locked_region(region['id'][0])

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
            self.qmain._reset_region('non_locked')
        else:
            #self.label_region.setText(region['name'][0])
            self.label_acronym.setText(region['acronym'][0])

            if region['acronym'][0] != 'void':
                self.qmain._refresh_non_locked_region(region['id'][0])
            else:
                self.qmain._reset_region('non_locked')


class BaseController:
    """
    Abstract class that implements mapping from axes to voxels for any window.
    Not instantiated directly.
    """
    def __init__(self, atlas: AllenAtlas):
        self.transform = None  # affine transform image indices 2 data domain
        self.image_layers = []
        self.atlas = atlas

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

    def get_image_layer(self, name=None):
        """Returns the either the first image item or the image item with specified name"""
        if not name:
            return self.image_layers[0]
        else:
            im_idx = np.where([im.name == name for im in self.image_layers])[0][0]
            return self.image_layers[im_idx]

    def set_image(self, image_item, im, dw, dh, w0, h0, **pg_kwargs):
        """
        :param im:
        :param dw:
        :param dh:
        :param w0:
        :param h0:
        :param pgkwargs: pg.ImageItem.setImage() parameters: level=None, lut=None, opacity=1
        :return:
        """
        self.im = im
        self.nw, self.nh = self.im.shape[0:2]
        image_item.setImage(self.im, **pg_kwargs)
        transform = [dw, 0., 0., 0., dh, 0., w0, h0, 1.]
        self.transform = np.array(transform).reshape((3, 3)).T
        image_item.setTransform(QTransform(*transform))

    def add_image_layer(self, fig, idx=None, **kwargs):
        """
        :param name: name of the image item to keep track of layers
        :param pg_kwargs: pyqtgraph setImage arguments: {'levels': None, 'lut': None,
        'opacity': 1.0}
        :param slice_kwargs: ibllib.atlas.slice arguments: {'volume': 'image', 'mode': 'clip'}
        :return:
        """
        il = ImageLayer(**kwargs)
        if idx is not None:
            self.image_layers.insert(idx, il)
        else:
            self.image_layers.append(il)
        fig.addItem(il.image_item)

    def add_mask_layer(self, fig, name='locked', cmap='Blues'):
        colormap = matplotlib.cm.get_cmap(cmap)
        colormap._init()
        # The last one is [0, 0, 0, 0] so remove this
        lut = (colormap._lut * 255).view(np.ndarray)[:-1]
        lut = np.insert(lut, 0, [0, 0, 0, 0], axis=0)
        self.add_image_layer(fig, name=name, pg_kwargs={'lut': lut, 'opacity': 1}, slice_kwargs={
            'volume': 'value', 'region_values': np.zeros_like(self.atlas.regions.id),
            'mode': 'clip'})

    def add_volume_layer(self, fig, volume, name='coverage', cmap='viridis'):
        colormap = matplotlib.cm.get_cmap(cmap)
        colormap._init()
        # The last one is [0, 0, 0, 0] so remove this
        lut = (colormap._lut * 255).view(np.ndarray)[:-1]
        lut = np.insert(lut, 0, [0, 0, 0, 0], axis=0)
        self.add_image_layer(fig, name=name, pg_kwargs={'lut': lut, 'opacity': 0.8}, slice_kwargs={
            'volume': 'volume', 'region_values': volume, 'mode': 'clip'})

    def change_base_layer(self, fig, image='Image'):
        base_layer = self.get_image_layer(name='base')
        if image == 'Image':
            base_layer.slice_kwargs = {'volume': 'image', 'mode': 'clip'}
        elif image == 'Annotation':
            base_layer.slice_kwargs = {'volume': 'annotation', 'mode': 'clip'}

    def remove_image_layer(self, fig, name):
        im_idx = np.where([im.name == name for im in self.image_layers])[0]
        if len(im_idx) != 0:
            il = self.image_layers.pop(im_idx[0])
            fig.removeItem(il.image_item)

class TopController(BaseController):
    """
    TopView ControllerTopView
    """
    def __init__(self, topview: TopView, qmain: MainWindow, atlas: AllenAtlas):
        super(TopController, self).__init__(atlas)
        self.atlas = atlas
        # self.fig_top = self.qwidget = qmain
        # Setup Coronal slice: width: ml, height: dv, depth: ap
        #print(qmain.atlas)

    def set_top(self):
        img = self.atlas.top.transpose()
        img[np.isnan(img)] = np.nanmin(img)  # img has dims ml, ap
        dw, dh = (self.atlas.bc.dxyz[0], self.atlas.bc.dxyz[1])
        wl, hl = (self.atlas.bc.xlim, self.atlas.bc.ylim)
        self.set_image(self.image_layers[0].image_item, img, dw, dh, wl[0], hl[0])


class SliceController(BaseController):

    def __init__(self, sliceview: SliceView, qmain: MainWindow, atlas: AllenAtlas,
                 waxis=None, haxis=None, daxis=None):
        """
        :param waxis: brain atlas axis corresponding to display abscissa (coronal: 0, sagittal: 1)
        :param haxis: brain atlas axis corresponding to display ordinate (coronal: 2, sagittal: 2)
        :param daxis: brain atlas axis corresponding to display abscissa (coronal: 1, sagittal: 0)
        """
        super(SliceController, self).__init__(atlas)
        self.waxis = waxis
        self.haxis = haxis
        self.daxis = daxis
        self.atlas = atlas
        self.qmain = qmain

    def cursor2xyamp(self, qpoint):
        """
        Extends the superclass method to also get the brain region from the model
        :param qpoint:
        :return:
        """
        iw, ih, w, h, v = super(SliceController, self).cursor2xyamp(qpoint)
        xyz = np.zeros(3)
        xyz[np.array([self.waxis, self.haxis, self.daxis])] = [w, h, self.slice_coord]
        try:
            region = self.atlas.regions.get(self.atlas.get_labels(xyz, mapping=self.qmain.get_mapping()))
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
    name: str = field(default='base')
    image_item: pg.ImageItem = field(default_factory=pg.ImageItem)
    pg_kwargs: dict = field(default_factory=lambda: {})
    slice_kwargs: dict = field(default_factory=lambda: {'volume': 'image', 'mode': 'clip'})


def view(title=None):
    """
    """
    qt.create_app()
    av = MainWindow._get_or_create(title=title)
    av.show()
    return av