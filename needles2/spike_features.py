import alf.io
from brainbox import ephys_plots
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore,  uic
from PyQt5.QtGui import QTransform
import pyqtgraph as pg
from dataclasses import dataclass, field
from pathlib import Path
import atlaselectrophysiology.ColorBar as cb
import qt

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

DATAPATH = r'C:\Users\Mayo\Downloads\FlatIron\SpikeSorting'
spikeSorters = ['ks2', 'pyks2.5']
EID = '8ca1a850-26ef-42be-8b28-c2e2d12f06d6'

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
        self.view = SpikeSortView(self)
        mainWidget = QtWidgets.QWidget()
        self.setCentralWidget(mainWidget)
        mainLayout = QtWidgets.QGridLayout()

        menuBar = QtWidgets.QMenuBar(self)
        menuBar.setNativeMenuBar(False)
        self.setMenuBar(menuBar)

        plotOptions = menuBar.addMenu('Plots')
        plotOptionsGroup = QtGui.QActionGroup(plotOptions)
        plotOptionsGroup.setExclusive(True)
        plotFrRaster = QtGui.QAction('Firing Rate Raster Plot', self, checkable=True, checked=True)
        plotFrRaster.triggered.connect(self.view.ctrl.plotFrRaster)
        plotOptions.addAction(plotFrRaster)
        plotOptionsGroup.addAction(plotFrRaster)

        plotAmpRaster = QtGui.QAction('Amplitude Raster Plot', self, checkable=True, checked=False)
        plotAmpRaster.triggered.connect(self.view.ctrl.plotAmpRaster)
        plotOptions.addAction(plotAmpRaster)
        plotOptionsGroup.addAction(plotAmpRaster)

        plotAmpDepthFr = QtGui.QAction('Cluster Amp vs Depth vs FR', self, checkable=True,
                                       checked=False)
        plotAmpDepthFr.triggered.connect(self.view.ctrl.plotAmpDepthFr)
        plotOptions.addAction(plotAmpDepthFr)
        plotOptionsGroup.addAction(plotAmpDepthFr)

        for iw, wind in enumerate(self.view.spikeSortWindows):
            mainLayout.addWidget(wind.widget, 0, iw)

        mainWidget.setLayout(mainLayout)


# View is going to have the main figures
class SpikeSortView:
    def __init__(self, qmain: MainWindow):
        self.qmain = qmain
        self.spikeSortWindows = []

        # Add a figure window for each spikesorter
        for spikeSort in spikeSorters:
            widget, fig, fig_cb = self.makePlotWindow()
            self.spikeSortWindows.append(SpikeSortFigures(name=spikeSort, widget=widget,
                                                          fig=fig, cb=fig_cb))
        self.ctrl = SpikeSortController(self)

    def makePlotWindow(self):
        widget = pg.GraphicsLayoutWidget()
        layout = pg.GraphicsLayout()
        fig = pg.PlotItem()
        fig_cb = pg.PlotItem()
        fig_cb.setMouseEnabled(x=False, y=False)
        self.setAxis(fig_cb, 'bottom', show=False)
        self.setAxis(fig_cb, 'left', pen='w', label='blank')
        layout.addItem(fig_cb, 0, 0)
        layout.addItem(fig, 1, 0)
        layout.layout.setRowStretchFactor(0, 1)
        layout.layout.setRowStretchFactor(1, 10)
        widget.addItem(layout)
        return widget, fig, fig_cb

    def plotImage(self, fig, fig_cb, data):
        [fig.removeItem(item) for item in fig.items]
        [fig_cb.removeItem(item) for item in fig_cb.items]

        self.setAxis(fig_cb, 'top', pen='w')

        image = pg.ImageItem()
        image.setImage(data.data['c'])
        transform = [data.scale[0], 0., 0., 0., data.scale[1], 0., data.offset[0],
                     data.offset[1], 1.]
        image.setTransform(QtGui.QTransform(*transform))

        color_bar = cb.ColorBar(data.cmap)
        lut = color_bar.getColourMap()
        image.setLookupTable(lut)
        image.setLevels((data.clim[0], data.clim[1]))
        cbar = color_bar.makeColourBar(20, 5, fig_cb, min=data.clim[0],
                                       max=data.clim[1], label=data.labels['clabel'])
        fig_cb.addItem(cbar)
        fig.addItem(image)
        fig.setXRange(min=data.xlim[0], max=data.xlim[1], padding=0)
        fig.setYRange(min=data.ylim[0], max=data.ylim[1], padding=0)
        self.setAxis(fig, 'bottom', label=data.labels['xlabel'])
        self.setAxis(fig, 'left', label=data.labels['ylabel'])

    def plotScatter(self, fig, fig_cb, data):
        [fig.removeItem(item) for item in fig.items]
        [fig_cb.removeItem(item) for item in fig_cb.items]

        self.setAxis(fig_cb, 'top', pen='w')

        size = data.marker_size.tolist() if not isinstance(data.marker_size, int) else \
            data.marker_size
        symbol = data.marker_type.tolist() if not isinstance(data.marker_type, str) else \
            data.marker_type
        color_bar = cb.ColorBar(data.cmap)
        brush = color_bar.getBrush(data.data['c'], levels=[data.clim[0], data.clim[1]])
        cbar = color_bar.makeColourBar(20, 5, fig_cb, min=data.clim[0],
                                       max=data.clim[1], label=data.labels['clabel'])
        plot = pg.ScatterPlotItem()
        plot.setData(x=data.data['x'], y=data.data['y'],
                     symbol=symbol, size=size, brush=brush, pen=data.line_color)

        fig_cb.addItem(cbar)
        fig.addItem(plot)
        fig.setXRange(min=data.xlim[0], max=data.xlim[1], padding=0)
        fig.setYRange(min=data.ylim[0], max=data.ylim[1], padding=0)
        self.setAxis(fig, 'bottom', label=data.labels['xlabel'])
        self.setAxis(fig, 'left', label=data.labels['ylabel'])


    def setAxis(self, fig, ax, show=True, label=None, pen='k', ticks=True):
        """
        Show/hide and configure axis of figure
        :param fig: figure associated with axis
        :type fig: pyqtgraph PlotWidget
        :param ax: orientation of axis, must be one of 'left', 'right', 'top' or 'bottom'
        :type ax: string
        :param show: 'True' to show axis, 'False' to hide axis
        :type show: bool
        :param label: axis label
        :type label: string
        :parm pen: colour on axis
        :type pen: string
        :param ticks: 'True' to show axis ticks, 'False' to hide axis ticks
        :param ticks: bool
        :return axis: axis object
        :type axis: pyqtgraph AxisItem
        """
        if not label:
            label = ''
        if type(fig) == pg.PlotItem:
            axis = fig.getAxis(ax)
        else:
            axis = fig.plotItem.getAxis(ax)
        if show:
            axis.show()
            axis.setPen(pen)
            axis.setTextPen(pen)
            axis.setLabel(label)
            if not ticks:
                axis.setTicks([[(0, ''), (0.5, ''), (1, '')]])
        else:
            axis.hide()

        return axis


# Controller is going to change the plots etc
class SpikeSortController:
    def __init__(self, view: SpikeSortView):

        self.view = view
        self.model = SpikeSortModel(self, DATAPATH, EID)
        self.plotAmpDepthFr()

    def plotFrRaster(self):
        for spikeSort in spikeSorters:
            data = self.model.getFrRaster(spikeSort)
            window = self.get_spikeSorterWindow(spikeSort)
            self.view.plotImage(window.fig, window.cb, data)

    def plotAmpRaster(self):
        for spikeSort in spikeSorters:
            data = self.model.getAmpRaster(spikeSort)
            window = self.get_spikeSorterWindow(spikeSort)
            self.view.plotScatter(window.fig, window.cb, data)

    def plotAmpDepthFr(self):
        for spikeSort in spikeSorters:
            data = self.model.getAmpDepthFr(spikeSort)
            window = self.get_spikeSorterWindow(spikeSort)
            self.view.plotScatter(window.fig, window.cb, data)


    def get_spikeSorterRawData(self, name=None):
        """Returns the either the first image item or the image item with specified name"""
        if not name:
            return self.model.spikeSortRawData[0]
        else:
            idx = np.where([sp.name == name for sp in self.model.spikeSortRawData])[0][0]
            return self.model.spikeSortRawData[idx]

    def get_spikeSorterPlotData(self, name=None):
        """Returns the either the first image item or the image item with specified name"""
        if not name:
            return self.model.spikeSortPlotData[0]
        else:
            idx = np.where([sp.name == name for sp in self.model.spikeSortPlotData])[0][0]
            return self.model.spikeSortPlotData[idx]

    def get_spikeSorterWindow(self, name=None):
        """Returns the either the first image item or the image item with specified name"""
        if not name:
            return self.view.spikeSortWindows[0]
        else:
            idx = np.where([sp.name == name for sp in self.view.spikeSortWindows])[0][0]
            return self.view.spikeSortWindows[idx]


class SpikeSortModel:
    def __init__(self, ctrl: SpikeSortController, dataPath, eid):
        self.ctrl = ctrl

        self.spikeSortRawData = []
        self.spikeSortPlotData = []

        for spikeSort in spikeSorters:
            spikeSortPath = Path(dataPath).joinpath(spikeSort, eid, 'alf')
            spikes = alf.io.load_object(spikeSortPath, object='spikes')
            clusters = alf.io.load_object(spikeSortPath, object='clusters')
            channels = alf.io.load_object(spikeSortPath, object='channels')
            self.spikeSortRawData.append(SpikeSortRawData(name=spikeSort, spikes=spikes,
                                                          clusters=clusters, channels=channels))
            self.spikeSortPlotData.append(SpikeSortPlotData(name=spikeSort))

    def getFrRaster(self, spikeSort):
        plotData = self.ctrl.get_spikeSorterPlotData(name=spikeSort)
        if plotData.frRaster is None:
            rawData = self.ctrl.get_spikeSorterRawData(name=spikeSort)
            data = ephys_plots.image_fr_plot(rawData.spikes.depths, rawData.spikes.times,
                                             rawData.channels.localCoordinates)
            data.set_offset()
            data.set_scale()
            plotData.frRaster = data

        return plotData.frRaster

    def getAmpRaster(self, spikeSort):
        plotData = self.ctrl.get_spikeSorterPlotData(name=spikeSort)
        if plotData.ampRaster is None:
            rawData = self.ctrl.get_spikeSorterRawData(name=spikeSort)
            data = ephys_plots.scatter_raster_plot(rawData.spikes.amps, rawData.spikes.depths,
                                                   rawData.spikes.times)
            plotData.ampRaster = data

        return plotData.ampRaster

    def getAmpDepthFr(self, spikeSort):
        plotData = self.ctrl.get_spikeSorterPlotData(name=spikeSort)
        if plotData.ampDepthFr is None:
            rawData = self.ctrl.get_spikeSorterRawData(name=spikeSort)
            data = (ephys_plots.
                    scatter_amp_depth_fr_plot(rawData.spikes.amps, rawData.spikes.clusters,
                                              rawData.spikes.depths, rawData.spikes.times))
            plotData.ampDepthFr = data
            data.set_line_color('k')

        return plotData.ampDepthFr

    def 





# Model is going to store the alf data




@dataclass
class SpikeSortFigures:
    """
    Class for keeping track of image layers.
    :param image_item
    :param pg_kwargs: pyqtgraph setImage arguments: {'levels': None, 'lut': None, 'opacity': 1.0}
    :param slice_kwargs: ibllib.atlas.slice arguments: {'volume': 'image', 'mode': 'clip'}
    :param
    """
    name: str = field(default='base')
    widget: pg.GraphicsLayoutWidget = field(default_factory=pg.GraphicsLayoutWidget)
    fig: pg.PlotItem = field(default_factory=pg.PlotItem)
    cb: pg.PlotItem = field(default_factory=pg.PlotItem)


@dataclass
class SpikeSortRawData:
    """
    Class for keeping track of image layers.
    :param image_item
    :param pg_kwargs: pyqtgraph setImage arguments: {'levels': None, 'lut': None, 'opacity': 1.0}
    :param slice_kwargs: ibllib.atlas.slice arguments: {'volume': 'image', 'mode': 'clip'}
    :param
    """
    name: str = field(default='base')
    spikes: dict = field(default=dict)
    clusters: dict = field(default=dict)
    channels: dict = field(default=dict)


@dataclass
class SpikeSortPlotData:
    """
    Class for keeping track of image layers.
    :param image_item
    :param pg_kwargs: pyqtgraph setImage arguments: {'levels': None, 'lut': None, 'opacity': 1.0}
    :param slice_kwargs: ibllib.atlas.slice arguments: {'volume': 'image', 'mode': 'clip'}
    :param
    """
    name: str = field(default='base')
    frRaster: dict = field(default=None)
    ampRaster: dict = field(default=None)
    ampDepthFr: dict = field(default=None)




def view(title=None):
    """
    """
    qt.create_app()
    av = MainWindow._get_or_create(title=title)
    av.show()
    return av
