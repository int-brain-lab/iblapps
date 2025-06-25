from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.exporters
from qt_helpers import qt
import atlaselectrophysiology.qt_utils.ColorBar as cb

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

import alf.io
from brainbox import ephys_plots
from brainbox.plot_base import LinePlot
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

DATAPATH = r'C:\Users\Mayo\Downloads\FlatIron\SpikeSorting'
spikeSorters = ['ks2', 'pyks2.5', 'ks3']

PLOT_DICT = {'Firing Rate Raster Plot': 'frRaster',
             'Amplitude Raster Plot': 'ampRaster',
             'Cluster Raster Plot': 'clustRaster',
             'Cluster Amp vs Depth vs FR': 'ampDepthFr',
             'Cluster count': 'clustCount'}


# TODO seperate windows, describe plots

class MainWindow(QtWidgets.QMainWindow):

    @staticmethod
    def _instances():
        app = QtWidgets.QApplication.instance()
        return [w for w in app.topLevelWidgets() if isinstance(w, MainWindow)]

    @staticmethod
    def _get_or_create(eid, title=None, **kwargs):
        av = next(filter(lambda e: e.isVisible() and e.windowTitle() == title,
                         MainWindow._instances()), None)
        if av is None:
            av = MainWindow(eid, **kwargs)
            av.setWindowTitle(title)
        return av

    def __init__(self, eid):
        super(MainWindow, self).__init__()
        self.resize(1650, 700)
        self.eid = eid
        self.view = SpikeSortView(self, eid)
        mainWidget = QtWidgets.QWidget()
        self.setCentralWidget(mainWidget)
        mainLayout = QtWidgets.QGridLayout()

        menuBar = QtWidgets.QMenuBar(self)
        menuBar.setNativeMenuBar(False)
        self.setMenuBar(menuBar)

        plotOptions = menuBar.addMenu('Plots')
        self.plotOptionsGroup = QtGui.QActionGroup(plotOptions)
        self.plotOptionsGroup.setExclusive(True)
        plotFrRaster = QtGui.QAction('Firing Rate Raster Plot', self, checkable=True, checked=True)
        plotFrRaster.triggered.connect(self.view.ctrl.plotFrRaster)
        plotOptions.addAction(plotFrRaster)
        self.plotOptionsGroup.addAction(plotFrRaster)

        plotClustRaster = QtGui.QAction('Cluster Raster Plot', self, checkable=True, checked=False)
        plotClustRaster.triggered.connect(self.view.ctrl.plotClusterRaster)
        plotOptions.addAction(plotClustRaster)
        self.plotOptionsGroup.addAction(plotClustRaster)

        plotAmpRaster = QtGui.QAction('Amplitude Raster Plot', self, checkable=True, checked=False)
        plotAmpRaster.triggered.connect(self.view.ctrl.plotAmpRaster)
        plotOptions.addAction(plotAmpRaster)
        self.plotOptionsGroup.addAction(plotAmpRaster)

        plotAmpDepthFr = QtGui.QAction('Cluster Amp vs Depth vs FR', self, checkable=True,
                                       checked=False)
        plotAmpDepthFr.triggered.connect(self.view.ctrl.plotAmpDepthFr)
        plotOptions.addAction(plotAmpDepthFr)
        self.plotOptionsGroup.addAction(plotAmpDepthFr)

        plotclustCount = QtGui.QAction('Cluster count', self, checkable=True,
                                       checked=False)
        plotclustCount.triggered.connect(self.view.ctrl.plotClusterCount)
        plotOptions.addAction(plotclustCount)
        self.plotOptionsGroup.addAction(plotclustCount)

        for iw, wind in enumerate(self.view.spikeSortWindows):
            mainLayout.addWidget(wind.widget, 0, iw)

        mainWidget.setLayout(mainLayout)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_N:
            self.view.normalisePlots()
        if event.key() == QtCore.Qt.Key_U:
            self.view.unNormalisePlots()
        if event.key() == QtCore.Qt.Key_P:
            self.view.printPlots()

# ONE DAY
#class SpikeSortWidget(QtWidgets.QWidget):
#    def __init__(self, qmain: MainWindow, eid: str):


# View is going to have the main figures
class SpikeSortView:
    def __init__(self, qmain: MainWindow, eid: str):
        self.qmain = qmain
        self.eid = eid
        self.spikeSortWindows = []

        # Add a figure window for each spikesorter
        for iS, spikeSort in enumerate(spikeSorters):
            widget, fig, fig_cb = self.makePlotWindow()
            if iS == 0:
                vb = fig.getViewBox()
            else:
                fig.getViewBox().setXLink(vb)
                fig.getViewBox().setYLink(vb)

            self.spikeSortWindows.append(SpikeSortFigures(name=spikeSort, widget=widget,
                                                          fig=fig, cb=fig_cb))
        self.ctrl = SpikeSortController(self, eid)

    def makePlotWindow(self):
        widget = pg.GraphicsLayoutWidget()
        layout = pg.GraphicsLayout()
        fig = pg.PlotItem()
        fig_cb = pg.PlotItem()
        fig_cb.setMouseEnabled(x=False, y=False)
        fig.getViewBox().setAspectLocked(False)
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
        fig.setLimits(xMin=data.xlim[0], xMax=data.xlim[1], yMin=data.ylim[0], yMax=data.ylim[1])
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
        fig.setLimits(xMin=data.xlim[0], xMax=data.xlim[1], yMin=data.ylim[0], yMax=data.ylim[1])
        fig.setXRange(min=data.xlim[0], max=data.xlim[1], padding=0)
        fig.setYRange(min=data.ylim[0], max=data.ylim[1], padding=0)
        self.setAxis(fig, 'bottom', label=data.labels['xlabel'])
        self.setAxis(fig, 'left', label=data.labels['ylabel'])

    def plotBar(self, fig, fig_cb, data):
        [fig.removeItem(item) for item in fig.items]
        [fig_cb.removeItem(item) for item in fig_cb.items]

        self.setAxis(fig_cb, 'top', pen='w')

        plot = pg.BarGraphItem(x=data.data['x'], height=data.data['y'], width=0.6, brush='r')
        fig.addItem(plot)
        fig.setLimits(xMin=data.xlim[0], xMax=data.xlim[1], yMin=data.ylim[0], yMax=data.ylim[1])
        fig.setXRange(min=data.xlim[0], max=data.xlim[1], padding=0)
        fig.setYRange(min=data.ylim[0], max=data.ylim[1], padding=0)
        self.setAxis(fig, 'bottom', label=data.labels['xlabel'])
        self.setAxis(fig, 'left', label=data.labels['ylabel'])

        # For bar plot also add a label
        fig_cb.setTitle(f'N clusters = {np.max(data.data["x"])}, N spikes = {np.sum(data.data["y"])}')


    def setAxis(self, fig, ax, show=True, label=None, pen='k', fontsize=12, ticks=True):
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
            labelStyle = {'font-size': f'{fontsize}pt'}
            axis.setLabel(label, **labelStyle)
            font = QtGui.QFont()
            font.setPixelSize(fontsize)
            axis.setTickFont(font)
            axis.setPen(pen)
            axis.setTextPen(pen)

            if not ticks:
                axis.setTicks([[(0, ''), (0.5, ''), (1, '')]])
        else:
            axis.hide()

        return axis

    def getPlotType(self):
        return self.qmain.plotOptionsGroup.checkedAction()


    def normalisePlots(self):
        selectedPlot = self.getPlotType()
        self.ctrl.model.normaliseAcrossSorters(PLOT_DICT[selectedPlot.text()])
        selectedPlot.trigger()

    def unNormalisePlots(self):
        selectedPlot = self.getPlotType()
        self.ctrl.model.unNormaliseAcrossSorters(PLOT_DICT[selectedPlot.text()])
        selectedPlot.trigger()

    def printPlots(self):
        image_path = Path(DATAPATH).joinpath('figures')
        image_path.mkdir(exist_ok=True, parents=True)
        for spikeSort in self.spikeSortWindows:
            exporter = pg.exporters.ImageExporter(spikeSort.widget.scene())
            exporter.export(str(image_path.joinpath(self.eid + '_' + spikeSort.name + '_' +
                                                self.getPlotType().text() + '.png')))


# Controller is going to change the plots etc
class SpikeSortController:
    def __init__(self, view: SpikeSortView, eid: str):

        self.view = view
        self.model = SpikeSortModel(self, DATAPATH, eid)
        self.plotFrRaster()

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

    def plotClusterRaster(self):
        for spikeSort in spikeSorters:
            data = self.model.getClusterRaster(spikeSort)
            window = self.get_spikeSorterWindow(spikeSort)
            self.view.plotScatter(window.fig, window.cb, data)

    def plotClusterCount(self):
        for spikeSort in spikeSorters:
            data = self.model.getClusterCount(spikeSort)
            window = self.get_spikeSorterWindow(spikeSort)
            self.view.plotBar(window.fig, window.cb, data)


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
            spikes = self.removeNans(spikes)
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
                                             rawData.channels.localCoordinates,  t_bin=0.2)
            data.set_offset()
            data.set_scale()
            self.setStartingRange(data)
            plotData.frRaster = data

        return plotData.frRaster

    def getAmpRaster(self, spikeSort):
        plotData = self.ctrl.get_spikeSorterPlotData(name=spikeSort)
        if plotData.ampRaster is None:
            rawData = self.ctrl.get_spikeSorterRawData(name=spikeSort)
            data = ephys_plots.scatter_raster_plot(rawData.spikes.amps, rawData.spikes.depths,
                                                   rawData.spikes.times)
            self.setStartingRange(data)
            plotData.ampRaster = data

        return plotData.ampRaster

    def getAmpDepthFr(self, spikeSort):
        plotData = self.ctrl.get_spikeSorterPlotData(name=spikeSort)
        if plotData.ampDepthFr is None:
            rawData = self.ctrl.get_spikeSorterRawData(name=spikeSort)
            data = (ephys_plots.
                    scatter_amp_depth_fr_plot(rawData.spikes.amps, rawData.spikes.clusters,
                                              rawData.spikes.depths, rawData.spikes.times))
            self.setStartingRange(data)
            data.set_line_color('k')
            plotData.ampDepthFr = data

        return plotData.ampDepthFr

    def getClusterRaster(self, spikeSort):
        plotData = self.ctrl.get_spikeSorterPlotData(name=spikeSort)
        if plotData.clustRaster is None:
            rawData = self.ctrl.get_spikeSorterRawData(name=spikeSort)
            data = ephys_plots.scatter_cluster_plot(rawData.spikes.clusters, rawData.spikes.depths,
                                                    rawData.spikes.times)
            self.setStartingRange(data)
            plotData.clustRaster = data

        return plotData.clustRaster

    def getClusterCount(self, spikeSort):
        plotData = self.ctrl.get_spikeSorterPlotData(name=spikeSort)
        if plotData.clustCount is None:
            rawData = self.ctrl.get_spikeSorterRawData(name=spikeSort)
            data = LinePlot(*np.unique(rawData.spikes.clusters, return_counts=True))
            data.set_labels(xlabel='Cluster number', ylabel='Cluster count')
            self.setStartingRange(data)
            plotData.clustCount = data

        return plotData.clustCount

    def removeNans(self, spikes):
        kp_idx = np.bitwise_and(~np.isnan(spikes['depths']), ~np.isnan(spikes['amps']))
        for key in spikes.keys():
            spikes[key] = spikes[key][kp_idx]

        return spikes

    def setStartingRange(self, data):
        data.clim_start = data.clim
        data.xlim_start = data.xlim
        data.ylim_start = data.ylim

    def normaliseAcrossSorters(self, plotType):
        # have to go through this twice
        min_val = {'x': [], 'y': [], 'c': []}
        max_val = {'x': [], 'y': [], 'c': []}
        for spikeSort in self.spikeSortPlotData:
            plotData = spikeSort.__getattribute__(plotType)
            min_val['x'].append(plotData.xlim[0])
            max_val['x'].append(plotData.xlim[1])
            min_val['y'].append(plotData.ylim[0])
            max_val['y'].append(plotData.ylim[1])
            min_val['c'].append(plotData.clim[0]) if plotData.clim[0] else min_val['c'].append(0)
            max_val['c'].append(plotData.clim[1]) if plotData.clim[1] else max_val['c'].append(1)

        for spikeSort in self.spikeSortPlotData:
            plotData = spikeSort.__getattribute__(plotType)
            plotData.set_clim(clim=[np.min(min_val['c']), np.max(max_val['c'])])
            plotData.set_xlim(xlim=[np.min(min_val['x']), np.max(max_val['x'])])
            plotData.set_ylim(ylim=[np.min(min_val['y']), np.max(max_val['y'])])

    def unNormaliseAcrossSorters(self, plotType):
        for spikeSort in self.spikeSortPlotData:
            plotData = spikeSort.__getattribute__(plotType)
            plotData.set_clim(plotData.clim_start)
            plotData.set_xlim(plotData.xlim_start)
            plotData.set_ylim(plotData.ylim_start)


@dataclass
class SpikeSortFigures:
    """
    Class for keeping track of image layers.
    :param image_item
    :param pg_kwargs: pyqtgraph setImage arguments: {'levels': None, 'lut': None, 'opacity': 1.0}
    :param slice_kwargs: iblatlas.atlas.slice arguments: {'volume': 'image', 'mode': 'clip'}
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
    :param slice_kwargs: iblatlas.atlas.slice arguments: {'volume': 'image', 'mode': 'clip'}
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
    :param slice_kwargs: iblatlas.atlas.slice arguments: {'volume': 'image', 'mode': 'clip'}
    :param
    """
    name: str = field(default='base')
    frRaster: dict = field(default=None)
    ampRaster: dict = field(default=None)
    ampDepthFr: dict = field(default=None)
    clustRaster: dict = field(default=None)
    clustCount: dict = field(default=None)

def view(eid, title=None):
    """
    """
    qt.create_app()
    av = MainWindow._get_or_create(eid, title=title)
    av.show()
    return av
