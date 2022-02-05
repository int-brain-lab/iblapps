from pathlib import Path

import numpy as np
import scipy.signal

from PyQt5 import QtWidgets, QtCore, QtGui, uic
import pyqtgraph as pg
from brainbox.processing import bincount2D

from easyqc.gui import viewseis
import one.alf.io as alfio
from one.alf.files import get_session_path
from iblutil.numerical import ismember
from ibllib.io import spikeglx
from ibllib.dsp import voltage

import qt
from brainbox.io.one import SpikeSortingLoader

T_BIN = .007  # time bin size in secs
D_BIN = 10  # depth bin size in um

SNS_PALETTE = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    (1.0, 0.4980392156862745, 0.054901960784313725),
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)]

YMAX = 4000


def view_raster(bin_file):
    bin_file = Path(bin_file)
    pname = bin_file.parent.name
    session_path = get_session_path(bin_file)
    ssl = SpikeSortingLoader(session_path=session_path, pname=pname)
    spikes, clusters, channels = ssl.load_spike_sorting(dataset_types=['spikes.samples'])
    trials = alfio.load_object(ssl.session_path.joinpath('alf'), 'trials')
    return RasterView(bin_file, spikes, clusters, trials=trials)


class RasterView(QtWidgets.QMainWindow):
    def __init__(self, bin_file, spikes, clusters, channels=None, trials=None, *args, **kwargs):
        self.sr = spikeglx.Reader(bin_file)
        self.spikes = spikes
        self.clusters = clusters
        self.channels = channels
        self.trials = trials
        self.eqcs = []
        super(RasterView, self).__init__(*args, **kwargs)
        # wave by Diana Militano from the Noun Projectp
        uic.loadUi(Path(__file__).parent.joinpath('raster.ui'), self)
        background_color = self.palette().color(self.backgroundRole())
        self.plotItem_raster.setAspectLocked(False)
        self.imageItem_raster = pg.ImageItem()
        self.plotItem_raster.setBackground(background_color)
        self.plotItem_raster.addItem(self.imageItem_raster)
        self.viewBox_raster = self.plotItem_raster.getPlotItem().getViewBox()
        s = self.viewBox_raster.scene()
        # vb.scene().sigMouseMoved.connect(self.mouseMoveEvent)
        s.sigMouseClicked.connect(self.mouseClick)
        ################################################### set image
        iok = ~np.isnan(spikes.depths)
        self.raster, self.rtimes, self.depths = bincount2D(
            spikes.times[iok], spikes.depths[iok], T_BIN, D_BIN)
        self.imageItem_raster.setImage(np.flip(self.raster.T))
        transform = [T_BIN, 0., 0., 0., D_BIN, 0.,  - .5, - .5, 1.]
        self.transform = np.array(transform).reshape((3, 3)).T
        self.imageItem_raster.setTransform(QtGui.QTransform(*transform))
        self.plotItem_raster.setLimits(xMin=0, xMax=self.rtimes[-1], yMin=0, yMax=self.depths[-1])
        # set colormap
        cm = pg.colormap.get('Greys', source='matplotlib')  # prepare a linear color map
        bar = pg.ColorBarItem(values=(0, .5), colorMap=cm)  # prepare interactive color bar
        # Have ColorBarItem control colors of img and appear in 'plot':
        bar.setImageItem(self.imageItem_raster)
        ################################################## plot location
        # self.view.layers[label] = {'layer': new_scatter, 'type': 'scatter'}
        self.line_eqc = pg.PlotCurveItem()
        self.plotItem_raster.addItem(self.line_eqc)
        # self.plotItem_raster.removeItem(new_curve)
        ################################################## plot trials
        if self.trials is not None:
            trial_times = dict(
                goCue_times=trials['goCue_times'],
                error_times=trials['feedback_times'][trials['feedbackType'] == -1],
                reward_times=trials['feedback_times'][trials['feedbackType'] == 1])
            self.trial_lines = {}
            for i, k in enumerate(trial_times):
                self.trial_lines[k] = pg.PlotCurveItem()
                self.plotItem_raster.addItem(self.trial_lines[k])
                x = np.tile(trial_times[k][:, np.newaxis], (1, 2)).flatten()
                y = np.tile(np.array([0, 1, 1, 0]), int(trial_times[k].shape[0] / 2 + 1))[
                    :trial_times[k].shape[0] * 2] * YMAX
                self.trial_lines[k].setData(x=x.flatten(), y=y.flatten(), pen=pg.mkPen(np.array(SNS_PALETTE[i]) * 256))

        self.show()

    def mouseClick(self, event):
        """Draws a line on the raster and display in EasyQC"""
        if not event.double():
            return
        qxy = self.imageItem_raster.mapFromScene(event.scenePos())
        x = qxy.x()
        self.show_ephys(t0=self.rtimes[int(x - .5)])
        ymax = np.max(self.depths) + 50
        self.line_eqc.setData(x=x + np.array([-.5, -.5, .5, .5]),
                              y=np.array([0, ymax, ymax, 0]),
                              pen=pg.mkPen((0, 255, 0)))


    def keyPressEvent(self, e):
        """
        page-up / ctrl + a :  gain up
        page-down / ctrl + z : gain down
        :param e:
        """
        k, m = (e.key(), e.modifiers())
        # page up / ctrl + a
        if k == QtCore.Qt.Key_PageUp or (
                m == QtCore.Qt.ControlModifier and k == QtCore.Qt.Key_A):
            self.imageItem_raster.setLevels([0, self.imageItem_raster.levels[1] / 1.4])
        # page down / ctrl + z
        elif k == QtCore.Qt.Key_PageDown or (
                m == QtCore.Qt.ControlModifier and k == QtCore.Qt.Key_Z):
            self.imageItem_raster.setLevels([0, self.imageItem_raster.levels[1] * 1.4])

    def show_ephys(self, t0, tlen=1):

        first = int(t0 * self.sr.fs)
        last = first + int(self.sr.fs * tlen)

        raw = self.sr[first:last, : - self.sr.nsync].T

        butter_kwargs = {'N': 3, 'Wn': 300 / self.sr.fs * 2, 'btype': 'highpass'}
        sos = scipy.signal.butter(**butter_kwargs, output='sos')
        butt = scipy.signal.sosfiltfilt(sos, raw)
        destripe = voltage.destripe(raw, fs=self.sr.fs)

        self.eqc_raw = viewephys(butt, self.sr.fs, channels=None, br=None, title='butt', t0=t0, t_scalar=1)
        self.eqc_des = viewephys(destripe, self.sr.fs, channels=None, br=None, title='destripe', t0=t0, t_scalar=1)

        eqc_xrange = [t0 + tlen / 2 - 0.01, t0 + tlen / 2 + 0.01]
        self.eqc_des.viewBox_seismic.setXRange(*eqc_xrange)
        self.eqc_raw.viewBox_seismic.setXRange(*eqc_xrange)

        # eqc2 = viewephys(butt - destripe, self.sr.fs, channels=None, br=None, title='diff')
        # overlay spikes
        tprobe = self.spikes.samples / self.sr.fs
        slice_spikes = slice(np.searchsorted(tprobe, t0), np.searchsorted(tprobe, t0 + tlen))
        t = tprobe[slice_spikes]
        c = self.clusters.channels[self.spikes.clusters[slice_spikes]]
        self.eqc_raw.ctrl.add_scatter(t, c)
        self.eqc_des.ctrl.add_scatter(t, c)




def viewephys(data, fs, channels=None, br=None, title='ephys', t0=0, t_scalar=1e3):
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
    if channels is None or br is None:
        channels = trace_header(version = 1)
        eqc = viewseis(data.T, si=1 / fs * t_scalar, h=channels, title=title, taxis=0, t0=t0)
        return eqc
    else:
        _, ir = ismember(channels['atlas_id'], br.id)
        image = br.rgb[ir].astype(np.uint8)
        image = image[np.newaxis, :, :]


    eqc = viewseis(data.T, si=1 / fs * t_scalar, h=channels, title=title, taxis=0)
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