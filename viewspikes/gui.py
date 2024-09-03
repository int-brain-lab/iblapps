from pathlib import Path

import numpy as np
import scipy.signal
from PyQt5 import QtWidgets, QtCore, QtGui, uic
import pyqtgraph as pg

from iblutil.numerical import bincount2D
from viewephys.gui import viewephys
import ibldsp
from brainbox.io.one import EphysSessionLoader, SpikeSortingLoader
from iblatlas.atlas import BrainRegions


regions = BrainRegions()
T_BIN = .007  # time bin size in secs
D_BIN = 20  # depth bin size in um

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


def view_raster(pid, one, stream=True):
    ssl = SpikeSortingLoader(one=one, pid=pid)
    sl = EphysSessionLoader(one=one, eid=ssl.eid)
    sl.load_trials()
    spikes, clusters, channels = ssl.load_spike_sorting(dataset_types=['spikes.samples'])

    return RasterView(ssl, spikes, clusters, channels, trials=sl.trials, stream=stream)


class RasterView(QtWidgets.QMainWindow):
    def __init__(self, ssl, spikes, clusters, channels=None, trials=None, stream=True, *args, **kwargs):
        self.ssl = ssl
        self.spikes = spikes
        self.clusters = clusters
        self.channels = channels
        self.trials = trials
        # self.sr_lf = ssl.raw_electrophysiology(band='lf', stream=False) that would be cool to also have the LFP
        self.sr = ssl.raw_electrophysiology(band='ap', stream=stream)
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
        self.imageItem_raster.setImage(self.raster.T)
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
        ################################################## plot trials
        if self.trials is not None:
            trial_times = dict(
                goCue_times=trials['goCue_times'].values,
                error_times=trials['feedback_times'][trials['feedbackType'] == -1].values,
                reward_times=trials['feedback_times'][trials['feedbackType'] == 1].values)
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

        s0 = int(self.ssl.samples2times(t0, direction='reverse'))
        s1 = s0 + int(self.sr.fs * tlen)
        raw = self.sr[s0:s1, : - self.sr.nsync].T

        butter_kwargs = {'N': 3, 'Wn': 300 / self.sr.fs * 2, 'btype': 'highpass'}

        sos = scipy.signal.butter(**butter_kwargs, output='sos')
        butt = scipy.signal.sosfiltfilt(sos, raw)
        destripe = ibldsp.voltage.destripe(raw, fs=self.sr.fs)

        self.eqc_raw = viewephys(butt, self.sr.fs, channels=self.channels, br=regions, title='butt', t0=t0, t_scalar=1)
        self.eqc_des = viewephys(destripe, self.sr.fs, channels=self.channels, br=regions, title='destripe', t0=t0, t_scalar=1)

        eqc_xrange = [t0 + tlen / 2 - 0.01, t0 + tlen / 2 + 0.01]
        self.eqc_des.viewBox_seismic.setXRange(*eqc_xrange)
        self.eqc_raw.viewBox_seismic.setXRange(*eqc_xrange)

        # we slice the spikes using the samples according to ephys time, but display in session times
        slice_spikes = slice(np.searchsorted(self.spikes['samples'], s0), np.searchsorted(self.spikes['samples'], s1))
        t = self.spikes['times'][slice_spikes]
        c = self.clusters.channels[self.spikes.clusters[slice_spikes]]
        self.eqc_raw.ctrl.add_scatter(t, c)
        self.eqc_des.ctrl.add_scatter(t, c)
