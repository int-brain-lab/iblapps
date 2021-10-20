from easyqc.gui import viewseis
from ibllib.dsp import voltage
from ibllib.ephys import neuropixel
from viewspikes.data import stream, get_ks2
from viewspikes.plots import overlay_spikes
import scipy
from PyQt5 import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import qt
from one.api import ONE
from iblutil.util import Bunch

import atlaselectrophysiology.ephys_atlas_gui as alignment_window
import data_exploration_gui.gui_main as trial_window


class AlignmentWindow(alignment_window.MainWindow):
    def __init__(self, probe_id=None, one=None, histology=False, spike_collection=None):

        self.ap = None  # spikeglx.Reader for ap band
        self.lf = None  # spikeglx.Reader for lf band
        self.line_x = None
        self.trial_curve = None
        self.time_plot = None
        self.trial_gui = None
        self.clicked = None
        self.eqc = {}  # handles for viewdata windows

        super(AlignmentWindow, self).__init__(probe_id=probe_id, one=one, histology=histology,
                                              spike_collection=spike_collection)
        # remove the lines from the plots
        self.remove_lines_points()
        self.lines_features = []
        self.lines_tracks = []
        self.points = []

        self.plotdata.channels = Bunch()
        self.plotdata.channels['localCoordinates'] = self.plotdata.chn_coords_all
        self.plotdata.channels['rawInd'] = self.plotdata.chn_ind_all

    def on_mouse_double_clicked(self, event):
        if not self.offline:
            if event.double() and event.modifiers() and QtCore.Qt.ShiftModifier:
                pos = self.data_plot.mapFromScene(event.scenePos())
                if self.line_x is not None:
                    self.fig_img.removeItem(self.line_x)

                self.line_x = pg.InfiniteLine(pos=pos.x() * self.x_scale, angle=90,
                                              pen=self.kpen_dot, movable=False)
                self.line_x.setZValue(100)
                self.fig_img.addItem(self.line_x)
                self.stream_ap(pos.x() * self.x_scale)
                self.stream_lf(pos.x() * self.x_scale)

                return

    def plot_image(self, data):
        super().plot_image(data)
        self.remove_trial_curve(data['xaxis'])
        self.remove_line_x(data['xaxis'])
        if 'Time' in data['xaxis']:
            self.time_plot = True
        else:
            self.time_plot = False

    def plot_scatter(self, data):
        super().plot_scatter(data)
        self.remove_trial_curve(data['xaxis'])
        self.remove_line_x(data['xaxis'])
        if 'Time' in data['xaxis']:
            self.time_plot = True
        else:
            self.time_plot = False

    def add_trials(self, trial_key='feedback_times'):
        self.selected_trials = self.plotdata.trials[trial_key]
        x, y = self.vertical_lines(self.selected_trials, 0, 3840)
        self.trial_curve = pg.PlotCurveItem()
        self.trial_curve.setData(x=x, y=y, pen=self.rpen_dot, connect='finite')
        self.trial_curve.setClickable(True)
        self.fig_img.addItem(self.trial_curve)
        self.fig_img.scene().sigMouseClicked.connect(self.on_mouse_clicked)
        self.trial_curve.sigClicked.connect(self.trial_line_clicked)

    def remove_trials(self):
        self.fig_img.removeItem(self.trial_curve)

    def vertical_lines(self, x, ymin, ymax):

        x = np.tile(x, (3, 1))
        x[2, :] = np.nan
        y = np.zeros_like(x)
        y[0, :] = ymin
        y[1, :] = ymax
        y[2, :] = np.nan

        return x.T.flatten(), y.T.flatten()

    def trial_line_clicked(self, ev):
        self.clicked = ev

    def on_mouse_clicked(self, event):
        if self.trial_gui is not None:
            if not event.double() and type(self.clicked) == pg.PlotCurveItem:
                self.pos = self.data_plot.mapFromScene(event.scenePos())
                x = self.pos.x() * self.x_scale
                trial_id = np.argmin(np.abs(self.selected_trials - x))
                print(trial_id)

                idx = np.where(self.trial_gui.data.y == 10 * trial_id)
                self.trial_scat = pg.ScatterPlotItem()
                self.trial_gui.plots.fig4_raster.fig.addItem(self.trial_scat)
                self.trial_scat.setData(self.trial_gui.data.x[idx], self.trial_gui.data.y[idx],
                                        brush='r', size=5)

                self.clicked = None

    def stream_lf(self, t):
        if self.lf is not None:
            self.lf.close()

        self.lf, dsets, t0 = stream(
            self.loaddata.probe_id, t=t, one=self.loaddata.one, cache=True, typ='lf')
        sos = scipy.signal.butter(3, 5 / self.lf.fs / 2, btype='highpass', output='sos')
        butt = scipy.signal.sosfiltfilt(sos, self.lf[:, :-1].T)
        h = neuropixel.trace_header()
        self.eqc['raw_lf'] = viewseis(
            butt.T, si=1 / self.lf.fs, h=h, t0=t0, title='raw_lf', taxis=0)
        self.lf.close()

    def stream_ap(self, t):
        if self.ap is not None:
            self.ap.close()

        self.ap, dsets, t0 = stream(
            self.loaddata.probe_id, t=t, one=self.loaddata.one, cache=True)
        raw = self.ap[:, :-1].T
        h = neuropixel.trace_header()
        sos = scipy.signal.butter(3, 300 / self.ap.fs / 2, btype='highpass', output='sos')
        butt = scipy.signal.sosfiltfilt(sos, raw)
        destripe = voltage.destripe(raw, fs=self.ap.fs)
        ks2 = get_ks2(raw, dsets, self.loaddata.one)
        self.eqc['butterworth'] = viewseis(butt.T, si=1 / self.ap.fs, h=h, t0=t0, title='butt',
                                           taxis=0)
        self.eqc['destripe'] = viewseis(destripe.T, si=1 / self.ap.fs, h=h, t0=t0, title='destr',
                                        taxis=0)
        self.eqc['ks2'] = viewseis(ks2.T, si=1 / self.ap.fs, h=h, t0=t0, title='ks2', taxis=0)

        overlay_spikes(self.eqc['butterworth'], self.plotdata.spikes, self.plotdata.clusters,
                       self.plotdata.channels)
        overlay_spikes(self.eqc['destripe'], self.plotdata.spikes, self.plotdata.clusters,
                       self.plotdata.channels)
        overlay_spikes(self.eqc['ks2'], self.plotdata.spikes, self.plotdata.clusters,
                       self.plotdata.channels)
        self.ap.close()

    def remove_line_x(self, xaxis):
        """
        If we have any horizontal lines to indicate the time points delete them if the x axis is
        not time
        :param xaxis:
        :return:
        """
        if self.line_x is not None:
            self.fig_img.removeItem(self.line_x)
            if 'Time' in xaxis:
                self.fig_img.addItem(self.line_x)

    def remove_trial_curve(self, xaxis):
        if self.trial_curve is not None:
            self.fig_img.removeItem(self.trial_curve)
            if 'Time' in xaxis:
                self.fig_img.addItem(self.trial_curve)

    def remove_lines_points(self):
        super().remove_lines_points()
        self.remove_line_x('la')
        self.remove_trial_curve('la')

    def add_lines_points(self):
        super().add_lines_points()
        if self.time_plot:
            self.remove_line_x('Time')
            self.remove_trial_curve('Time')

    def closeEvent(self, event):
        """
        Close the spikeglx file when window is closed
        """
        super().closeEvent(event)
        if self.ap is not None:
            self.ap.close()

    def complete_button_pressed(self):
        QtGui.QMessageBox.information(self, 'Status', ("Not going to upload any results, to do"
                                                       " an alignment, launch normally"))


class TrialWindow(trial_window.MainWindow):
    def __init__(self):
        super(TrialWindow, self).__init__()
        self.alignment_gui = None
        self.scat = None

    def on_scatter_plot_clicked(self, scatter, point):
        super().on_scatter_plot_clicked(scatter, point)
        self.add_clust_scatter()

    def on_cluster_list_clicked(self):
        super().on_cluster_list_clicked()
        self.add_clust_scatter()

    def on_next_cluster_clicked(self):
        super().on_next_cluster_clicked()
        self.add_clust_scatter()

    def on_previous_cluster_clicked(self):
        super().on_previous_cluster_clicked()
        self.add_clust_scatter()

    def add_clust_scatter(self):
        if not self.scat:
            self.scat = pg.ScatterPlotItem()
            self.alignment_gui.fig_img.addItem(self.scat)

        self.scat.setData(self.data.spikes.times[self.data.clus_idx],
                          self.data.spikes.depths[self.data.clus_idx], brush='g', size=5)


def load_extra_data(probe_id, one=None, spike_collection=None):
    one = one or ONE()
    eid, probe = one.pid2eid(probe_id)
    if spike_collection == '':
        collection = f'alf/{probe}'
    elif spike_collection:
        collection = f'alf/{probe}/{spike_collection}'
    else:
        # Pykilosort is default, if not present look for normal kilosort
        # Find all collections
        all_collections = one.list_collections(eid)

        if f'alf/{probe}/pykilosort' in all_collections:
            collection = f'alf/{probe}/pykilosort'
        else:
            collection = f'alf/{probe}'

    _ = one.load_object(eid, obj='spikes', collection=collection,
                        attribute='samples')
    trials = one.load_object(eid, obj='trials')

    return trials


def viewer(probe_id=None, one=None, data_explore=False, spike_collection=None):
    """
    """
    qt.create_app()
    trials = load_extra_data(probe_id, one=one, spike_collection=spike_collection)
    av = AlignmentWindow(probe_id=probe_id, one=one, spike_collection=spike_collection)
    av.plotdata.trials = trials
    av.show()

    if data_explore:
        data = Bunch()
        data['spikes'] = av.plotdata.spikes
        data['clusters'] = av.plotdata.clusters
        data['trials'] = av.plotdata.trials
        bv = TrialWindow()
        bv.on_data_given(data)
        av.trial_gui = bv
        bv.alignment_gui = av
        bv.show()

    return av
