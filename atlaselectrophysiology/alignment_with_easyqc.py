from easyqc.gui import viewseis
from ibllib.dsp import voltage
from ibllib.ephys import neuropixel
from viewspikes.data import stream, get_ks2, get_spikes
from viewspikes.plots import overlay_spikes
import scipy
from PyQt5 import QtCore
import numpy as np
import pyqtgraph as pg
import qt
from oneibl.one import ONE
from brainbox.core import Bunch

import atlaselectrophysiology.ephys_atlas_gui as alignment_window
import data_exploration_gui.gui_main as trial_window


class AlignmentWindow(alignment_window.MainWindow):
    def __init__(self, probe_id=None, one=None, histology=False):

        self.sr = None
        self.line_x = None
        self.trial_curve = None
        self.time_plot = None
        self.trial_gui = None

        super(AlignmentWindow, self).__init__(probe_id=probe_id, one=one, histology=histology)


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
                self.stream_raw_data(pos.x() * self.x_scale)

                return

        super().on_mouse_double_clicked(event)

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


                idx = np.where(self.trial_gui.data.y == 10*trial_id)
                self.trial_scat = pg.ScatterPlotItem()
                self.trial_gui.plots.fig4_raster.fig.addItem(self.trial_scat)
                self.trial_scat.setData(self.trial_gui.data.x[idx], self.trial_gui.data.y[idx],
                                        brush='r', size=1)

                self.clicked = None


    def stream_raw_data(self, t):
        if self.sr is not None:
            self.sr.close()
        self.sr, dsets = stream(self.loaddata.probe_id, t0=t, one=self.loaddata.one, cache=True)
        raw = self.sr[:, :-1].T
        h = neuropixel.trace_header()
        sos = scipy.signal.butter(3, 300 / self.sr.fs / 2, btype='highpass', output='sos')
        butt = scipy.signal.sosfiltfilt(sos, raw)
        fk_kwargs = {'dx': 1, 'vbounds': [0, 1e6], 'ntr_pad': 160, 'ntr_tap': 0, 'lagc': .01,
                     'btype': 'lowpass'}
        destripe = voltage.destripe(raw, fs=self.sr.fs, fk_kwargs=fk_kwargs,
                                    tr_sel=np.arange(raw.shape[0]))
        ks2 = get_ks2(raw, dsets, self.loaddata.one)
        eqc_butt = viewseis(butt.T, si=1 / self.sr.fs, h=h, t0=t, title='butt', taxis=0)
        eqc_dest = viewseis(destripe.T, si=1 / self.sr.fs, h=h, t0=t, title='destr', taxis=0)
        eqc_ks2 = viewseis(ks2.T, si=1 / self.sr.fs, h=h, t0=t, title='ks2', taxis=0)

        overlay_spikes(eqc_butt, self.plotdata.spikes, self.plotdata.clusters,
                       self.plotdata.channels)
        overlay_spikes(eqc_dest, self.plotdata.spikes, self.plotdata.clusters,
                       self.plotdata.channels)
        overlay_spikes(eqc_ks2, self.plotdata.spikes, self.plotdata.clusters,
                       self.plotdata.channels)

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
                          self.data.spikes.depths[self.data.clus_idx], brush='r')



def load_extra_data(probe_id, one=None):
    one = one or ONE()
    dtypes_probe = ['spikes.samples']

    dsets = one.alyx.rest('datasets', 'list', probe_insertion=probe_id,
                          dataset_type='spikes.samples')
    _ = one.download_datasets(dsets)

    trials = one.load_object(id=dsets[0]['session'][-36:], obj='trials')

    return trials


def viewer(probe_id=None, one=None, data_explore=False):
    """
    """
    qt.create_app()
    trials = load_extra_data(probe_id, one=one)
    av = AlignmentWindow(probe_id=probe_id, one=one)
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
