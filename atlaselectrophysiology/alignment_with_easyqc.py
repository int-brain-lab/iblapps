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

import atlaselectrophysiology.ephys_atlas_gui as alignment_window


class AlignmentWindow(alignment_window.MainWindow):
    def __init__(self, probe_id=None, one=None, histology=False):

        self.sr = None
        self.line_x = None
        self.trial_curve = None
        self.time_plot = None

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




def load_extra_data(probe_id, one=None):
    one = one or ONE()
    dtypes_probe = ['spikes.samples']

    dsets = one.alyx.rest('datasets', 'list', probe_insertion=probe_id,
                          dataset_type='spikes.samples')
    _ = one.download_datasets(dsets)

    trials = one.load_object(id=dsets[0]['session'][-36:], obj='trials')

    return trials


def viewer(probe_id=None, one=None):
    """
    """
    qt.create_app()
    trials = load_extra_data(probe_id, one=one)
    av = AlignmentWindow(probe_id=probe_id, one=one)
    av.plotdata.trials = trials
    av.show()
    return av
