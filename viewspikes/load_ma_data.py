from pathlib import Path
from oneibl.one import ONE
import alf.io
from brainbox.core import Bunch
import qt
import numpy as np
import pyqtgraph as pg


import atlaselectrophysiology.ephys_atlas_gui as alignment_window
import data_exploration_gui.gui_main as trial_window


# some extra controls

class AlignmentWindow(alignment_window.MainWindow):
    def __init__(self, offline=False, probe_id=None, one=None):
        super(AlignmentWindow, self).__init__(probe_id=probe_id, one=one)
        self.trial_gui = None


    def cluster_clicked(self, item, point):
        clust = super().cluster_clicked(item, point)
        print(clust)
        self.trial_gui.on_cluster_chosen(clust)

    def add_trials_to_raster(self, trial_key='feedback_times'):
        self.selected_trials = self.trial_gui.data.trials[trial_key]
        x, y = self.vertical_lines(self.selected_trials, 0, 3840)
        trial_curve = pg.PlotCurveItem()
        trial_curve.setData(x=x, y=y, pen=self.rpen_dot, connect='finite')
        trial_curve.setClickable(True)
        self.fig_img.addItem(trial_curve)
        self.fig_img.scene().sigMouseClicked.connect(self.on_mouse_clicked)
        trial_curve.sigClicked.connect(self.trial_line_clicked)


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
        if not event.double() and type(self.clicked) == pg.PlotCurveItem:
            self.pos = self.data_plot.mapFromScene(event.scenePos())
            x = self.pos.x() * self.x_scale
            trial_id = np.argmin(np.abs(self.selected_trials - x))








            # highlight the trial in the trial gui



            self.clicked = None









class TrialWindow(trial_window.MainWindow):
    def __init__(self):
        super(TrialWindow, self).__init__()
        self.alignment_gui = None


def load_data(eid, probe, one=None):
    one = one or ONE()
    session_path = one.path_from_eid(eid).joinpath('alf')
    probe_path = session_path.joinpath(probe)
    data = Bunch()
    data['trials'] = alf.io.load_object(session_path, 'trials', namespace='ibl')
    data['spikes'] = alf.io.load_object(probe_path, 'spikes')
    data['clusters'] = alf.io.load_object(probe_path, 'clusters')

    return data

def viewer(probe_id=None, one=None):
    """
    """
    probe = one.alyx.rest('insertions', 'list', id=probe_id)[0]
    data = load_data(probe['session'], probe['name'], one=one)
    qt.create_app()
    av = AlignmentWindow(probe_id=probe_id, one=one)
    bv = TrialWindow()
    bv.on_data_given(data)
    av.trial_gui = bv

    av.show()
    bv.show()
    return av, bv

