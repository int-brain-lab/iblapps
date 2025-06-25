from qtpy import QtWidgets
from atlaselectrophysiology.qt_utils.utils import PopupWindow
from iblatlas.atlas import Insertion
from ibllib.pipes import histology
import numpy as np
from datetime import timedelta


PLUGIN_NAME = "Nearby Sessions"

def setup(parent):

    parent.plugins[PLUGIN_NAME] = dict()
    parent.plugins[PLUGIN_NAME]['loader'] = LoadData(parent.shank.one, parent.shank.brain_atlas)

    action = QtWidgets.QAction(PLUGIN_NAME, parent)
    action.triggered.connect(lambda: callback(parent))
    parent.plugin_options.addAction(action)


def callback(parent):
    loader = parent.plugins[PLUGIN_NAME]['loader']
    loader.load(parent.shank.traj_id)
    parent.nearby_sessions = NearbySessions._get_or_create('Nearby Sessions', data=loader.data, parent=parent)


class LoadData:
    def __init__(self, one, brain_atlas):
        self.one = one
        self.brain_atlas = brain_atlas
        self.traj_id = None
        self.traj_coords = None
        self.traj_ids = None
        self.data = dict()

    def load(self, traj_id):
        """
        Find sessions that have trajectories close to the currently selected session
        :return close_session: list of nearby sessions ordered by absolute distance, displayed as
        subject + date + probe
        :type: list of strings
        :return close_dist: absolute distance to nearby sessions
        :type: list of float
        :return close_dist_mlap: absolute distance to nearby sessions, only using ml and ap
        directions
        :type: list of float
        """

        if self.traj_ids is None:
            self.sess_with_hist = (
                self.one.alyx.rest('trajectories', 'list', provenance='Histology track',
                                    django='x__isnull,False,probe_insertion__datasets__name__icontains,spikes.times',
                                    expires=timedelta(days=30)))

            depths = np.arange(200, 4100, 20) / 1e6
            trajectories = [Insertion.from_dict(sess, self.brain_atlas) for sess in self.sess_with_hist]
            self.traj_ids = [sess['id'] for sess in self.sess_with_hist]
            self.traj_coords = np.empty((len(self.traj_ids), len(depths), 3))
            for iT, traj in enumerate(trajectories):
                self.traj_coords[iT, :] = (histology.interpolate_along_track(np.vstack([traj.tip, traj.entry]), depths))

        if traj_id != self.traj_id:
            self.traj_id = traj_id
            chosen_traj = self.traj_ids.index(self.traj_id)
            avg_dist = np.mean(np.sqrt(np.sum((self.traj_coords - self.traj_coords[chosen_traj]) ** 2,axis=2)), axis=1)
            avg_dist_mlap = np.mean(np.sqrt(np.sum((self.traj_coords[:, :, 0:2] - self.traj_coords[chosen_traj][:, 0:2]) ** 2, axis=2)), axis=1)

            closest_traj = np.argsort(avg_dist)
            self.data['close_dist'] = avg_dist[closest_traj[0:10]] * 1e6
            self.data['close_dist_mlap'] = avg_dist_mlap[closest_traj[0:10]] * 1e6

            self.data['close_sessions'] = []
            for sess_idx in closest_traj[0:10]:
                self.data['close_sessions'].append((self.sess_with_hist[sess_idx]['session']['subject'] + ' ' +
                                       self.sess_with_hist[sess_idx]['session']['start_time'][:10] +
                                       ' ' + self.sess_with_hist[sess_idx]['probe_name']))

    def reset(self):
        self.data = dict()
        self.traj_id = None


class NearbySessions(PopupWindow):
    def __init__(self, title, data=None, parent=None):
        self.data = data
        super().__init__(title, parent=parent, size=(400, 300), graphics=False)

    def setup(self):

        nearby_table = QtWidgets.QTableWidget()
        nearby_table.setRowCount(10)
        nearby_table.setColumnCount(3)

        nearby_table.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem('Session'))
        nearby_table.setHorizontalHeaderItem(1, QtWidgets.QTableWidgetItem('dist'))
        nearby_table.setHorizontalHeaderItem(2, QtWidgets.QTableWidgetItem('dist_mlap'))
        nearby_table.setSortingEnabled(True)
        for iT, (near, dist, dist_mlap) in enumerate(zip(self.data['close_sessions'], self.data['close_dist'], self.data['close_dist_mlap'])):
            dist_item = QtWidgets.QTableWidgetItem()
            dist_item.setData(0, int(dist))
            dist_mlap_item = QtWidgets.QTableWidgetItem()
            dist_mlap_item.setData(0, int(dist_mlap))
            nearby_table.setItem(iT, 0, QtWidgets.QTableWidgetItem(near))
            nearby_table.setItem(iT, 1, dist_item)
            nearby_table.setItem(iT, 2, dist_mlap_item)

        self.layout.addWidget(nearby_table)
