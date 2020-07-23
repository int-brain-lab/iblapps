from oneibl.one import ONE
from ibllib.atlas import atlas
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import matplotlib.pyplot as plt
from ibllib.pipes.ephys_alignment import EphysAlignment
from pathlib import Path
import alf.io

brain_atlas = atlas.AllenAtlas(25)
ONE_BASE_URL = "https://alyx.internationalbrainlab.org"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.resize(1600, 800)
        self.setWindowTitle('Closest Boundary')

        main_widget = QtGui.QWidget()
        self.setCentralWidget(main_widget)

        one = ONE()
        eid = 'ac935c7e-0de7-42d7-b751-a29d7de74fe0'
        #eid = 'a92c4b1d-46bd-457e-a1f4-414265f0e2d4'
        probe_label = 'probe00'

        picks = one.alyx.rest('insertions', 'list', session=eid, name=probe_label)
        xyz_picks = np.array(picks[0]['json']['xyz_picks']) / 1e6

        ephys_align = EphysAlignment(xyz_picks)
        xyz_samples = ephys_align.xyz_samples

        sampling_trk = np.arange(ephys_align.track_extent[0],
                                 ephys_align.track_extent[-1] - 10 * 1e-6, 10 * 1e-6)
        insertion = atlas.Insertion.from_track(xyz_samples, brain_atlas=brain_atlas)

        allen_path = Path(Path(atlas.__file__).parent, 'allen_structure_tree.csv')
        allen = alf.io.load_file_content(allen_path)

        #start = time.time()
        nearby_bounds = ephys_align.get_nearest_boundary(xyz_samples, allen)
        #end = time.time()
        #print(f"Runtime of the program is {end - start}")



        #vector = insertion.trajectory.vector
        #points = xyz_samples
        #extent = 100 / 1e6
        #steps = 8
        #min_dist = []
        #region_id = []
        #parent_dist = []
        #parent_id = []
        #parent_colour = []
#
        #color = np.empty((len(points)-1, steps+1, steps+1, 3))
        #for iP, point in enumerate(points[:, :]):
        #    d = np.dot(vector, point)
        #    x_vals = np.sort(np.r_[np.linspace(point[0] - extent, point[0] + extent, steps),
        #                           point[0]])
        #    y_vals = np.sort(np.r_[np.linspace(point[1] - extent, point[1] + extent, steps),
        #                           point[1]])
#
        #    X, Y = np.meshgrid(x_vals, y_vals)
        #    Z = (d - vector[0] * X - vector[1] * Y) / vector[2]
        #    XYZ = np.c_[np.reshape(X, X.size), np.reshape(Y, Y.size), np.reshape(Z, Z.size)]
        #    dist = np.sqrt(np.sum((XYZ - point) ** 2, axis=1))
#
        #    try:
        #        brain_id = brain_atlas.regions.get(brain_atlas.get_labels(XYZ))['id']
        #        brain_colour = brain_atlas.regions.get(brain_atlas.get_labels(XYZ))['rgb']
        #    except Exception as err:
        #        print('errored')
        #        continue
        #    #brain_acronym = brain_atlas.regions.get(brain_atlas.get_labels(XYZ))['acronym']
        #    #brain_colour = brain_atlas.regions.get(brain_atlas.get_labels(XYZ))['rgb']
#
        #    color[iP, :, :, :] = np.reshape(brain_colour, (steps+1, steps+1, 3))
#
        #    dist_sorted = np.argsort(dist)
        #    brain_sorted = brain_id[dist_sorted]
        #    region_id.append(brain_sorted[0])
        #    diff_idx = np.where(brain_sorted != brain_sorted[0])[0]
        #    if np.any(diff_idx):
        #        min_dist.append(dist[dist_sorted[diff_idx[0]]] * 1e6)
        #    else:
        #        min_dist.append(200)
#
        #    # Now compute for the parents
        #    brain_parent = np.array([allen['parent_structure_id'][np.where(allen['id'] == br)[0][0]] for br
        #                    in brain_sorted])
        #    brain_parent[np.isnan(brain_parent)] = 0
#
        #    parent_id.append(brain_parent[0])
        #    parent_colour.append(allen['color_hex_triplet'][np.where(allen['id'] == brain_parent[0])[0][0]])
        #    parent_idx = np.where(brain_parent != brain_parent[0])[0]
#
        #    if np.any(parent_idx):
        #        parent_dist.append(dist[dist_sorted[parent_idx[0]]] * 1e6)
        #    else:
        #        parent_dist.append(200)
#
        #[_, _, region_colour, _] = ephys_align.get_histology_regions(xyz_samples, sampling_trk)
#
        #boundaries = np.where(np.diff(np.array(region_id)))[0]
        #bound = np.r_[0, boundaries + 1, len(region_id)]
#
        #all_y = []
        #all_x = []
        #for iB in np.arange(len(bound) - 1):
        #    y = sampling_trk[bound[iB]:(bound[iB + 1])]
        #    y = np.r_[y[0], y, y[-1]]
        #    x = min_dist[bound[iB]:(bound[iB + 1])]
        #    x = np.r_[0, x, 0]
        #    all_y.append(y)
        #    all_x.append(x)
#
        #parent_boundaries = np.where(np.diff(np.array(parent_id)))[0]
        #parent_bound = np.r_[0, parent_boundaries + 1, len(parent_id)]
#
        #all_parent_y = []
        #all_parent_x = []
        #all_parent_colour = []
        #for iB in np.arange(len(parent_bound) - 1):
        #    y = sampling_trk[parent_bound[iB]:(parent_bound[iB + 1])]
        #    y = np.r_[y[0], y, y[-1]]
        #    x = parent_dist[parent_bound[iB]:(parent_bound[iB + 1])]
        #    x = np.r_[0, x, 0]
        #    all_parent_y.append(y)
        #    all_parent_x.append(x)
        #    all_parent_colour.append(parent_colour[parent_bound[iB]])
#
        self.fig_hist = pg.PlotWidget()

        all_x, all_y, all_col = ephys_align.arrange_into_regions(sampling_trk, nearby_bounds['id'],
                                                                 nearby_bounds['dist'],
                                                                 nearby_bounds['col'])

        for iR, (x, y, c) in enumerate(zip(all_x, all_y, all_col)):
            if type(c) != str:
                c = '#FFFFFF'
            else:
                c = '#' + c
            colour = QtGui.QColor(c)
            plot = pg.PlotCurveItem()
            plot.setData(x=x, y=y*1000, fillLevel=10, fillOutline=True)
            plot.setBrush(colour)
            plot.setPen(colour)
            self.fig_hist.addItem(plot)

        [all_parent_x, all_parent_y,
         all_parent_col] = ephys_align.arrange_into_regions(sampling_trk, nearby_bounds['parent_id'],
                                                            nearby_bounds['parent_dist'],
                                                            nearby_bounds['parent_col'])

        for iR, (x, y, c) in enumerate(zip(all_parent_x, all_parent_y, all_parent_col)):
            if type(c) != str:
                c = '#FFFFFF'
            else:
                c = '#' + c
            colour = QtGui.QColor(c)
            colour.setAlpha(70)
            plot = pg.PlotCurveItem()
            plot.setData(x=x, y=y*1000, fillLevel=10, fillOutline=True)
            plot.setBrush(colour)
            plot.setPen(colour)
            self.fig_hist.addItem(plot)


        self.plot = pg.PlotItem()
        #imv = pg.ImageView(view=self.plot)
        #imv.setImage(color)
        #imv.sigTimeChanged.connect(self.update_text)
        layout_main = QtWidgets.QHBoxLayout()
        #layout_main.addWidget(imv)
        layout_main.addWidget(self.fig_hist)
        main_widget.setLayout(layout_main)

    def update_text(self, sig1):
        text = str(sig1) + ' ' + str(self.min_dist[sig1]) + ' ' + str(self.region[sig1][0]) + ' ' + str(self.region[sig1][1])
        self.plot.setTitle(text)



if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    mainapp = MainWindow()
    mainapp.show()
    app.exec_()


