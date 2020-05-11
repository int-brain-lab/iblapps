import numpy as np
from brainbox.io.one import load_spike_sorting, load_channel_locations
from oneibl.one import ONE
import ibllib.atlas as atlas
from PyQt5 import QtWidgets, QtCore, QtGui
from pyqtgraph.widgets import MatplotlibWidget as ptl_widget
import pyqtgraph as pg

ONE_BASE_URL = "https://alyx.internationalbrainlab.org"
one = ONE(base_url=ONE_BASE_URL)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        #self.resize(1000, 800)
        main_widget = QtGui.QWidget()
        self.setCentralWidget(main_widget)
        main_widget_layout = QtWidgets.QHBoxLayout()

        eids = one.search(subject='ZM_2406', date='2019-11-12', number=1, task_protocol='ephys')
        eid = eids[0]
        probe_id = 0

        dtypes_extra = [
            'spikes.depths',
            'spikes.amps',
            'clusters.peakToTrough',
            'channels.localCoordinates'
        ]
        spikes, _ = load_spike_sorting(eid=eid, one=one, dataset_types=dtypes_extra)
        probe_label = [key for key in spikes.keys() if int(key[-1]) == probe_id][0]

        spikes = spikes[probe_label]

        size = np.random.rand((100))*10

        self.figure = pg.PlotWidget(background='w')
        scatter = pg.ScatterPlotItem()
        scatter.setData(x=spikes.times[:100], y=spikes.depths[:100], size=size)

        self.figure.addItem(scatter)
        #self.figure.setYRange(min=-1000, max=5000)

        main_widget_layout.addWidget(self.figure)

        main_widget.setLayout(main_widget_layout)
    





if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    mainapp = MainWindow()
    mainapp.show()
    app.exec_()
