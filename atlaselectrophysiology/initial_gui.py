from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.exporters
from pathlib import Path

import numpy as np
import load_data as ld


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.resize(1000, 800)
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_widget_layout = QtWidgets.QHBoxLayout()

        eid, one = ld.get_session('ZM_2407', '2019-11-07')
        scatter_data = ld.get_scatter_data(eid, one, probe_id=0)
        histology_data = ld.get_histology_data(eid, one, probe_id=0)


        #Create scatter plot (for now only plot first 60s)
        #--> to do: functionality to scroll through session
        t = len(np.where(scatter_data['times'] < 60)[0])
        fig_scatter = pg.PlotWidget(background='w')
        fig_scatter.setFixedSize(800, 800)
        scatter_plot = pg.ScatterPlotItem()
        scatter_plot.setData(x=scatter_data['times'][0:t], y=scatter_data['depths'][0:t], size=2,
                             color='k')
        fig_scatter.addItem(scatter_plot)

        #Create histology figure
        #--> to do: incorporate actual data into plot
        #--> to do: add unique colours and y-label for each region
        fig_histology = pg.PlotWidget(background='w')
        fig_histology.setFixedSize(400, 800)

        for bound in histology_data['boundaries']:
            x, y = self.create_data(bound, histology_data['chan_int'])
            curve_item = pg.PlotCurveItem()
            curve_item.setData(x=x, y=y, fillLevel=50)
            curve_item.setBrush('b')
            fig_histology.addItem(curve_item)

        main_widget_layout.addWidget(fig_scatter)
        main_widget_layout.addWidget(fig_histology)
        main_widget.setLayout(main_widget_layout)

    def create_data(self, bound, chan_int):

        y = np.arange(bound[0], bound[1] + 1, chan_int, dtype=int)
        x = np.ones(len(y), dtype=int)
        x = np.insert(x, 0, 0)
        x = np.append(x, 0)
        y = np.insert(y, 0, bound[0])
        y = np.append(y, bound[1])

        return x, y


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    mainapp = MainWindow()
    mainapp.show()
    app.exec_()
