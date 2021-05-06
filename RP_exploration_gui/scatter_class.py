from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np

class ScatterGroup:

    def __init__(self):

        self.fig_scatter = pg.PlotWidget(background='w')
        self.fig_scatter.setMouseTracking(True)
        self.fig_scatter.setLabel('bottom', 'Cluster Amplitude (uV)')
        self.fig_scatter.setLabel('left', 'Cluster Distance From Tip (uV)')

        self.scatter_plot = pg.ScatterPlotItem()
        self.scatter_text = pg.TextItem(color='k')
        self.scatter_text.hide()

        self.scatter_reset_button = QtWidgets.QPushButton('Reset Axis')
        self.scatter_reset_button.setFixedSize(70, 30)
        self.scatter_reset_button.setParent(self.fig_scatter)

        self.fig_scatter.addItem(self.scatter_plot)
        self.fig_scatter.addItem(self.scatter_text)
        self.fig_scatter.setFixedSize(400, 580)

        self.clust_amps = []
        self.clust_depths = []
        self.clust_color = []
        self.clust_ids = []
        self.point_pos = []
        self.point_pos_prev = []

    def reset(self):
        self.scatter_plot.setData()
        self.clust_amps = []
        self.clust_depths = []
        self.clust_color = []
        self.clust_ids = []
        self.point_pos = []
        self.point_pos_prev = []

    def populate(self, clust_amps, clust_depths, clust_ids, clust_color):
        self.clust_amps = clust_amps
        self.clust_depths = clust_depths
        self.clust_color = clust_color
        self.clust_ids = clust_ids
        self.scatter_plot.setData(x=self.clust_amps, y=self.clust_depths, size=8)
        self.scatter_plot.setBrush(clust_color)
        self.scatter_plot.setPen(QtGui.QColor(30, 50, 2))
        self.x_min = self.clust_amps.min() - 2
        self.x_max = self.clust_amps.max() - 2
        self.fig_scatter.setXRange(min=self.x_min, max=self.x_max)
        self.fig_scatter.setYRange(min=0, max=4000)
   
    def on_scatter_plot_clicked(self, point):
        self.point_pos = point[0].pos()
        clust = np.argwhere(self.scatter_plot.data['x'] == self.point_pos.x())[0][0]
        return clust

    def on_mouse_hover(self, pos):
        mouse_pos = self.scatter_plot.mapFromScene(pos)
        point = self.scatter_plot.pointsAt(mouse_pos)

        if len(point) != 0:
            point_pos = point[0].pos()
            clust = np.argwhere(self.scatter_plot.data['x'] == point_pos.x())[0][0]
            self.scatter_text.setText('Cluster no. ' + str(self.clust_ids[clust]))
            self.scatter_text.setPos(mouse_pos.x(), (mouse_pos.y()))
            self.scatter_text.show()
        else:
            self.scatter_text.hide()

    def on_scatter_plot_reset(self):
        self.fig_scatter.setXRange(min=self.x_min, max=self.x_max)
        self.fig_scatter.setYRange(min=0, max=4000)

    def update_scatter_icon(self, clust_prev):
        point = self.scatter_plot.pointsAt(self.point_pos)
        point[0].setBrush('b')
        point[0].setPen('b')
        point_prev = self.scatter_plot.pointsAt(self.point_pos_prev)
        point_prev[0].setPen('w')
        point_prev[0].setBrush(self.clust_color[clust_prev])

    def set_current_point(self, clust):
        self.point_pos.setX(self.clust_amps[clust])
        self.point_pos.setY(self.clust_depths[clust])

        return self.point_pos

    def update_prev_point(self):
        self.point_pos_prev.setX(self.point_pos.x())
        self.point_pos_prev.setY(self.point_pos.y())

        return self.point_pos_prev
 
    def initialise_scatter_index(self):
        self.point_pos = QtCore.QPointF()
        self.point_pos_prev = QtCore.QPointF()
        self.point_pos.setX(self.clust_amps[0])
        self.point_pos.setY(self.clust_depths[0])
        self.point_pos_prev.setX(self.clust_amps[0])
        self.point_pos_prev.setY(self.clust_depths[0])

        point = self.scatter_plot.pointsAt(self.point_pos)
        point[0].setBrush('b')
        point[0].setPen('b')

        return self.point_pos, self.point_pos_prev
