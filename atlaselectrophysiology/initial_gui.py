from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.exporters
from pathlib import Path

from CustomEventWidget import CustomEventWidget

import numpy as np
import load_data as ld


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.resize(1000, 800)
        self.menuBar
        main_widget = CustomEventWidget()
        self.setCentralWidget(main_widget)
        main_widget_layout = QtWidgets.QHBoxLayout()
        
        #Load the data
        eid, one = ld.get_session('ZM_2407', '2019-11-07')
        scatter_data, probe_label = ld.get_scatter_data(eid, one, probe_id=0)
        histology_data = ld.get_histology_data(eid, probe_label, one)

      
        #Menu options --> Still need to work on this
        menuBar = QtWidgets.QMenuBar(self)
        options = menuBar.addMenu("Plot Options")
        menuBar.setGeometry(QtCore.QRect(0, 0, 1002, 22))
        options.addAction('Scatter Plot')
        options.addAction('Depth Plot')
        self.setMenuBar(menuBar)

        #Holds the infiniteLine objects added to plot
        self.added_lines = []


        #Create scatter plot
        print(len(scatter_data['times']))
        self.fig_scatter = pg.PlotWidget(background='w')
        self.fig_scatter.scene().sigMouseClicked.connect(self.on_mouse_double_clicked)
        self.fig_scatter.setMouseEnabled(x=False, y=False)
        self.fig_scatter.setFixedSize(800, 800)
        connect = np.zeros(len(scatter_data['times']), dtype=int)
        self.scatter_plot = pg.PlotDataItem()
        self.scatter_plot.setData(x=scatter_data['times'], y=scatter_data['depths'], connect = connect, symbol = 'o',symbolSize = 2)
        #self.scatter_plot = pg.ScatterPlotItem()
        #self.scatter_plot.setData(x=scatter_data['times'][:], y=scatter_data['depths'][:], size = 2, color = 'k')
        self.fig_scatter.addItem(self.scatter_plot)

        #Create histology figure
        #--> to do: incorporate actual data into plot
        self.fig_histology = pg.PlotWidget(background='w')
        self.fig_histology.scene().sigMouseClicked.connect(self.on_mouse_double_clicked)
        self.fig_histology.setMouseEnabled(x=False, y=False)
        axis = self.fig_histology.plotItem.getAxis('left')
        axis.setTicks([histology_data['boundary_label']])
        axis.setPen('k')
        self.fig_histology.setFixedSize(400, 800)

        for idx, bound in enumerate(histology_data['boundary']):
            x, y = self.create_data(bound, histology_data['chan_int'])
            curve_item = pg.PlotCurveItem()
            curve_item.setData(x=x, y=y, fillLevel=50)
            colour = QtGui.QColor(histology_data['boundary_colour'][idx])
            curve_item.setBrush(colour)
            self.fig_histology.addItem(curve_item)

        #When you hit enter key and three Infinitelines added -> print y position of lines
        main_widget.keyPressed.connect(self.on_enter_clicked)

        main_widget_layout.addWidget(self.fig_scatter)
        main_widget_layout.addWidget(self.fig_histology)
        main_widget.setLayout(main_widget_layout)


    def create_data(self, bound, chan_int):
        y = np.arange(bound[0], bound[1] + chan_int, chan_int, dtype=int)
        x = np.ones(len(y), dtype=int)
        x = np.insert(x, 0, 0)
        x = np.append(x, 0)
        y = np.insert(y, 0, bound[0])
        y = np.append(y, bound[1])

        return x, y

    def on_mouse_double_clicked(self, event):
        if event.double():
            pos = self.scatter_plot.mapFromScene(event.scenePos())
            pos_atlas = self.fig_histology.mapFromScene(event.scenePos())
            line_scat = pg.InfiniteLine(pos=pos.y(), angle=0, pen='k', movable=True)
            line_hist = pg.InfiniteLine(pos=pos.y(), angle=0, pen='k', movable=True)
            self.fig_histology.addItem(line_scat)
            self.fig_scatter.addItem(line_hist)

            lines = [line_scat, line_hist]
            self.added_lines.append(lines)
      
            #print(pos)       
            #print(pos_atlas)
            #todo: map atlas y position to same coordinate system as scatter y position
            #then take click positions into 3 y values on each, and fit line to 3 points. 
            # [histology_data['boundary_label']]
            # [scatter_data['depths']]
            #a = pos.y() b=pos_atlas.y()
            #z = np.polyfit(a,b,1)

    def on_enter_clicked(self):
        if len(self.added_lines) == 3:
            scatter_line_pos = [line[0].pos().y() for line in self.added_lines]
            print(scatter_line_pos)
            hist_line_pos = [line[1].pos().y() for line in self.added_lines]
            print(hist_line_pos)
        else:
            print('need to add 3 lines')

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    mainapp = MainWindow()
    mainapp.show()
    app.exec_()
