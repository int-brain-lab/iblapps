from PyQt5 import QtWidgets, QtCore, QtGui
from pyqtgraph.widgets import MatplotlibWidget as matplot
import pyqtgraph as pg

import numpy as np
import atlaselectrophysiology.load_data as ld
from random import randrange


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.resize(1600, 800)
        self.setWindowTitle('Electrophysiology Atlas')
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        self.init_variables()
        self.init_layout()

        subj = 'ZM_2406'
        date = '2019-11-12'
        probe = 0
        data = ld.LoadData(subj, date, probe_id=probe)
        self.sdata = data.get_scatter_data()
        self.hist_data = data.get_histology_data()
        self.title_string.setText(f"{subj} {date} probe0{probe}")
        
        self.brain_atlas = data.brain_atlas
        self.probe_coord = data.probe_coord
                
        #Initialise with scatter plot
        self.data_plot = self.plot_scatter()
        self.plot_histology(self.fig_hist)
        self.plot_histology(self.fig_hist_ref)
        self.brain_atlas.plot_cslice(self.probe_coord[0, 1], volume='annotation', ax=self.fig_slice_ax)
        self.fig_slice_ax.plot(self.probe_coord[:, 0] * 1e6, self.probe_coord[:, 2] * 1e6, 'k*')
        self.fig_slice.draw()

    def init_layout(self):
        #Main Widget    
        main_widget = QtGui.QWidget()
        self.setCentralWidget(main_widget)

        #Make menu bar

        #Fit associated items
        hlayout_fit = QtWidgets.QHBoxLayout()
        self.fit_button = QtWidgets.QPushButton('Fit', font=self.font)
        self.fit_button.clicked.connect(self.fit_button_pressed)
        self.fit_string = QtWidgets.QLabel(font=self.font)
        self.idx_string = QtWidgets.QLabel(font=self.font)
        hlayout_fit.addWidget(self.fit_button, stretch=1)
        hlayout_fit.addWidget(self.fit_string, stretch=3)

        #Idx associated items
        hlayout_button = QtWidgets.QHBoxLayout()
        hlayout_string = QtWidgets.QHBoxLayout()
        self.next_button = QtWidgets.QPushButton('Next', font=self.font)
        self.next_button.clicked.connect(self.next_button_pressed)
        self.prev_button = QtWidgets.QPushButton('Previous', font=self.font)
        self.prev_button.clicked.connect(self.prev_button_pressed)
        hlayout_button.addWidget(self.prev_button)
        hlayout_button.addWidget(self.next_button)
        self.idx_string = QtWidgets.QLabel(font=self.font)
        self.tot_idx_string = QtWidgets.QLabel(font=self.font)
        self.reset_button = QtWidgets.QPushButton('Reset', font=self.font)
        self.reset_button.clicked.connect(self.reset_button_pressed)
        hlayout_string.addWidget(self.idx_string)
        hlayout_string.addWidget(self.tot_idx_string)
        vlayout = QtWidgets.QVBoxLayout()
        vlayout.addLayout(hlayout_button)
        vlayout.addLayout(hlayout_string)
        vlayout.addWidget(self.reset_button)
        
        #Title item
        self.title_string = QtWidgets.QLabel(font=self.title_font)
        self.update_string()


        #Figure items
        self.init_figures()

        #Main Widget    
        main_widget = QtGui.QWidget()
        self.setCentralWidget(main_widget)
        #Add everything to the main widget
        layout_main = QtWidgets.QGridLayout()
        layout_main.addWidget(self.fig_data, 0, 0, 10, 4)
        layout_main.addWidget(self.fig_hist, 0, 4, 10, 2)
        layout_main.addWidget(self.fig_hist_ref, 0, 6, 10, 2)
        layout_main.addWidget(self.title_string, 0, 8, 1, 2)
        #layout_main.addWidget(self.fig_slice, 1, 8, 3, 2)
        layout_main.addLayout(hlayout_fit, 2, 8, 1, 2)
        layout_main.addWidget(self.fig_fit, 3, 8, 5, 2)
        layout_main.addLayout(vlayout, 8, 8, 2, 2)
        layout_main.setColumnStretch(0, 4)
        layout_main.setColumnStretch(4, 2)
        layout_main.setColumnStretch(6, 2)
        layout_main.setColumnStretch(8, 2)
        #layout_main.setRowStretch(0, 1)
        #layout_main.setRowStretch(2, 1)
        #layout_main.setRowStretch(4, 1)
        #layout_main.setRowStretch(5, 3)
        #layout_main.setRowStretch(8, 2)

        main_widget.setLayout(layout_main)

    def init_variables(self):

        #Line styles and fonts
        self.kpen_dot = pg.mkPen(color='k', style=QtCore.Qt.DotLine, width=2)
        self.kpen_solid = pg.mkPen(color='k', style=QtCore.Qt.SolidLine, width=2)
        self.bpen_solid = pg.mkPen(color='b', style=QtCore.Qt.SolidLine, width=3)
        self.font = QtGui.QFont()
        self.font.setPointSize(12)
        self.title_font = QtGui.QFont()
        self.title_font.setPointSize(18)

        #Constant variables
        self.probe_tip = 0
        self.probe_top = 3840
        self.probe_extra = 500
        self.depth = np.arange(self.probe_tip, self.probe_top, 20)
        
        #Variables to keep track of number of fits (max 10)
        self.idx = 0
        self.total_idx = 0
        self.max_idx = 10

        #Variables to keep track of applied fits
        self.depth_fit = np.empty((11, len(self.depth)))
        self.depth_fit[0] = self.depth
        self.tot_fit = np.empty((self.max_idx + 1, 2))
        self.tot_fit[0] = [1, 0]

        #Variables to keep track of number of lines added
        self.lines = np.empty((0, 2))
        self.points = np.empty((0, 1))
    
    def init_figures(self):

        #This can be added as a seperate UI file at some point
        
        #Create data figure
        self.fig_data = pg.PlotWidget(background='w')
        self.fig_data.setMouseEnabled(x=False, y=False)
        self.fig_data.scene().sigMouseClicked.connect(self.on_mouse_double_clicked)
        self.fig_data.addLine(y=self.probe_tip, pen=self.kpen_dot)
        self.fig_data.addLine(y=self.probe_top, pen=self.kpen_dot)
        self.set_axis(self.fig_data)

        #Create histology figure
        self.fig_hist = pg.PlotWidget(background='w')
        self.fig_hist.scene().sigMouseClicked.connect(self.on_mouse_double_clicked)
        self.fig_hist.setMouseEnabled(x=False, y=False)
        self.fig_hist.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top + self.probe_extra)
        axis = self.fig_hist.plotItem.getAxis('bottom')
        axis.setTicks([[(0, ''), (0.5, ''), (1, '')]])
        axis.setLabel('')
 
        #Create reference histology figure
        self.fig_hist_ref = pg.PlotWidget(background='w')
        self.fig_hist_ref.setMouseEnabled(x=False, y=False)
        self.fig_hist_ref.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top + self.probe_extra)
        axis = self.fig_hist_ref.plotItem.getAxis('bottom')
        axis.setTicks([[(0, ''), (0.5, ''), (1, '')]])
        axis.setLabel('')

        #Create figure showing fit line
        self.fig_fit = pg.PlotWidget(background='w')
        self.fig_fit.setMouseEnabled(x=False, y=False)
        self.fig_fit.setYRange(min=self.probe_tip, max=self.probe_top + 200)
        self.fig_fit.setXRange(min=self.probe_tip, max=self.probe_top + 200)
        self.set_axis(self.fig_fit)
        plot = pg.PlotCurveItem()
        plot.setData(x=self.depth, y=self.depth, pen=self.kpen_dot)
        self.fit_plot = pg.PlotCurveItem(pen=self.kpen_solid)
        self.tot_fit_plot = pg.PlotCurveItem(pen=self.bpen_solid)
        self.fig_fit.addItem(plot)
        self.fig_fit.addItem(self.fit_plot)
        self.fig_fit.addItem(self.tot_fit_plot)

        #Create figure to show coronal slice and probe
        self.fig_slice = matplot.MatplotlibWidget()
        fig = self.fig_slice.getFigure()
        fig.canvas.toolbar.hide()
        self.fig_slice_ax = fig.gca()


    def set_axis(self, fig, label=None):
        if not label:
            label = ['', '']
        ax_x = fig.plotItem.getAxis('bottom')
        ax_x.setPen('k')
        ax_x.setLabel(label[0])
        ax_y = fig.plotItem.getAxis('left')
        ax_y.setPen('k')
        ax_y.setLabel(label[1])

##Plot functions
    def plot_histology(self, fig):

        axis = fig.plotItem.getAxis('left')
        axis.setTicks([self.hist_data['axis_label'][self.idx]])
        axis.setPen('k')

        for ir, reg in enumerate(self.hist_data['region'][self.idx]):
            x, y = self.create_hist_data(reg, self.hist_data['chan_int'])
            curve_item = pg.PlotCurveItem()
            curve_item.setData(x=x, y=y, fillLevel=50)
            colour = QtGui.QColor(*self.hist_data['colour'][ir])
            curve_item.setBrush(colour)
            fig.addItem(curve_item)

        fig.addLine(y=self.probe_tip, pen=self.kpen_dot)
        fig.addLine(y=self.probe_top, pen=self.kpen_dot)
    
    def create_hist_data(self, reg, chan_int):
        
        y = np.arange(reg[0], reg[1] + chan_int, chan_int, dtype=int)
        x = np.ones(len(y), dtype=int)
        x = np.insert(x, 0, 0)
        x = np.append(x, 0)
        y = np.insert(y, 0, reg[0])
        y = np.append(y, reg[1])

        return x, y

    def scale_hist_data(self):
    
        line_pos_h = [line[0].pos().y() for line in self.lines]
        line_pos_d = [line[1].pos().y() for line in self.lines]
        coeff = np.polyfit(line_pos_h, line_pos_d, 1)
        self.fit = np.poly1d(coeff)
        print(self.fit)
        for ir, reg in enumerate(self.hist_data['region'][self.idx - 1]):
            new_reg = self.fit(reg)
            self.hist_data['region'][self.idx, ir, :] = new_reg
            self.hist_data['axis_label'][self.idx, ir, :] = (np.mean(new_reg), self.hist_data['label'][ir][0])
        
        self.depth_fit[self.idx] = self.fit(self.depth_fit[self.idx - 1])
        
        
        self.tot_fit[self.idx] = np.polyfit(self.depth, self.depth_fit[self.idx], 1)
        print(self.tot_fit[self.idx])
        

    def plot_fit(self):
        if self.idx != 0:
            self.fit_plot.setData(x=self.depth_fit[self.idx - 1], y=self.depth_fit[self.idx])
            fit = np.poly1d(self.tot_fit[self.idx])
            self.tot_fit_plot.setData(x=self.depth, y=fit(self.depth))


        else:
            self.fit_plot.setData()
            self.tot_fit_plot.setData()

        self.update_string()
    
    def update_fit(self, line):
        idx = np.where(self.lines == line)[0][0]
        self.points[idx][0].setData(x=[self.lines[idx][0].pos().y()], y=[self.lines[idx][1].pos().y()])

    
    def plot_scatter(self):
            
        connect = np.zeros(len(self.sdata['times']), dtype=int)
        plot = pg.PlotDataItem()
        plot.setData(x=self.sdata['times'], y=self.sdata['depths'], connect = connect, symbol = 'o',symbolSize = 2)
        self.fig_data.addItem(plot)
        self.fig_data.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top + self.probe_extra)
        return plot

##Interaction functions
    def keyPressEvent(self, event):

        if event.key() == QtCore.Qt.Key_Return:
            self.fit_button_pressed()
        
        if event.key() == QtCore.Qt.Key_Left:
            self.prev_button_pressed()

        if event.key() == QtCore.Qt.Key_Right:
            self.next_button_pressed()

    def fit_button_pressed(self):
        if self.idx <= self.max_idx:
            self.idx += 1
            self.scale_hist_data()
            self.plot_histology(self.fig_hist)
            self.plot_fit()
            self.remove_lines_points()
            self.add_lines_points()
            self.total_idx = self.idx
            self.update_string()

    def next_button_pressed(self):
        if self.idx < self.total_idx:
            self.idx += 1
            self.plot_histology(self.fig_hist)
            self.plot_fit()
            self.update_string()

    def prev_button_pressed(self):
        if self.idx > 0:
            self.idx -= 1
            self.plot_histology(self.fig_hist)
            self.plot_fit()
            self.update_string()

    def reset_button_pressed(self):
        self.remove_lines_points()
        self.init_variables()
        self.plot_histology(self.fig_hist)
        self.tot_fit_plot.setData()
        self.fit_plot.setData()
        self.update_string()

    def on_mouse_double_clicked(self, event):
        if event.double():
            pos = self.data_plot.mapFromScene(event.scenePos())
            marker, pen, brush = self.create_line_style()
            line_d = pg.InfiniteLine(pos=pos.y(), angle=0, pen=pen, movable=True)
            line_d.sigPositionChangeFinished.connect(self.update_fit)
            #line_scat.addMarker(marker[0], position=0.98, size=marker[1]) #requires latest pyqtgraph
            line_h = pg.InfiniteLine(pos=pos.y(), angle=0, pen=pen, movable=True)
            line_h.sigPositionChangeFinished.connect(self.update_fit)
            point = pg.PlotDataItem()
            #point.setData(x=[line_d.pos().y()], y=[line_h.pos().y()], symbolBrush=brush, symbol = 'o',symbolSize = 10)
            point.setData(x=[line_h.pos().y()], y=[line_d.pos().y()], symbolBrush=brush, symbol = 'o',symbolSize = 10)
            self.fig_fit.addItem(point)
         
            #line_hist.addMarker(marker[0], position=0.98, size=marker[1]) #requires latest pyqtgraph
            self.fig_data.addItem(line_d)
            self.fig_hist.addItem(line_h)
            self.lines = np.vstack([self.lines, [line_h, line_d]])
            self.points = np.vstack([self.points, point])

    def remove_lines_points(self):
        for idx, lines in enumerate(self.lines):
            self.fig_hist.removeItem(lines[0])
            self.fig_data.removeItem(lines[1])
            self.fig_fit.removeItem(self.points[idx][0])
        #self.lines = np.empty((0, 2))
        #self.points = np.empty((0, 1))

    def add_lines_points(self):
        for idx, lines in enumerate(self.lines):
            self.fig_hist.addItem(lines[0])
            self.fig_data.addItem(lines[1])
            self.fig_fit.addItem(self.points[idx][0])
        #self.lines = np.empty((0, 2))
        #self.points = np.empty((0, 1))

    def create_line_style(self):
        #Create random choice of line colour and style for infiniteLine
        markers = [['o', 10], ['v', 15]]
        mark = markers[randrange(len(markers))]
        colours = ['#000000', '#cc0000', '#6aa84f', '#1155cc', '#a64d79'] 
        style = [QtCore.Qt.SolidLine, QtCore.Qt.DashLine, QtCore.Qt.DashDotLine]
        col = QtGui.QColor(colours[randrange(len(colours))])
        sty = style[randrange(len(style))]
        pen = pg.mkPen(color=col, style=sty, width=3)
        brush = pg.mkBrush(color=col)
        return mark, pen, brush
  
    def update_string(self):
            
        self.idx_string.setText(f"Current Index = {self.idx}")
        self.tot_idx_string.setText(f"Total Index = {self.total_idx}")
        self.fit_string.setText(f"Scale = {round(self.tot_fit[self.idx][0],2)}, Offset = {round(self.tot_fit[self.idx][1],2)}")
        


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    mainapp = MainWindow()
    mainapp.show()
    app.exec_()
