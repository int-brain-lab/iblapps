from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

import numpy as np
#import load_data as ld
import atlaselectrophysiology.load_data as ld
from random import randrange




class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.resize(1000, 800)
        #main_widget = CustomEventWidget()
        main_widget = QtGui.QWidget()
        self.setCentralWidget(main_widget)
        main_widget_layout = QtWidgets.QHBoxLayout()

        self.idx = 0
        self.lines = np.empty((0, 2))
        self.points = np.empty((0, 1))

        data = ld.LoadData('ZM_2407', '2019-11-07', probe_id=0)
        self.sdata = data.get_scatter_data()
        self.hdata = data.get_histology_data()

        #Things that are used many times
        self.probe_pen = pg.mkPen(color='k', style=QtCore.Qt.DotLine, width=2)
        self.probe_tip = 0
        self.probe_top = 3840
        self.probe_extra = 500
        self.x = np.arange(self.probe_tip, self.probe_top, 20)
        self.x_orig = self.x

        #Create data figure
        self.fig_d = pg.PlotWidget(background='w')
        self.fig_d.setMouseEnabled(x=False, y=False)
        self.fig_d.scene().sigMouseClicked.connect(self.on_mouse_double_clicked)
        self.fig_d.setFixedSize(800, 800)
        self.fig_d.addLine(y=self.probe_tip, pen=self.probe_pen)
        self.fig_d.addLine(y=self.probe_top, pen=self.probe_pen)

        #Initialise with scatter plot
        self.current_plot = self.plot_scatter()


        #Create histology figure
        self.fig_h = pg.PlotWidget(background='w')
        self.fig_h.scene().sigMouseClicked.connect(self.on_mouse_double_clicked)
        self.fig_h.setMouseEnabled(x=False, y=False)
        self.fig_h.setFixedSize(300, 800)
        self.fig_h.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top + self.probe_extra)
        self.plot_histology(self.fig_h)
 
        #Create reference histology figure
        self.fig_href = pg.PlotWidget(background='w')
        self.fig_href.setMouseEnabled(x=False, y=False)
        self.fig_href.setFixedSize(300, 800)
        self.fig_href.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top + self.probe_extra)

        self.plot_histology(self.fig_href)

        #Create figure showing fit line
        self.fig_f = pg.PlotWidget(background='w')
        self.fig_f.setMouseEnabled(x=False, y=False)
        self.fig_f.setFixedSize(300, 300)
        self.fig_f.setYRange(min=self.probe_tip, max=self.probe_top)
        self.fig_f.setXRange(min=self.probe_tip, max=self.probe_top)
        plot = pg.PlotCurveItem()
        plot.setData(x=self.x, y=self.x, pen=self.probe_pen)
        self.plot_cum = pg.PlotCurveItem()
        self.fig_f.addItem(plot)
        self.fig_f.addItem(self.plot_cum)

        
        main_widget_layout.addWidget(self.fig_d)
        main_widget_layout.addWidget(self.fig_h)
        main_widget_layout.addWidget(self.fig_href)
        main_widget_layout.addWidget(self.fig_f)
        main_widget.setLayout(main_widget_layout)
    


    def plot_histology(self, fig):

        axis = fig.plotItem.getAxis('left')
        axis.setTicks([self.hdata['axis_label'][self.idx]])
        axis.setPen('k')

        for ir, reg in enumerate(self.hdata['region'][self.idx]):
            x, y = self.create_hdata(reg, self.hdata['chan_int'])
            curve_item = pg.PlotCurveItem()
            curve_item.setData(x=x, y=y, fillLevel=50)
            colour = QtGui.QColor(*self.hdata['colour'][ir])
            curve_item.setBrush(colour)
            fig.addItem(curve_item)

        fig.addLine(y=self.probe_tip, pen=self.probe_pen)
        fig.addLine(y=self.probe_top, pen=self.probe_pen)
    
    def create_hdata(self, reg, chan_int):
        
        y = np.arange(reg[0], reg[1] + chan_int, chan_int, dtype=int)
        x = np.ones(len(y), dtype=int)
        x = np.insert(x, 0, 0)
        x = np.append(x, 0)
        y = np.insert(y, 0, reg[0])
        y = np.append(y, reg[1])

        return x, y
    
    def plot_fit(self):
        y = self.fit(self.x)
        plot = pg.PlotCurveItem()
        plot.setData(x=self.x, y=y, pen='k')
        self.plot_cum.setData(x=self.x_orig, y=y, pen='b')
        self.fig_f.addItem(plot)

        coeff_rel = np.polyfit(self.x_orig, y, 1)
        print(coeff_rel)
        self.x = y


        #coeff_rel = np.polyfit(self.x, y, 1)
        #print(coeff_rel)

    def plot_scatter(self):
            
        connect = np.zeros(len(self.sdata['times']), dtype=int)
        plot = pg.PlotDataItem()
        plot.setData(x=self.sdata['times'], y=self.sdata['depths'], connect = connect, symbol = 'o',symbolSize = 2)
        self.fig_d.addItem(plot)
        self.fig_d.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top + self.probe_extra)
        return plot

    def scale_hdata(self):

        line_pos_d = [line[0].pos().y() for line in self.lines]
        line_pos_h = [line[1].pos().y() for line in self.lines]
        coeff = np.polyfit(line_pos_d, line_pos_h, 1)
        print(coeff)
        self.fit = np.poly1d(coeff)

        idx_n = np.mod(self.idx, 4) + 1
        print(idx_n)
        for ir, reg in enumerate(self.hdata['region'][self.idx]):
            new_reg = self.fit(reg)
            self.hdata['region'][idx_n, ir, :] = new_reg
            self.hdata['axis_label'][idx_n, ir, :] = (np.mean(new_reg), self.hdata['label'][ir][0])
        
        self.idx = idx_n

    def remove_lines(self):
        for idx, lines in enumerate(self.lines):
            self.fig_d.removeItem(lines[0])
            self.fig_h.removeItem(lines[1])
            self.fig_f.removeItem(self.points[idx][0])
        self.lines = np.empty((0,2))
        self.points = np.empty((0,1))
        
    def keyPressEvent(self, event):

        if event.key() == QtCore.Qt.Key_Return:
            self.scale_hdata()
            self.plot_histology(self.fig_h)
            self.plot_fit()
            self.remove_lines()
        
        if event.key() == QtCore.Qt.Key_Left:
            #also need case for 0
            if self.idx == 1:
                self.idx = 4
            else:
                self.idx -= 1
            
            self.plot_histology(self.fig_h)
  
        if event.key() == QtCore.Qt.Key_Right:
            if self.idx == 4:
                self.idx = 1
            else:
                self.idx += 1

            self.plot_histology(self.fig_h)

    def on_mouse_double_clicked(self, event):
        if event.double():
            pos = self.current_plot.mapFromScene(event.scenePos())
            marker, pen, brush = self.create_line_style()
            line_d = pg.InfiniteLine(pos=pos.y(), angle=0, pen=pen, movable=True)
            line_d.sigPositionChangeFinished.connect(self.update_fit)
            #line_scat.addMarker(marker[0], position=0.98, size=marker[1]) #requires latest pyqtgraph
            line_h = pg.InfiniteLine(pos=pos.y(), angle=0, pen=pen, movable=True)
            line_h.sigPositionChangeFinished.connect(self.update_fit)
            point = pg.PlotDataItem()
            point.setData(x=[line_d.pos().y()], y=[line_h.pos().y()], symbolBrush=brush, symbol = 'o',symbolSize = 10)

            self.fig_f.addItem(point)
            
            #line_hist.addMarker(marker[0], position=0.98, size=marker[1]) #requires latest pyqtgraph
            self.fig_d.addItem(line_d)
            self.fig_h.addItem(line_h)
            self.lines = np.vstack([self.lines, [line_d, line_h]])
            self.points = np.vstack([self.points, point])
 


    
    def update_fit(self, line):
        #idx = [idx for idx, val in enumerate(lines) if any(x ==4 for x in val)]
        #idx = [idx for idx, val in enumerate(self.lines) if line == *val]
        idx = np.where(self.lines == line)[0][0]
        self.points[idx][0].setData(x=[self.lines[idx][0].pos().y()], y=[self.lines[idx][1].pos().y()])

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
        


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    mainapp = MainWindow()
    mainapp.show()
    app.exec_()
