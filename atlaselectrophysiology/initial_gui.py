from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.exporters

#Import custom QtGui.QWidget that allows keyPressEvents
from CustomEventWidget import CustomEventWidget

import numpy as np
#import load_data as ld
import load_data as ld
from random import randrange


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.resize(1000, 800)
        main_widget = CustomEventWidget()
        self.setCentralWidget(main_widget)
        main_widget_layout = QtWidgets.QHBoxLayout()
        
        #Load the data
        data = ld.LoadData('ZM_2407', '2019-11-07', probe_id=0)
        self.scatter_data = data.get_scatter_data()
        histology_data = data.get_histology_data()
        self.amplitude_data = data.get_amplitude_data()

      
        #Menu options --> Still need to work on this
        menuBar = QtWidgets.QMenuBar(self)
        menu_options = menuBar.addMenu("Plot Options")

        menuBar.setGeometry(QtCore.QRect(0, 0, 1002, 22))

        menu_options.addAction('Scatter Plot')
        menu_options.addAction('Depth Plot')
        menu_options.addAction('LFP Plot')
        menuBar.triggered.connect(self.on_menu_clicked)

        self.setMenuBar(menuBar)

        #Holds the infiniteLine objects added to plot
        self.added_lines = []


        self.fig_data = pg.PlotWidget(background='w')
        self.fig_data.setMouseEnabled(x=False, y=False)
        self.fig_data.scene().sigMouseClicked.connect(self.on_mouse_double_clicked)
        self.fig_data.setFixedSize(800, 800)
        
        #Initialise with depth scatter plot
        self.current_plot = []
        self.current_plot = self.plot_scatter()



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

        main_widget_layout.addWidget(self.fig_data)
        main_widget_layout.addWidget(self.fig_histology)
        main_widget.setLayout(main_widget_layout)


    def plot_scatter(self):
        
        connect = np.zeros(len(self.scatter_data['times']), dtype=int)
        scatter_plot = pg.PlotDataItem()
        scatter_plot.setData(x=self.scatter_data['times'], y=self.scatter_data['depths'], connect = connect, symbol = 'o',symbolSize = 2)
        self.fig_data.addItem(scatter_plot)
        self.fig_data.setYRange(min=0, max=3840)
        return scatter_plot

    
    def plot_bar(self):
    
        bar_plot = pg.BarGraphItem()
        bar_plot.setOpts(x=self.amplitude_data['bins'], height=self.amplitude_data['amps'], width=40, brush='b')
        bar_plot.rotate(90)
        self.fig_data.getPlotItem().invertX(True)
        self.fig_data.addItem(bar_plot)
        self.fig_data.setYRange(min=0, max=3840)
        
        return bar_plot
    
    def plot_image(self):
        image_plot = pg.ImageItem()
        image_plot.setImage(self.amplitude_data['corr'])
        self.fig_data.addItem(image_plot)
  
        return image_plot
   

    def on_menu_clicked(self, action):
        self.fig_data.removeItem(self.current_plot)
        if action.text() == 'Scatter Plot':
            self.current_plot = self.plot_scatter()
        if action.text() == 'Depth Plot':
            self.current_plot = self.plot_bar()
        if action.text() == 'LFP Plot':
            self.current_plot = self.plot_image()
        #print(action.text())
    

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
            #pos = self.scatter_plot.mapFromScene(event.scenePos())
            pos = self.current_plot.mapFromScene(event.scenePos())
            #print(pos)
            #pos_atlas = self.fig_histology.mapFromScene(event.scenePos())

            #print(pos_atlas)
            marker, pen = self.create_line_style()
            line_scat = pg.InfiniteLine(pos=pos.y(), angle=0, pen=pen, movable=True)
            #line_scat.addMarker(marker[0], position=0.98, size=marker[1]) #requires latest pyqtgraph
            line_hist = pg.InfiniteLine(pos=pos.y(), angle=0, pen=pen, movable=True)
            #line_hist.addMarker(marker[0], position=0.98, size=marker[1]) #requires latest pyqtgraph
            self.fig_histology.addItem(line_scat)
            #self.fig_scatter.addItem(line_hist)
            self.fig_data.addItem(line_hist)

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

    def on_enter_clicked(self, event):
        if event == QtCore.Qt.Key_Return:
            if len(self.added_lines) >= 3:
                scatter_line_pos = [line[0].pos().y() for line in self.added_lines]
                print(scatter_line_pos)
                hist_line_pos = [line[1].pos().y() for line in self.added_lines]
                print(hist_line_pos)
                self.z = np.polyfit(scatter_line_pos, hist_line_pos, 1)
                print(z)
            else:
                print('need to add 3 lines')
    
    def create_line_style(self):
        #Create random choice of line colour and style for infiniteLine
        markers = [['o', 10], ['v', 15]]
        mark = markers[randrange(len(markers))]
        colours = ['#000000', '#cc0000', '#6aa84f', '#1155cc', '#a64d79'] 
        style = [QtCore.Qt.SolidLine, QtCore.Qt.DashLine, QtCore.Qt.DashDotLine]
        col = QtGui.QColor(colours[randrange(len(colours))])
        sty = style[randrange(len(style))]
        pen = pg.mkPen(color=col, style=sty, width=3)
        return mark, pen

    #def update_boundaries(self):




if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    mainapp = MainWindow()
    mainapp.show()
    app.exec_()
