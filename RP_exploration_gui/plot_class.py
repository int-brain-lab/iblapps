from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.dockarea
import time


class PlotGroup:

    def __init__(self):

        self.create_dock_area()

        self.fig1_peth = PlotTemplate('Time After Event (s)', 'Firing Rate (Hz)')
        self.fig1_button = QtWidgets.QPushButton('Add')
        self.fig1_button.setParent(self.fig1_peth.fig)
        self.fig1_button.setFixedSize(30, 30)
        self.fig1_area.addWidget(self.fig1_peth.fig)
        
        self.fig2_peth = PlotTemplate('Time After Event (s)', 'Firing Rate (Hz)')
        self.fig2_button = QtWidgets.QPushButton('Add')
        self.fig2_button.setParent(self.fig2_peth.fig)
        self.fig2_button.setFixedSize(30, 30)
        self.fig2_area.addWidget(self.fig2_peth.fig)

        self.fig3_raster = ScatterTemplate('Time After Event (s)', 'No. of Trials')
        self.fig3_button = QtWidgets.QPushButton('Reset Axis')
        self.fig3_button.setParent(self.fig3_raster.fig)
        self.fig3_button.setFixedSize(60, 30)
        self.fig3_area.addWidget(self.fig3_raster.fig)

        self.fig4_raster = ScatterTemplate('Time After Event (s)', 'No. of Trials')
        self.fig4_button = QtWidgets.QPushButton('Reset Axis')
        self.fig4_button.setParent(self.fig4_raster.fig)
        self.fig4_button.setFixedSize(60, 30)
        self.fig4_area.addWidget(self.fig4_raster.fig)
        
        self.fig5_autocorr = BarTemplate('Time (ms)', 'Corr')
        self.fig5_area.addWidget(self.fig5_autocorr.fig)

        self.fig6_template = PlotTemplate('Time (ms)', 'V (uV)')
        self.fig6_area.addWidget(self.fig6_template.fig)

        self.fig7_waveform = PlotTemplate('Time(ms)', 'V (uV)')
        self.fig7_area.addWidget(self.fig7_waveform.fig)

        self.plot_status = {}
        self.y_range = {}
        self.case_prev = []
        self.raster_x = []
        self.init_flag = False

    def create_dock_area(self):
        self.fig_area = pg.dockarea.DockArea()
        self.fig1_area = pg.dockarea.Dock('Figure 1: Go Cue PETH', autoOrientation='horizontal')
        self.fig2_area = pg.dockarea.Dock('Figure 2: Feedback PETH', autoOrientation='horizontal')
        self.fig3_area = pg.dockarea.Dock('Figure 3: Go Cue Raster', autoOrientation='horizontal')
        self.fig4_area = pg.dockarea.Dock('Figure 4: Feedback Raster', autoOrientation='horizontal')
        self.fig5_area = pg.dockarea.Dock('Figure 5: Autocorrelogram', autoOrientation='horizontal')
        self.fig6_area = pg.dockarea.Dock('Figure 6: Template Waveform', autoOrientation='horizontal')
        self.fig7_area = pg.dockarea.Dock('Figure 7: Waveform', autoOrientation='horizontal')

        self.fig_area.addDock(self.fig2_area, 'top')
        self.fig_area.addDock(self.fig1_area, 'left', self.fig2_area)
        self.fig_area.addDock(self.fig3_area, 'bottom', self.fig1_area)
        self.fig_area.addDock(self.fig4_area, 'bottom', self.fig2_area)
        self.fig_area.addDock(self.fig5_area, 'right')
        self.fig_area.addDock(self.fig6_area, 'bottom', self.fig5_area)
        self.fig_area.addDock(self.fig7_area, 'bottom', self.fig6_area)

        self.fig3_area.setStretch(x=1, y=18)
        self.fig4_area.setStretch(x=1, y=18)
        
        self.fig5_area.setStretch(x=7, y=1)
        self.fig6_area.setStretch(x=7, y=1)
        self.fig7_area.setStretch(x=7, y=1)
    
    def reset(self):
       
        self.fig1_peth.fig.plotItem.clear()
        self.fig2_peth.fig.plotItem.clear()
        self.plot_status = {}
        self.y_range = {}
        self.fig3_raster.reset()
        self.fig4_raster.reset()
        self.fig5_autocorr.reset()
        self.fig6_template.reset()
        self.fig7_waveform.reset()


    def change_rasters(self, data, clust, trials, case, sort_method):
        option = trials[case]
        trial_id = option[sort_method]['trials']    

        if len(trial_id) != 0:
            self.fig3_raster.remove_lines()
            [x, y, n_trials] = data.compute_rasters('goCue_times', clust, trial_id)
            if len(x) != 0:
                self.fig3_raster.plot(x, y, n_trials, self.raster_x)
                self.fig3_raster.add_lines(option[sort_method]['lines'], option[sort_method]['linecolours'], option[sort_method]['text'])
                self.fig3_raster.fig.plotItem.addLine(x = 0, pen = 'k')
            else:
                self.fig3_raster.reset()
            self.fig4_raster.remove_lines()
            [x, y, n_trials] = data.compute_rasters('feedback_times', clust, trial_id)
            if len(x) != 0:
                self.fig4_raster.plot(x, y, n_trials, self.raster_x)
                self.fig4_raster.add_lines(option[sort_method]['lines'], option[sort_method]['linecolours'], option[sort_method]['text'])
                self.fig4_raster.fig.plotItem.addLine(x = 0, pen = 'k')
            else:
                self.fig4_raster.reset()
        else:
            self.fig3_raster.reset()
            self.fig4_raster.reset()



    def plot_psth(self, data, clust, trials, case, sort_method):
        option = trials[case]
        trial_id = option[sort_method]['trials']

        if len(trial_id) != 0:
            plot_item1 = self.fig1_peth.plot_item()
            [x1, y1, errbar1] = data.compute_peth('goCue_times', clust, trial_id)
            y_range1 = [(y1-errbar1).min(), (y1 + errbar1).max()]
            [x2, y2, errbar2] = data.compute_peth('feedback_times', clust, trial_id)
            y_range2 = [(y2-errbar2).min(), (y2 + errbar2).max()]
            self.y_range[case] = [y_range1, y_range2]
            if self.init_flag is False:
                self.raster_x = x1
                self.init_flag = True
            
            plot_item1 = self.fig1_peth.plot_item()
            self.fig1_peth.plot_errbar_hold(x1, y1, errbar1, plot_item1, option['colour'], option['fill'], option['linestyle'], self.y_range, 0)
            self.fig1_peth.fig.plotItem.addLine(x = 0, pen = 'm')
 
            plot_item2 = self.fig2_peth.plot_item()
            self.fig2_peth.plot_errbar_hold(x2, y2, errbar2, plot_item2, option['colour'], option['fill'], option['linestyle'], self.y_range, 1)
            self.fig2_peth.fig.plotItem.addLine(x = 0, pen = 'm')

            plots = [plot_item1, plot_item2]
            self.plot_status[case] = plots
        else:
            plot_item1 = self.fig1_peth.plot_item()
            plot_item2 = self.fig2_peth.plot_item()
            plots = [plot_item1, plot_item2]
            self.plot_status[case] = plots
            y_range1 = [0, 0]
            y_range2 = [0, 0]
            self.y_range[case] = [y_range1, y_range2]

    
    def change_all_plots_final(self, data, clust, trials, case, sort_method, hold):
        bla = self.plot_status.keys()
        if len(bla) == 0:
            self.plot_psth(data, clust, trials, case, sort_method)
        else:
            for key in bla:
                self.remove_plot_item(key)
                self.plot_psth(data, clust, trials, key, sort_method)
        
        if hold is True:
            self.change_rasters(data, clust, trials, 'all', sort_method)
        else:
            self.change_rasters(data, clust, trials, case, sort_method)
        
        [x, y] = data.compute_autocorr(clust)
        self.fig5_autocorr.plot(x, y, data.autocorr_bin)
        [x, y] = data.compute_template(clust)
        self.fig6_template.plot_line(x, y)
        self.fig7_waveform.reset()
        self.case_prev = case
 

    def change_plots_final(self, data, clust, trials, case, sort_method, hold):
        if hold is False:
            self.remove_plot_item(self.case_prev)

        self.plot_psth(data, clust, trials, case, sort_method)

        if hold is True:
            self.change_rasters(data, clust, trials, 'all', sort_method)
        else:
            self.change_rasters(data, clust, trials, case, sort_method)

        [x, y] = data.compute_autocorr(clust)
        self.fig5_autocorr.plot(x, y, data.autocorr_bin)
        [x, y] = data.compute_template(clust)
        self.fig6_template.plot_line(x, y)
        self.fig7_waveform.reset()

        self.case_prev = case

    def remove_plot_item(self, case):
        plot_item = self.plot_status[case]

        self.fig1_peth.remove_item(plot_item[0])
        self.fig2_peth.remove_item(plot_item[1])

        self.plot_status.pop(case)
        self.y_range.pop(case)

        #return self.plot_status

class PlotTemplate:
    def __init__(self, xlabel, ylabel):
        self.fig = pg.PlotWidget(background='w')
        self.fig.setMouseEnabled(x=False, y=False)
        self.fig.setLabel('bottom', xlabel)
        self.fig.setLabel('left', ylabel)
        self.plot = pg.PlotCurveItem()
        self.plot_lb = pg.PlotCurveItem()
        self.plot_ub = pg.PlotCurveItem()
        self.plot_fill = pg.FillBetweenItem()
        self.fig.addItem(self.plot)
        self.fig.addItem(self.plot_fill)
        #self.fig.plotItem.legend

    def plot_item(self):
        plot_items = []
        #plot
        plot_items.append(pg.PlotCurveItem())
        #plot_lb
        plot_items.append(pg.PlotCurveItem())
        #plot_ub
        plot_items.append(pg.PlotCurveItem())
        #plot_fill
        plot_items.append(pg.FillBetweenItem())
        self.fig.addItem(plot_items[0])
        self.fig.addItem(plot_items[3])

        return plot_items
    
    def reset(self):
        self.plot.setData()
        self.plot_lb.setData()
        self.plot_ub.setData()
        self.plot_fill.setCurves(self.plot_ub, self.plot_lb)
    
    def remove_item(self, plot_item):
        self.fig.removeItem(plot_item[0])
        self.fig.removeItem(plot_item[3])

    
    def plot_errbar(self, x, y, errbar, colour, fill, linestyle):
        self.plot.setData(x=x, y=y)
        #self.plot.setBrush(colour)
        self.plot.setPen(colour)
        self.plot_lb.setData(x=x, y=(y - errbar))
        self.plot_ub.setData(x=x, y=(y + errbar))
        self.plot_fill.setCurves(self.plot_ub, self.plot_lb)
        fill.setAlpha(50)
        self.plot_fill.setBrush(fill)
        self.fig.setXRange(min=x.min(), max=x.max())
        self.fig.setYRange(min=0.95 * (y - errbar).min(), max=1.05 * (y + errbar).max())
    
    def plot_errbar_hold(self, x, y, errbar, plot_item, colour, fill, linestyle, y_range, id):
        min_val = min([val[id][0] for val in y_range.values()])
        max_val = max([val[id][1] for val in y_range.values()])
        case_min = [idx for idx, val in y_range.items() if val[id][0] == min_val][0]
        case_max = [idx for idx, val in y_range.items() if val[id][1] == max_val][0]
        y_min = y_range[case_min][id][0]
        y_max = y_range[case_max][id][1]
        
        plot_item[0].setData(x=x, y=y)
        #plot_item[0].setBrush('k')
        #plot_item[0].setBrush(colour)
        plot_item[0].setPen(colour)
        plot_item[1].setData(x=x, y=(y - errbar))
        plot_item[2].setData(x=x, y=(y + errbar))
        #plot_item[3].setCurves(self.plot_ub, self.plot_lb)
        plot_item[3].setCurves(plot_item[2], plot_item[1])
        fill.setAlpha(50)
        plot_item[3].setBrush(fill)
        #plot_item[3].setBrush('b')
        self.fig.setXRange(min=x.min(), max=x.max())
        self.fig.setYRange(min=0.95 * y_min, max=1.05 * y_max)

        return y_range

    def plot_line(self, x, y):
        self.plot.setData(x=x, y=y)
        self.plot.setPen('b')
        self.fig.setXRange(min=x.min(), max=x.max())
        self.fig.setYRange(min=0.95 * y.min(), max=1.05 * y.max())


class ScatterTemplate:
    def __init__(self, xlabel, ylabel):

        self.viewbox = pg.ViewBox()
        self.viewbox.setMouseMode(self.viewbox.RectMode)
        self.fig = pg.PlotWidget(background='w', viewBox=self.viewbox)
        self.fig.setMouseEnabled(x=False, y=False)
        self.fig.setMouseTracking(True)
        #self.fig.setMouseEnabled(y=False)
        self.fig.setLabel('bottom', xlabel)
        self.fig.setLabel('left', ylabel)
        self.scatter = pg.ScatterPlotItem()
        self.scatter_text = pg.TextItem(color='k')
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        self.scatter_text.setFont(font)
        self.scatter_text.hide()
        self.fig.addItem(self.scatter)
        self.fig.addItem(self.scatter_text)
        self.axis = self.fig.plotItem.getAxis('left')
        self.axis.setScale(0.1)
        self.line_item = []
        #self.fig.
        self.fig.scene().sigMouseMoved.connect(self.on_mouse_hover)
        self.lines = []
        self.text = []
        self.x = []
        self.n_trials = []

 
    def reset(self):
        self.scatter.setData()
    
    def reset_axis(self):
        self.fig.setXRange(min=self.x.min(), max=self.x.max())
        self.fig.setYRange(min=0, max=(1.00 * self.n_trials * 10))
    
    def plot(self, x, y, n_trials, x_axis):
        self.fig.setXRange(min=x_axis.min(), max=x_axis.max())
        self.fig.setYRange(min=0, max=(1.00 * n_trials * 10))
        self.scatter.setData(x=x, y=y, size=1, symbol='s')
        self.scatter.setPen('k')
        self.x = x
        self.n_trials = n_trials

        return self.x, self.n_trials

    def add_lines(self, lines, line_colour, text):
        self.lines = lines
        self.text = text
        for idx, val in enumerate(lines):
            orient = pg.LinearRegionItem.Horizontal
            colour = line_colour[idx]
            colour.setAlpha(50)
            region = pg.LinearRegionItem(values = (val[0]*10, val[1]*10), brush = colour,
            movable = False, orientation = orient)
            self.fig.plotItem.addItem(region)
            self.line_item.append(region)
    
    def on_mouse_hover(self, pos):
        if len(self.lines) > 0:
            pos = self.scatter.mapFromScene(pos)

            if (pos.y() > min(min(self.lines))*10) & (pos.y() < max(max(self.lines))*10):
                text = self.find_text(pos.y())
                self.scatter_text.setText(text)
                self.scatter_text.setPos(pos.x() + 0.1, pos.y() - 10)
                #self.scatter_text.setPos(pos.x(), pos.y())
                self.scatter_text.show()

            else:
                self.scatter_text.hide()
        else:
            self.scatter_text.hide()


    def find_text(self, y):
        idx = [x for x, val in enumerate(self.lines) if (y > val[0]*10) & (y < val[1]*10)][0]
        
        return self.text[idx]
 
    def remove_lines(self):
        for val in self.line_item:
            self.fig.plotItem.removeItem(val)
            


class BarTemplate:
    def __init__(self, xlabel, ylabel):
        self.fig = pg.PlotWidget(background='w')
        self.fig.setMouseEnabled(x=False, y=False)
        self.fig.setLabel('bottom', xlabel)
        self.fig.setLabel('left', ylabel)
        self.bar = pg.BarGraphItem(x=[0], height=[0], width=0)
        self.fig.addItem(self.bar)
    
    def reset(self):
        self.bar.setOpts(x=[0], height=[0], width=0)
    
    def plot(self, x, y, n_bin):
        self.fig.setXRange(min=x.min(), max=x.max())
        self.fig.setYRange(min=0, max=1.05 * y.max())
        self.bar.setOpts(x=x, height=y, width=0.0009, brush='b')



