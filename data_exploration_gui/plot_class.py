from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.dockarea


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
        self.fig1_peth.reset()
        self.fig2_peth.reset()
        self.fig3_raster.reset()
        self.fig4_raster.reset()
        self.fig5_autocorr.reset()
        self.fig6_template.reset()
        self.fig7_waveform.reset()

    def change_plots(self, data, clust, trial_id, line, line_colour):
        if len(trial_id) != 0:
            [x, y, errbar] = data.compute_peth('goCue_times', clust, trial_id)
            self.fig1_peth.plot_errbar(x, y, errbar)
            self.fig1_peth.fig.plotItem.addLine(x = 0, pen = 'm')
            [x, y, errbar] = data.compute_peth('feedback_times', clust, trial_id)
            self.fig2_peth.plot_errbar(x, y, errbar)
            self.fig2_peth.fig.plotItem.addLine(x = 0, pen = 'm')

            raster_x = x

            self.fig3_raster.remove_lines()
            [x, y, n_trials] = data.compute_rasters('goCue_times', clust, trial_id)
            if len(x) != 0:
                self.fig3_raster.plot(x, y, n_trials, raster_x)
                self.fig3_raster.add_lines(line, line_colour)
                self.fig3_raster.fig.plotItem.addLine(x = 0, pen = 'm')
            else:
                self.fig3_raster.reset()

            self.fig4_raster.remove_lines()
            [x, y, n_trials] = data.compute_rasters('feedback_times', clust, trial_id)
            if len(x) != 0:
                self.fig4_raster.plot(x, y, n_trials, raster_x)
                self.fig4_raster.add_lines(line, line_colour)
                self.fig4_raster.fig.plotItem.addLine(x = 0, pen = 'm')
            else:
                self.fig4_raster.reset()

        else:
            self.fig1_peth.reset()
            self.fig2_peth.reset()
            self.fig3_raster.reset()
            self.fig4_raster.reset()
        
    def change_all_plots(self, data, clust, trial_id, line, line_colour):
        if len(trial_id) != 0:
            [x, y, errbar] = data.compute_peth('goCue_times', clust, trial_id)
            self.fig1_peth.plot_errbar(x, y, errbar)
            self.fig1_peth.fig.plotItem.addLine(x = 0, pen = 'm')
            [x, y, errbar] = data.compute_peth('feedback_times', clust, trial_id)
            self.fig2_peth.plot_errbar(x, y, errbar)
            self.fig2_peth.fig.plotItem.addLine(x = 0, pen = 'm')

            raster_x = x

            self.fig3_raster.remove_lines()
            [x, y, n_trials] = data.compute_rasters('goCue_times', clust, trial_id)
            if len(x) != 0:
                self.fig3_raster.plot(x, y, n_trials, raster_x)
                self.fig3_raster.add_lines(line, line_colour)
                self.fig3_raster.fig.plotItem.addLine(x = 0, pen = 'm')
            else:
                self.fig3_raster.reset()

            self.fig4_raster.remove_lines()
            [x, y, n_trials] = data.compute_rasters('feedback_times', clust, trial_id)
            if len(x) != 0:
                self.fig4_raster.plot(x, y, n_trials, raster_x)
                self.fig4_raster.add_lines(line, line_colour)
                self.fig4_raster.fig.plotItem.addLine(x = 0, pen = 'm')
            else:
                self.fig3_raster.reset()

            [x, y] = data.compute_autocorr(clust)
            self.fig5_autocorr.plot(x, y, data.autocorr_bin)

            [x, y] = data.compute_template(clust)
            self.fig6_template.plot_line(x, y)
            self.fig7_waveform.reset()

        else:
            self.fig1_peth.reset()
            self.fig2_peth.reset()
            self.fig3_raster.reset()
            self.fig4_raster.reset()
    
    def plot_waveform(self, data, clust):
        [x, y, errbar] = data.compute_waveform(clust)
        self.fig7_waveform.plot_errbar(x, y, errbar)


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
    
    def reset(self):
        self.plot.setData()
        self.plot_lb.setData()
        self.plot_ub.setData()
        self.plot_fill.setCurves(self.plot_ub, self.plot_lb)
    
    def plot_errbar(self, x, y, errbar):
        self.plot.setData(x=x, y=y)
        self.plot_lb.setData(x=x, y=(y - errbar))
        self.plot_ub.setData(x=x, y=(y + errbar))
        self.plot_fill.setCurves(self.plot_ub, self.plot_lb)
        self.plot_fill.setBrush('b')
        self.fig.setXRange(min=x.min(), max=x.max())
        self.fig.setYRange(min=0.95 * (y - errbar).min(), max=1.05 * (y + errbar).max())

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
        #self.fig.setMouseEnabled(y=False)
        self.fig.setLabel('bottom', xlabel)
        self.fig.setLabel('left', ylabel)
        self.scatter = pg.ScatterPlotItem()
        self.fig.addItem(self.scatter)
        self.axis = self.fig.plotItem.getAxis('left')
        self.axis.setScale(0.1)
        self.line_item = []
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

    def add_lines(self, lines, line_colour):
        for idx, val in enumerate(lines):
            self.line_item.append(self.fig.plotItem.addLine(y=(val * 10), pen= line_colour[idx]))
    
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



