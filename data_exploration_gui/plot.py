from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.dockarea
from data_exploration_gui.utils import (MAP_SORT_OPTIONS, MAP_SIDE_OPTIONS, MAP_CHOICE_OPTIONS,
                                        PSTH_OPTIONS, RASTER_OPTIONS, colours)
from data_exploration_gui import utils
import atlaselectrophysiology.ColorBar as cb
import numpy as np


class PlotGroup:

    def __init__(self, data_model):

        self.data = data_model
        self.create_dock_area()

        self.fig_spike_psth = PlotTemplate('Time After Event (s)', 'Firing Rate (Hz)')
        self.fig1_area.addWidget(self.fig_spike_psth.fig)
        self.fig_behav_psth = PlotTemplate('Time After Event (s)', 'Firing Rate (Hz)')
        self.fig2_area.addWidget(self.fig_behav_psth.fig)

        self.fig_spike_raster = ScatterTemplate('Time After Event (s)', 'No. of Trials')
        self.fig3_area.addWidget(self.fig_spike_raster.fig)
        self.fig_behav_raster = ImageTemplate('Time After Event (s)', 'No. of Trials')
        self.fig4_area.addWidget(self.fig_behav_raster.fig)

        self.plot_status = {key: False for key in MAP_SORT_OPTIONS.keys()}


    def create_dock_area(self):
        self.fig_area = pg.dockarea.DockArea()
        self.fig1_area = pg.dockarea.Dock('', autoOrientation='horizontal')
        self.fig2_area = pg.dockarea.Dock('', autoOrientation='horizontal')
        self.fig3_area = pg.dockarea.Dock('', autoOrientation='horizontal')
        self.fig4_area = pg.dockarea.Dock('', autoOrientation='horizontal')
        #self.fig5_area = pg.dockarea.Dock('Figure 5: Autocorrelogram',
        #                                  autoOrientation='horizontal')
        #self.fig6_area = pg.dockarea.Dock('Figure 6: Template Waveform',
        #                                  autoOrientation='horizontal')
        #self.fig7_area = pg.dockarea.Dock('Figure 7: Waveform', autoOrientation='horizontal')

        self.fig_area.addDock(self.fig2_area, 'top')
        self.fig_area.addDock(self.fig1_area, 'left', self.fig2_area)
        self.fig_area.addDock(self.fig3_area, 'bottom', self.fig1_area)
        self.fig_area.addDock(self.fig4_area, 'bottom', self.fig2_area)
        #self.fig_area.addDock(self.fig5_area, 'right')
        #self.fig_area.addDock(self.fig6_area, 'bottom', self.fig5_area)
        #self.fig_area.addDock(self.fig7_area, 'bottom', self.fig6_area)

        self.fig3_area.setStretch(x=1, y=18)
        self.fig4_area.setStretch(x=1, y=18)

        #self.fig5_area.setStretch(x=7, y=1)
        #self.fig6_area.setStretch(x=7, y=1)
        #self.fig7_area.setStretch(x=7, y=1)

    def reset(self):

        for key, val in self.plot_status.items():
            if val:
                self.fig_spike_psth.remove_item(key)
                self.plot_status[key] = False

    def change_rasters(self, trial_set, contrast, order, sort, hold, event):
        side = MAP_SIDE_OPTIONS[trial_set]
        choice = MAP_CHOICE_OPTIONS[trial_set]

        if hold and trial_set != 'all':
            #sort = MAP_SORT_OPTIONS[trial_set]
            spike_raster, behav_raster = self.data.get_rasters_for_selection('all', 'all', order,
                                                                             sort, contrast, event)
            raster_options = RASTER_OPTIONS['all'][sort]
        else:
            spike_raster, behav_raster = self.data.get_rasters_for_selection(side, choice, order,
                                                                             sort, contrast, event)
            raster_options = RASTER_OPTIONS[trial_set][sort]

        self.fig_spike_raster.remove_regions()
        self.fig_spike_raster.plot(spike_raster.raster, spike_raster.time, spike_raster.n_trials)
        self.fig_spike_raster.add_regions(spike_raster.dividers, raster_options)

        self.fig_behav_raster.remove_regions()
        self.fig_behav_raster.plot(behav_raster.raster, behav_raster.time, cmap=behav_raster.cmap,
                                   clevels=behav_raster.clevels)
        self.fig_behav_raster.add_regions(behav_raster.dividers, raster_options)

    def change_psths(self, trial_set, contrast, order, sort, event):
        side = MAP_SIDE_OPTIONS[trial_set]
        choice = MAP_CHOICE_OPTIONS[trial_set]

        spike_psth, behav_psth = self.data.get_psths_for_selection(side, choice, order,
                                                                   sort, contrast, event)
        self.fig_spike_psth.plot(trial_set, spike_psth.time, spike_psth.psth_mean,
                                 spike_psth.psth_std, PSTH_OPTIONS[trial_set],
                                 ylabel=spike_psth.ylabel)
        self.fig_behav_psth.plot(trial_set, behav_psth.time, behav_psth.psth_mean,
                                 behav_psth.psth_std, PSTH_OPTIONS[trial_set],
                                 ylabel=behav_psth.ylabel)

    def change_plots(self, contrast, trial_set, order, sort, hold, event):

        self.plot_status[trial_set] = True

        if not hold:
            self.remove_plots(self.prev_trial_set)

        self.change_rasters(trial_set, contrast, order, sort, hold, event)
        self.change_psths(trial_set, contrast, order, sort, event)

        self.prev_trial_set = trial_set

    def change_all_plots(self, contrast, trial_set, order, sort, hold, event):

        n_plots = np.sum(list(self.plot_status.values()))
        if n_plots == 0:
            self.change_psths(trial_set, contrast, order, sort, event)
            self.plot_status[trial_set] = True
        else:
            for key, val in self.plot_status.items():
                if val:
                    self.remove_plots(key)
                    self.change_psths(key, contrast, order, sort, event)
                    self.plot_status[key] = True

        self.change_rasters(trial_set, contrast, order, sort, hold, event)
        self.prev_trial_set = trial_set

    def remove_plots(self, trial_set):
        self.plot_status[trial_set] = False
        self.fig_spike_psth.remove_item(trial_set)
        self.fig_behav_psth.remove_item(trial_set)

    def reset(self):

        for key, val in self.plot_status.items():
            if val:
                self.remove_plots(key)
        self.fig_spike_raster.reset()
        self.fig_behav_raster.reset()


class PlotTemplate:
    def __init__(self, xlabel, ylabel):
        self.fig = pg.PlotWidget(background='w')
        self.fig.setMouseEnabled(x=False, y=False)
        self.fig.setLabel('bottom', xlabel)
        self.fig.setLabel('left', ylabel)
        self.plot_items = dict()
        self.fig.plotItem.addLine(x=0, pen=colours['line'])

    def add_item(self, trial_set, yrange):

        if trial_set not in self.plot_items.keys():

            curve = {'centre': pg.PlotCurveItem(),
                     'upper': pg.PlotCurveItem(),
                     'lower': pg.PlotCurveItem(),
                     'fill': pg.FillBetweenItem(),
                     'yrange': yrange}

            self.plot_items[trial_set] = curve
            self.fig.addItem(self.plot_items[trial_set]['centre'])
            self.fig.addItem(self.plot_items[trial_set]['fill'])

    def remove_item(self, trial_set):
        self.fig.removeItem(self.plot_items[trial_set]['centre'])
        self.fig.removeItem(self.plot_items[trial_set]['fill'])
        self.plot_items.pop(trial_set)

    def plot(self, trial_set, x, y, se, plot_info, ylabel=None ):
        self.add_item(trial_set, [np.nanmin(y - se), np.nanmax(y + se)])

        self.plot_items[trial_set]['centre'].setData(x=x, y=y)
        self.plot_items[trial_set]['centre'].setPen(plot_info['colour'])
        self.plot_items[trial_set]['lower'].setData(x=x, y=y - se)
        self.plot_items[trial_set]['upper'].setData(x=x, y=y + se)
        self.plot_items[trial_set]['fill'].setCurves(self.plot_items[trial_set]['upper'],
                                                     self.plot_items[trial_set]['lower'])
        plot_info['fill'].setAlpha(50)
        self.plot_items[trial_set]['fill'].setBrush(plot_info['fill'])

        # find the correct y range based on all the lines on the plot
        y_min = np.nanmin([val['yrange'][0] for _, val in self.plot_items.items()])
        y_max = np.nanmax([val['yrange'][1] for _, val in self.plot_items.items()])

        self.fig.setXRange(min=np.min(x), max=np.max(x))
        self.fig.setYRange(min=0.95 * y_min, max=1.05 * y_max)

        if ylabel is not None:
            self.fig.setLabel('left', ylabel)



class ImageTemplate:
    def __init__(self, xlabel, ylabel):
        self.fig = pg.PlotWidget(background='w')
        self.fig.setLabel('bottom', xlabel)
        self.fig.setLabel('left', ylabel)
        self.fig.scene().sigMouseMoved.connect(self.on_mouse_hover)

        self.image = pg.ImageItem()
        self.fig.addItem(self.image)
        self.fig.plotItem.addLine(x=0, pen=colours['line'])

        self.text_popup = pg.TextItem(color=colours['line'])
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        self.text_popup.setFont(font)
        self.text_popup.hide()
        self.fig.addItem(self.text_popup)

        self.region_items = []
        self.regions = None
        self.region_text = None

    def reset(self):
        self.image.setImage()

    def plot(self, image, t, cmap='binary', clevels=(0, 1)):

        # TODO makes sure this is in the order that you expect!!!
        self.image.setImage(image.T)
        self.x_range = [np.min(t), np.max(t)]
        self.x_scale = (np.max(t) - np.min(t)) / image.shape[1]
        transform = [self.x_scale, 0., 0., 0., 1., 0., np.min(t),
                     0., 1.]
        self.image.setTransform(QtGui.QTransform(*transform))

        color_bar = cb.ColorBar(cmap)
        lut = color_bar.getColourMap()
        self.image.setLookupTable(lut)
        self.image.setLevels(clevels)
        self.fig.setXRange(min=np.min(t), max=np.max(t))
        self.fig.setYRange(min=0, max=image.shape[0])


    def add_regions(self, regions, region_info):
        self.regions = regions
        self.region_text = region_info['text']
        for reg, col in zip(self.regions, region_info['colours']):
            # Hack so we can easily revert to the coloured option if users prefer
            col.setAlpha(0)
            region = pg.LinearRegionItem(values=(reg[0], reg[1]), brush=col, pen=colours['line'],
                                         movable=False, orientation='horizontal')
            self.fig.plotItem.addItem(region)
            self.region_items.append(region)

    def add_lines(self, regions):
        self.lines = []

    def remove_regions(self):
        for reg in self.region_items:
            self.fig.plotItem.removeItem(reg)

    def on_mouse_hover(self, pos):
        if len(self.regions) > 0:
            pos = self.image.mapFromScene(pos)

            # only show if mouse is in x range
            if ((pos.x() * self.x_scale) + self.x_range[0] > self.x_range[0]) & \
                    ((pos.x() * self.x_scale) + self.x_range[0] < self.x_range[1]):
                if (pos.y() > np.min(np.min(self.regions))) & \
                        (pos.y() < np.max(np.max(self.regions))):
                    text = self.find_text(pos.y())
                    self.text_popup.setText(text)
                    self.text_popup.setPos((pos.x() * self.x_scale) + self.x_range[0] + 0.05,
                                           pos.y() - 10)
                    self.text_popup.show()
                else:
                    self.text_popup.hide()
            else:
                self.text_popup.hide()
        else:
            self.text_popup.hide()

    def find_text(self, y):
        idx = [i for i, val in enumerate(self.regions) if (y > val[0]) & (y < val[1])][0]
        return self.region_text[idx]


class ScatterTemplate:
    def __init__(self, xlabel, ylabel):

        self.fig = pg.PlotWidget(background='w')
        self.fig.setLabel('bottom', xlabel)
        self.fig.setLabel('left', ylabel)
        self.fig.scene().sigMouseMoved.connect(self.on_mouse_hover)

        self.scatter = pg.ScatterPlotItem()
        self.fig.addItem(self.scatter)
        self.fig.plotItem.addLine(x=0, pen=colours['line'])

        self.text_popup = pg.TextItem(color=colours['line'])
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        self.text_popup.setFont(font)
        self.text_popup.hide()
        self.fig.addItem(self.text_popup)

        self.region_items = []
        self.regions = None
        self.region_text = None

    def reset(self):
        self.scatter.setData()

    def plot(self, scatter, t, n_trials):
        self.scatter.setData(x=scatter[:, 0], y=scatter[:, 1], size=1, symbol='s')
        self.scatter.setPen('k')
        self.fig.setXRange(min=t[0], max=t[1])
        self.fig.setYRange(min=0, max=n_trials)
        self.x_range = t

    def add_regions(self, regions, region_info):
        self.regions = regions
        self.region_text = region_info['text']
        for reg, col in zip(self.regions, region_info['colours']):
            # Hack so we can easily revert to the coloured option if users prefer
            col.setAlpha(0)
            region = pg.LinearRegionItem(values=(reg[0], reg[1]), brush=col, pen=colours['line'],
                                         movable=False, orientation='horizontal')
            self.fig.plotItem.addItem(region)
            self.region_items.append(region)

    def remove_regions(self):
        for reg in self.region_items:
            self.fig.plotItem.removeItem(reg)

    def on_mouse_hover(self, pos):
        if len(self.regions) > 0:
            pos = self.scatter.mapFromScene(pos)

            # only show if mouse is in x range
            if (pos.x() > self.x_range[0]) & (pos.x() < self.x_range[1]):
                if (pos.y() > np.min(np.min(self.regions))) & \
                        (pos.y() < np.max(np.max(self.regions))):
                    text = self.find_text(pos.y())
                    self.text_popup.setText(text)
                    self.text_popup.setPos(pos.x() + 0.05, pos.y() - 10)
                    self.text_popup.show()
                else:
                    self.text_popup.hide()
            else:
                self.text_popup.hide()
        else:
            self.text_popup.hide()

    def find_text(self, y):
        idx = [i for i, val in enumerate(self.regions) if (y > val[0]) & (y < val[1])][0]
        return self.region_text[idx]



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



