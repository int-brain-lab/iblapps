from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.exporters

import data_exploration_gui.cluster as clust
import data_exploration_gui.scatter as scatt
import data_exploration_gui.filter as filt
import data_exploration_gui.plot as plt
import data_exploration_gui.data_model as dat
from data_exploration_gui import utils
import data_exploration_gui.misc_class as misc

from pathlib import Path
import numpy as np

import qt
import matplotlib.pyplot as mpl  # noqa  # This is needed to make qt show properly :/


class MainWindow(QtWidgets.QMainWindow):

    @staticmethod
    def _instances():
        app = QtWidgets.QApplication.instance()
        return [w for w in app.topLevelWidgets() if isinstance(w, MainWindow)]

    @staticmethod
    def _get_or_create(title='Data Exploration GUI', **kwargs):
        av = next(filter(lambda e: e.isVisible() and e.windowTitle() == title,
                         MainWindow._instances()), None)
        if av is None:
            av = MainWindow(**kwargs)
            av.setWindowTitle(title)

            return av

    def __init__(self):
        super(MainWindow, self).__init__()

        self.data = dat.DataModel()
        self.plot = plt.PlotGroup(self.data)

        # Initialise cluster group
        self.cluster = clust.ClusterGroup()
        # Connect signals
        self.cluster.cluster_list.clicked.connect(self.on_cluster_list_clicked)
        self.cluster.cluster_next_button.clicked.connect(self.on_next_cluster_clicked)
        self.cluster.cluster_previous_button.clicked.connect(self.on_previous_cluster_clicked)
        self.cluster.cluster_buttons.buttonClicked.connect(self.on_cluster_sort_clicked)

        # Initialise scatter group
        self.scatter = scatt.ScatterGroup()
        # connect signals
        self.scatter.scatter_plot.sigClicked.connect(self.on_scatter_plot_clicked)

        # Initialise filter group
        self.filter = filt.FilterGroup()
        # Connect signals
        self.filter.contrast_buttons.buttonToggled.connect(self.on_filters_changed)
        self.filter.hold_button.stateChanged.connect(self.on_hold_button_clicked)
        self.filter.trial_buttons.buttonToggled.connect(self.on_trial_set_changed)
        self.filter.order_buttons.buttonClicked.connect(self.on_filters_changed)
        self.filter.sort_buttons.buttonClicked.connect(self.on_filters_changed)
        self.filter.reset_button.clicked.connect(self.on_reset_filters_clicked)

        self.filter.populate(self.data.trial_events, self.data.behav_events)
        self.filter.event_combobox.activated.connect(self.on_event_selected)
        self.filter.behav_combobox.activated.connect(self.on_behav_selected)

        self.resize(1800, 1000)
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)

        main_widget_layout = QtWidgets.QGridLayout()
        main_widget_layout.setHorizontalSpacing(20)
        main_widget_layout.setVerticalSpacing(20)
        main_widget_layout.addWidget(self.cluster.cluster_list_group, 0, 0)
        main_widget_layout.addWidget(self.scatter.fig_scatter, 1, 0)
        main_widget_layout.addWidget(self.filter.filter_options_group, 0, 1, 2, 1)
        main_widget_layout.addWidget(self.plot.fig_area, 0, 2, 2, 1)

        main_widget.setLayout(main_widget_layout)

        self.initialise_gui()

    def initialise_gui(self):

        self.cluster.reset()
        self.scatter.reset()
        self.plot.reset()

        # Reset all filters
        self.filter.trial_buttons.blockSignals(True)
        self.filter.reset_filters()
        self.filter.trial_buttons.blockSignals(False)

        # cluster sort, trial event, behav_event, sort, order, hold, contrasts, trial_set
        self.filter.trial_buttons.setExclusive(False)
        # Get filter options
        self.contrasts, self.order, self.sort, self.hold = self.filter.get_selected_filters()
        # Get trial event to align to
        self.trial_event = self.filter.get_selected_event()
        # Get behaviour event to display
        self.behav = self.filter.get_selected_behaviour()
        # Get initial trial_set to show
        self.trial_set = self.filter.get_selected_trials()[0]
        # Get the order to sort clusters
        self.cluster_sort = self.cluster.get_selected_sort()

        # Sort the clusters
        self.clust_data = self.data.sort_clusters(self.cluster_sort)
        # Populate cluster list
        self.cluster.populate(self.clust_data.ids, self.clust_data.colours_ks,
                              self.clust_data.colours_ibl)
        self.clust, self.clust_prev = self.cluster.initialise_cluster_index()

        # Populate scatter plot
        self.scatter.populate(self.clust_data.amps, self.clust_data.depths, self.clust_data.ids,
                              self.clust_data.colours_ks, self.clust_data.colours_ibl )
        self.scatter.initialise_scatter_index()

        # Get the relevant data and plot
        #self.data.alt_raster(self.clust, self.trial_event)
        self.data._get_spike_data_for_selection(self.clust, self.trial_event)
        self.data._get_behaviour_data_for_selection(self.behav, self.trial_event)
        self.plot.change_all_plots(self.contrasts, self.trial_set, self.order, self.sort,
                                   self.hold, self.trial_event)

    def on_event_selected(self, idx):
        self.trial_event = self.filter.event_combobox.itemText(idx)
        self.data._get_spike_data_for_selection(self.clust, self.trial_event)
        self.data._get_behaviour_data_for_selection(self.behav, self.trial_event)
        self.plot.change_all_plots(self.contrasts, self.trial_set, self.order, self.sort,
                                   self.hold, self.trial_event)

    def on_behav_selected(self, idx):
        self.behav = self.filter.behav_combobox.itemText(idx)
        self.data._get_behaviour_data_for_selection(self.behav, self.trial_event)
        self.plot.change_all_plots(self.contrasts, self.trial_set, self.order, self.sort,
                                   self.hold, self.trial_event)

    def on_filters_changed(self):
        self.contrasts, self.order, self.sort, self.hold = self.filter.get_selected_filters()
        self.plot.change_plots(self.contrasts, self.trial_set, self.order, self.sort, self.hold,
                               self.trial_event)

    def on_hold_button_clicked(self):
        self.hold = self.filter.get_hold_status()
        if self.hold:
            self.filter.trial_buttons.setExclusive(False)
        else:
            self.filter.trial_buttons.blockSignals(True)
            self.filter.trial_buttons.setExclusive(True)
            self.filter.reset_filters()
            self.plot.reset()
            self.filter.trial_buttons.blockSignals(False)

            self.contrasts, self.order, self.sort, self.hold = self.filter.get_selected_filters()
            self.trial_set = self.filter.get_selected_trials()[0]
            self.plot.change_all_plots(self.contrasts, self.trial_set, self.order, self.sort,
                                       self.hold, self.trial_event)

    def on_trial_set_changed(self, button, signal):
        self.trial_set = button.text()
        if self.hold is False:
            self.plot.change_plots(self.contrasts, self.trial_set, self.order, self.sort,
                                   self.hold, self.trial_event)
        else:
            if signal is True:
                self.sort = utils.MAP_SORT_OPTIONS[self.trial_set]
                self.filter.set_sorted_button(self.sort)
                self.plot.change_plots(self.contrasts, self.trial_set, self.order, self.sort,
                                       self.hold, self.trial_event)
            else:
                # removing a plot
                self.plot.remove_plots(self.trial_set)

    def on_cluster_list_clicked(self):
        # Get new cluster index
        self.clust = self.cluster.on_cluster_list_clicked()

        # Update cluster list and scatter plot
        self.cluster.update_list_icon()
        self.scatter.set_current_point(self.clust)
        self.scatter.update_scatter_icon(self.clust_prev)
        self.scatter.update_prev_point()
        self.clust_prev = self.cluster.update_cluster_index()

        # Change plots
        self.data._get_spike_data_for_selection(self.clust, self.trial_event)
        self.plot.change_all_plots(self.contrasts, self.trial_set, self.order, self.sort,
                                   self.hold, self.trial_event)

    def on_next_cluster_clicked(self):

        if self.clust != len(self.clust_data.ids) - 1:
            self.clust = self.cluster.on_next_cluster_clicked()
            self.cluster.update_list_icon()
            self.scatter.set_current_point(self.clust)
            self.scatter.update_scatter_icon(self.clust_prev)
            self.scatter.update_prev_point()
            self.clust_prev = self.cluster.update_cluster_index()
            self.data._get_spike_data_for_selection(self.clust, self.trial_event)
            self.plot.change_all_plots(self.contrasts, self.trial_set, self.order, self.sort,
                                       self.hold, self.trial_event)


    def on_previous_cluster_clicked(self):

        if self.clust != 0:
            self.clust = self.cluster.on_previous_cluster_clicked()
            self.cluster.update_list_icon()
            self.scatter.set_current_point(self.clust)
            self.scatter.update_scatter_icon(self.clust_prev)
            self.scatter.update_prev_point()
            self.clust_prev = self.cluster.update_cluster_index()
            self.data._get_spike_data_for_selection(self.clust, self.trial_event)
            self.plot.change_all_plots(self.contrasts, self.trial_set, self.order, self.sort,
                                       self.hold, self.trial_event)

    def on_scatter_plot_clicked(self, scatter, point):

        self.clust = self.scatter.on_scatter_plot_clicked(point)
        self.cluster.update_current_row(self.clust)
        self.cluster.update_list_icon()
        self.scatter.set_current_point(self.clust)
        self.scatter.update_scatter_icon(self.clust_prev)
        self.scatter.update_prev_point()
        self.clust_prev = self.cluster.update_cluster_index()
        self.data._get_spike_data_for_selection(self.clust, self.trial_event)
        self.plot.change_all_plots(self.contrasts, self.trial_set, self.order, self.sort,
                                   self.hold, self.trial_event)

    def on_cluster_sort_clicked(self):
        self.initialise_gui()


    def on_reset_filters_clicked(self):

        self.filter.trial_buttons.blockSignals(True)
        self.filter.reset_filters()
        self.filter.trial_buttons.blockSignals(False)

        # cluster sort, trial event, behav_event, sort, order, hold, contrasts, trial_set
        self.filter.trial_buttons.setExclusive(False)
        # Get filter options
        self.contrasts, self.order, self.sort, self.hold = self.filter.get_selected_filters()
        self.trial_set = self.filter.get_selected_trials()[0]
        self.plot.reset()

        self.plot.change_all_plots(self.contrasts, self.trial_set, self.order, self.sort,
                                   self.hold, self.trial_event)




def viewer(data=None):
    """
    """
    qt.create_app()
    av = MainWindow._get_or_create()
    av.show()
    if data is not None:
        av.on_data_given(data)
    return av


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    mainapp = MainWindow()
    mainapp.show()
    app.exec_()







