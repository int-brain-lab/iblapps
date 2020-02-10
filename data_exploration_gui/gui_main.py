from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.exporters

import cluster_class as clust
import scatter_class as scatt
import filter_class as filt
import plot_class as plt
import data_class as dat
import misc_class as misc

from pathlib import Path
import os
import sys

import numpy as np


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Initialise cluster group
        self.cluster = clust.ClusterGroup()
        self.cluster.cluster_list.clicked.connect(self.on_cluster_list_clicked)
        self.cluster.cluster_next_button.clicked.connect(self.on_next_cluster_clicked)
        self.cluster.cluster_previous_button.clicked.connect(self.on_previous_cluster_clicked)
        self.cluster.cluster_option1.clicked.connect(lambda: self.on_cluster_sort_clicked(
            self.cluster.cluster_option1))
        self.cluster.cluster_option2.clicked.connect(lambda: self.on_cluster_sort_clicked(
            self.cluster.cluster_option2))
        self.cluster.cluster_option3.clicked.connect(lambda: self.on_cluster_sort_clicked(
            self.cluster.cluster_option3))

        # Initialise scatter group
        self.scatter = scatt.ScatterGroup()
        self.scatter.fig_scatter.scene().sigMouseMoved.connect(self.on_mouse_hover)
        self.scatter.scatter_plot.sigClicked.connect(self.on_scatter_plot_clicked)
        self.scatter.scatter_reset_button.clicked.connect(self.on_scatter_reset_button_clicked)
        self.scatter_exporter = pg.exporters.ImageExporter(self.scatter.fig_scatter.plotItem)
        self.scatter.fig_scatter.sigDeviceRangeChanged.connect(lambda: self.on_fig_size_changed(
            self.scatter_exporter, self.scatter.scatter_reset_button))

        # Initialise filter group
        self.filter = filt.FilterGroup()
        self.filter.reset_filter_button.clicked.connect(self.on_reset_filter_button_clicked)
        self.filter.contrast_options.itemClicked.connect(self.on_contrast_list_changed)

        self.filter.choice_option1.clicked.connect(lambda: self.on_choice_list_change(
            self.filter.choice_option1))
        self.filter.choice_option2.clicked.connect(lambda: self.on_choice_list_change(
            self.filter.choice_option2))
        self.filter.choice_option3.clicked.connect(lambda: self.on_choice_list_change(
            self.filter.choice_option3))

        self.filter.stim_option1.clicked.connect(lambda: self.on_stim_list_change(
            self.filter.stim_option1))
        self.filter.stim_option2.clicked.connect(lambda: self.on_stim_list_change(
            self.filter.stim_option2))
        self.filter.stim_option3.clicked.connect(lambda: self.on_stim_list_change(
            self.filter.stim_option3))

        self.filter.trial_option1.clicked.connect(lambda: self.on_trial_sort_change(
            self.filter.trial_option1))
        self.filter.trial_option2.clicked.connect(lambda: self.on_trial_sort_change(
            self.filter.trial_option2))
        self.filter.trial_option3.clicked.connect(lambda: self.on_trial_sort_change(
            self.filter.trial_option3))
        self.filter.trial_option4.clicked.connect(lambda: self.on_trial_sort_change(
            self.filter.trial_option4))

        # Initialise figure group
        self.plots = plt.PlotGroup()
        self.plots.fig1_button.clicked.connect(self.on_fig1_button_clicked)
        self.plots.fig2_button.clicked.connect(self.on_fig2_button_clicked)
        
        self.fig1_exporter = pg.exporters.ImageExporter(self.plots.fig1_peth.fig.plotItem)
        self.plots.fig1_peth.fig.sigDeviceRangeChanged.connect(lambda: self.on_fig_size_changed(self.fig1_exporter, self.plots.fig1_button))
        self.fig2_exporter = pg.exporters.ImageExporter(self.plots.fig2_peth.fig.plotItem)
        self.plots.fig2_peth.fig.sigDeviceRangeChanged.connect(lambda: self.on_fig_size_changed(self.fig2_exporter, self.plots.fig2_button))

        # Intitialise data group
        self.data = dat.DataGroup()
        self.data.waveform_list.clicked.connect(self.on_waveform_list_clicked)
        self.data.waveform_button.clicked.connect(self.on_waveform_button_clicked)


        # Initialise misc group
        self.misc = misc.MiscGroup()
        self.misc.folder_button.clicked.connect(self.on_folder_button_clicked)
        self.misc.remove_clust_button1.clicked.connect(lambda: self.on_remove_cluster_button_clicked(self.misc.clust_list1))
        self.misc.remove_clust_button2.clicked.connect(lambda: self.on_remove_cluster_button_clicked(self.misc.clust_list2))
        self.misc.save_clust_button.clicked.connect(self.on_save_button_clicked)
        self.misc.terminal_button.clicked.connect(self.on_terminal_button_clicked)

        # Setup main figure and add all widgets to display
        self.resize(1800, 1000)
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)

        main_widget_layout = QtWidgets.QGridLayout()
        main_widget_layout.setHorizontalSpacing(20)
        main_widget_layout.setVerticalSpacing(20)
        main_widget_layout.setColumnStretch(0, 5)
        main_widget_layout.setColumnStretch(1, 1)
        main_widget_layout.addWidget(self.misc.folder_group, 0, 0, 1, 2)
        main_widget_layout.addWidget(self.cluster.cluster_list_group, 1, 0, 2, 1)
        main_widget_layout.addWidget(self.scatter.fig_scatter, 3, 0, 4, 1)
        main_widget_layout.addWidget(self.misc.terminal, 7, 0, 1, 2)
        main_widget_layout.addWidget(self.filter.filter_options_group, 1, 1, 3, 1)
        main_widget_layout.addWidget(self.misc.clust_interest, 4, 1, 3, 1)
        main_widget_layout.addWidget(self.plots.fig_area, 0, 2, 7, 8)
        main_widget_layout.addWidget(self.filter.trial_options, 7, 6)
        main_widget_layout.addWidget(self.data.waveform_group, 7, 9)
        main_widget.setLayout(main_widget_layout)

    def reset_gui(self):
        self.cluster.reset()
        self.scatter.reset()
        self.plots.reset()
        self.data.reset()
        [self.stim_contrast, self.stim_side, self.stim_choice, self.sort_method] = \
            self.filter.reset_filters()

    def initialise_gui(self):
        self.cluster.populate(self.clust_ids, self.clust_colours)
        [self.clust, self.clust_prev] = self.cluster.initialise_cluster_index()
        self.scatter.populate(self.clust_amps, self.clust_depths, self.clust_ids, self.clust_colours)
        self.scatter.initialise_scatter_index()
        self.data.populate(self.clust)

    def on_folder_button_clicked(self):
    
        self.reset_gui()
        self.cluster.cluster_option1.setChecked(True)
        folder_path = Path(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Folder"))
        
        if str(folder_path) == '.':
            self.misc.terminal.append('>> Cancelling loading')
        else:
            self.misc.folder_line.setText(str(folder_path))
            [self.ephys_file_path, self.gui_path] = self.data.load(folder_path)
            self.misc.terminal.append('>> Loaded data for ' + str(folder_path))
            if len(self.ephys_file_path) != 0:
                self.misc.terminal.append('>> Found ephys file: ' + str(self.ephys_file_path))
            else:
                self.misc.terminal.append('>> Ephys file not found... WARNING Figure 7 will not work')

            order = self.data.sort_by_id
            [self.clust_ids, self.clust_amps, self.clust_depths, self.clust_colours] = \
                self.data.sort_data(order)

            self.initialise_gui()
            nan_trials = self.filter.compute_trial_options(self.data.trials)
            self.misc.terminal.append('>> Found ' + str(len(nan_trials)) + ' nan trials in data ...... these will be removed')

            [self.trials_id, self.lines, self.line_colours] = self.filter.filter_and_sort_trials(
                self.stim_contrast, self.stim_side, self.stim_choice, self.sort_method)

            self.plots.change_all_plots(self.data, self.clust, self.trials_id, self.lines, self.line_colours)
            self.update_display_string()

    # All commands associated with cluster list
    def on_cluster_list_clicked(self):
        # Get new cluster index
        self.clust = self.cluster.on_cluster_list_clicked()
        
        # Update cluster list and scatter plot 
        self.cluster.update_list_icon()
        self.scatter.set_current_point(self.clust)
        self.scatter.update_scatter_icon(self.clust_prev)
        self.scatter.update_prev_point()
        self.clust_prev = self.cluster.update_cluster_index()

        # Repopulate spike list for current cluster
        self.data.populate(self.clust)
        # Change plots
        self.plots.change_all_plots(self.data, self.clust, self.trials_id, self.lines, self.line_colours)
        self.update_display_string()
    
    def on_next_cluster_clicked(self):
        #print(len(self.clust_ids))
        if self.clust != len(self.clust_ids) - 1:
            self.clust = self.cluster.on_next_cluster_clicked()
            self.cluster.update_list_icon()
            self.scatter.set_current_point(self.clust)
            self.scatter.update_scatter_icon(self.clust_prev)
            self.scatter.update_prev_point()
            self.clust_prev = self.cluster.update_cluster_index()
            self.data.populate(self.clust)
            self.plots.change_all_plots(self.data, self.clust, self.trials_id, self.lines, self.line_colours)
            self.update_display_string()
        else:
            self.misc.terminal.append('>> Already on last cluster ... cannot move to next cluster')

    def on_previous_cluster_clicked(self):
        if self.clust != 0:
            self.clust = self.cluster.on_previous_cluster_clicked()
            self.cluster.update_list_icon()
            self.scatter.set_current_point(self.clust)
            self.scatter.update_scatter_icon(self.clust_prev)
            self.scatter.update_prev_point()
            self.clust_prev = self.cluster.update_cluster_index()
            self.data.populate(self.clust)
            self.plots.change_all_plots(self.data, self.clust, self.trials_id, self.lines, self.line_colours)
            self.update_display_string()
        else:
            self.misc.terminal.append('>> Already on first cluster ... cannot move to previous cluster')

    def on_cluster_sort_clicked(self, button):
        option = str(button.text())
        if option == self.cluster.cluster_option1.text():
            order = self.data.sort_by_id
        elif option == self.cluster.cluster_option2.text():
            order = self.data.sort_by_nspikes
        else:
            order = self.data.sort_by_good
        
        [self.clust_ids, self.clust_amps, self.clust_depths, self.clust_colours] = \
            self.data.sort_data(order)
        
        self.reset_gui()
        self.initialise_gui()
        [self.trials_id, self.lines, self.line_colours] = self.filter.filter_and_sort_trials(
            self.stim_contrast, self.stim_side, self.stim_choice, self.sort_method)

        self.plots.change_all_plots(self.data, self.clust, self.trials_id, self.lines, self.line_colours)
        self.update_display_string()
        self.misc.terminal.append('>> Clusters sorted by ' + str(option))
                 

    def on_mouse_hover(self, pos):
        self.scatter.on_mouse_hover(pos)

    def on_scatter_plot_clicked(self, scatter, point):
        self.clust = self.scatter.on_scatter_plot_clicked(point)
        self.cluster.update_current_row(self.clust)
        self.cluster.update_list_icon()
        self.scatter.set_current_point(self.clust)
        self.scatter.update_scatter_icon(self.clust_prev)
        self.scatter.update_prev_point()
        self.clust_prev = self.cluster.update_cluster_index()
        self.data.populate(self.clust)
        self.plots.change_all_plots(self.data, self.clust, self.trials_id, self.lines, self.line_colours)
        self.update_display_string()
    
    def on_scatter_reset_button_clicked(self):
        self.scatter.on_scatter_plot_reset()

    def on_contrast_list_changed(self):
        self.stim_contrast = self.filter.get_checked_contrasts()
        [self.trials_id, self.lines, self.line_colours] = self.filter.filter_and_sort_trials(
            self.stim_contrast, self.stim_side, self.stim_choice, self.sort_method)
        self.plots.change_plots(self.data, self.clust, self.trials_id, self.lines, self.line_colours)
        self.check_n_trials()
        self.update_display_string()

    def on_choice_list_change(self, button):
        self.stim_choice = str(button.text())
        [self.trials_id, self.lines, self.line_colours]=self.filter.filter_and_sort_trials(
            self.stim_contrast, self.stim_side, self.stim_choice, self.sort_method)
        self.plots.change_plots(self.data, self.clust, self.trials_id, self.lines, self.line_colours)
        self.check_n_trials()
        self.update_display_string()

    def on_stim_list_change(self, button):
        self.stim_side = str(button.text())
        [self.trials_id, self.lines, self.line_colours] = self.filter.filter_and_sort_trials(
            self.stim_contrast, self.stim_side, self.stim_choice, self.sort_method)
        self.plots.change_plots(self.data, self.clust, self.trials_id, self.lines, self.line_colours)
        self.check_n_trials()
        self.update_display_string()

    def on_trial_sort_change(self, button):
        self.sort_method = str(button.text())
        [self.trials_id, self.lines, self.line_colours] = self.filter.filter_and_sort_trials(
            self.stim_contrast, self.stim_side, self.stim_choice, self.sort_method)
        self.plots.change_plots(self.data, self.clust, self.trials_id, self.lines, self.line_colours)

    def on_reset_filter_button_clicked(self):
        [self.stim_contrast, self.stim_side, self.stim_choice, self.sort_method] = self.filter.reset_filters()
        [self.trials_id, self.lines, self.line_colours] = self.filter.filter_and_sort_trials(
            self.stim_contrast, self.stim_side, self.stim_choice, self.sort_method)
        self.plots.change_plots(self.data, self.clust, self.trials_id, self.lines, self.line_colours)
        self.update_display_string()
        self.misc.terminal.append('>> All filters reset')

    def on_fig1_button_clicked(self):
        item = QtWidgets.QListWidgetItem('Cluster ' + str(self.clust_ids[self.clust]))
        self.misc.clust_list1.addItem(item)

    def on_fig2_button_clicked(self):
        item = QtWidgets.QListWidgetItem('Cluster ' + str(self.clust_ids[self.clust]))
        self.misc.clust_list2.addItem(item)
    
    def on_remove_cluster_button_clicked(self, clust_list):
        item = clust_list.currentRow()
        clust_list.takeItem(item)
    
    def on_save_button_clicked(self):
        file = self.misc.on_save_button_clicked(self.gui_path)
        if file:
            self.misc.terminal.append('>> Clusters saved to file ' + file)
        else:
            self.misc.terminal.append('>> Cancelled saving')

    def on_waveform_list_clicked(self):
        self.data.n_waveform = self.data.waveform_list.currentRow()
        
    def on_waveform_button_clicked(self):
        if len(self.ephys_file_path) != 0:
            self.plots.plot_waveform(self.data, self.clust)
        else:
            self.misc.terminal.append('>> WARNING Did not find ephys file... cannot generate waveform')

    def on_terminal_button_clicked(self):
        self.misc.terminal.clear()

    def on_fig_size_changed(self, exporter, button):
        fig_width = exporter.getTargetRect().width() 
        button_width = button.width()
        button.move(fig_width - button_width, 0)

    def check_n_trials(self):
        if len(self.trials_id) == 0:
            self.misc.terminal.append('>> No trials for this filter combination')
    
    def update_display_string(self):
        self.filter.ntrials_text.setText('No. of trials = ' + str(len(self.trials_id)))
        self.data.waveform_text.setText('No. of spikes = ' + str(len(self.data.clus_idx)))


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    mainapp = MainWindow()
    mainapp.show()
    app.exec_() 
