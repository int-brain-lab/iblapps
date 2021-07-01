from PyQt5 import QtGui, QtWidgets
import numpy as np
import os
import one.alf.io as alfio
from brainbox.processing import get_units_bunch
from brainbox.population.decode import xcorr
from brainbox.singlecell import calculate_peths
from brainbox.io.spikeglx import extract_waveforms
from pathlib import Path


class DataGroup:
    def __init__(self):

        self.waveform_button = QtWidgets.QPushButton('Generate Waveform')
        self.waveform_list = QtWidgets.QListWidget()
        self.waveform_list.SingleSelection
        self.waveform_text = QtWidgets.QLabel('No. of spikes =')

        waveform_layout = QtWidgets.QGridLayout()
        waveform_layout.addWidget(self.waveform_button, 0, 0)
        waveform_layout.addWidget(self.waveform_text, 1, 0)
        waveform_layout.addWidget(self.waveform_list, 0, 1, 3, 1)

        self.waveform_group = QtWidgets.QGroupBox()
        self.waveform_group.setLayout(waveform_layout)


        #For peths and rasters
        self.t_before = 0.4
        self.t_after = 1

        #For autocorrolelogram
        self.autocorr_window = 0.1
        self.autocorr_bin = 0.001


        #For waveform (N.B in ms)
        self.waveform_window = 2
        self.CAR = False

    def load(self, folder_path):
        self.folder_path = folder_path
        self.find_files()
        self.load_data()
        self.compute_timescales()

        return self.ephys_file_path, self.gui_path

    def find_files(self):
        self.probe_path = self.folder_path
        self.alf_path = self.folder_path.parent
        dir_path = self.folder_path.parent.parent
        self.gui_path = os.path.join(self.probe_path, 'gui')
        probe = os.path.split(self.probe_path)[1]
        ephys_path = os.path.join(dir_path, 'raw_ephys_data', probe)

        self.ephys_file_path = []

        try:
            for i in os.listdir(ephys_path):
                if 'ap' in i and 'bin' in i:
                    self.ephys_file_path = os.path.join(ephys_path, i)
        except:
            self.ephys_file_path = []


    def load_data(self):
        self.spikes = alfio.load_object(self.probe_path, 'spikes')
        self.trials = alfio.load_object(self.alf_path, 'trials')
        self.clusters = alfio.load_object(self.probe_path, 'clusters')
        self.ids = np.unique(self.spikes.clusters)
        self.metrics = np.array(self.clusters.metrics.ks2_label[self.ids])
        self.colours = np.array(self.clusters.metrics.ks2_label[self.ids])
        self.colours[np.where(self.colours == 'mua')[0]] = QtGui.QColor('#fdc086')
        self.colours[np.where(self.colours == 'good')[0]] = QtGui.QColor('#7fc97f')

        _, self.depths, self.nspikes = compute_cluster_average(spikes.clusters, spikes.depths)
        _, self.amps, _ = compute_cluster_average(spikes.clusters, spikes.amps)
        self.amps = self.amps * 1e6

        self.sort_by_id = np.arange(len(self.ids))
        self.sort_by_nspikes = np.argsort(self.nspikes)
        self.sort_by_nspikes = self.sort_by_nspikes[::-1]
        self.sort_by_good = np.append(np.where(self.metrics == 'good')[0],
                                      np.where(self.metrics == 'mua')[0])
        self.n_trials = len(self.trials['contrastLeft'])


    def compute_depth_and_amplitudes(self):
        units_b = get_units_bunch(self.spikes)
        self.depths = []
        self.amps = []
        self.nspikes = []
        for clu in self.ids:
            self.depths = np.append(self.depths, np.nanmean(units_b.depths[str(clu)]))
            self.amps = np.append(self.amps, np.nanmean(units_b.amps[str(clu)]) * 1e6)
            self.nspikes = np.append(self.nspikes, len(units_b.amps[str(clu)]))

        np.save((self.gui_path + '/cluster_depths'), self.depths)
        np.save((self.gui_path + '/cluster_amps'), self.amps)
        np.save((self.gui_path + '/cluster_nspikes'), self.nspikes)

    def sort_data(self, order):
        self.clust_ids = self.ids[order]
        self.clust_amps = self.amps[order]
        self.clust_depths = self.depths[order]
        self.clust_colours = self.colours[order]

        return self.clust_ids, self.clust_amps, self.clust_depths, self.clust_colours

    def reset(self):
        self.waveform_list.clear()

    def populate(self, clust):
        self.reset()
        self.clus_idx = np.where(self.spikes.clusters == self.clust_ids[clust])[0]

        if len(self.clus_idx) <= 500:
            self.spk_intervals = [0, len(self.clus_idx)]
        else:
            self.spk_intervals = np.arange(0, len(self.clus_idx), 500)
            self.spk_intervals = np.append(self.spk_intervals, len(self.clus_idx))

        for idx in range(0, len(self.spk_intervals) - 1):
            item = QtWidgets.QListWidgetItem(str(self.spk_intervals[idx]) + ':' + str(self.spk_intervals[idx+1]))
            self.waveform_list.addItem(item)

        self.waveform_list.setCurrentRow(0)
        self.n_waveform = 0

    def compute_peth(self, trial_type, clust, trials_id):
        peths, bin = calculate_peths(self.spikes.times, self.spikes.clusters,
        [self.clust_ids[clust]], self.trials[trial_type][trials_id], self.t_before, self.t_after)

        peth_mean = peths.means[0, :]
        peth_std = peths.stds[0, :] / np.sqrt(len(trials_id))
        t_peth = peths.tscale

        return t_peth, peth_mean, peth_std

    def compute_rasters(self, trial_type, clust, trials_id):
        self.x = np.empty(0)
        self.y = np.empty(0)
        spk_times = self.spikes.times[self.spikes.clusters == self.clust_ids[clust]]
        for idx, val in enumerate(self.trials[trial_type][trials_id]):
            spks_to_include = np.bitwise_and(spk_times >= val - self.t_before, spk_times <= val + self.t_after)
            trial_spk_times = spk_times[spks_to_include]
            trial_spk_times_aligned = trial_spk_times - val
            trial_no = (np.ones(len(trial_spk_times_aligned))) * idx * 10
            self.x = np.append(self.x, trial_spk_times_aligned)
            self.y = np.append(self.y, trial_no)

        return self.x, self.y, self.n_trials

    def compute_autocorr(self, clust):
        self.clus_idx = np.where(self.spikes.clusters == self.clust_ids[clust])[0]

        x_corr = xcorr(self.spikes.times[self.clus_idx], self.spikes.clusters[self.clus_idx],
        self.autocorr_bin, self.autocorr_window)

        corr = x_corr[0, 0, :]

        return self.t_autocorr, corr

    def compute_template(self, clust):
        template = (self.clusters.waveforms[self.clust_ids[clust], :, 0]) * 1e6
        return self.t_template, template

    def compute_waveform(self, clust):
        if len(self.ephys_file_path) != 0:
            spk_times = self.spikes.times[self.clus_idx][self.spk_intervals[self.n_waveform]:self.spk_intervals[self.n_waveform + 1]]
            max_ch = self.clusters['channels'][self.clust_ids[clust]]
            wf = extract_waveforms(self.ephys_file_path, spk_times, max_ch, t = self.waveform_window, car = self.CAR)
            wf_mean = np.mean(wf[:,:,0], axis = 0)
            wf_std = np.std(wf[:,:,0], axis = 0)

            return self.t_waveform, wf_mean, wf_std

    def compute_timescales(self):
        self.t_autocorr = np.arange((self.autocorr_window/2) - self.autocorr_window, (self.autocorr_window/2)+ self.autocorr_bin, self.autocorr_bin)
        sr = 30000
        n_template = len(self.clusters.waveforms[self.ids[0], :, 0])
        self.t_template = 1e3 * (np.arange(n_template)) / sr

        n_waveform = int(sr / 1000 * (self.waveform_window))
        self.t_waveform = 1e3 * (np.arange(n_waveform)) / sr
        #self.t_waveform = 1e3 * (np.arange(n_waveform/2 - n_waveform, n_waveform/2, 1))/sr















