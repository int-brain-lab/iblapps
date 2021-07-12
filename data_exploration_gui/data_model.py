#
from one.api import ONE
import numpy as np
from brainbox.processing import compute_cluster_average
from brainbox.task.trials import find_trial_ids, get_event_aligned_raster
from brainbox.behavior.wheel import velocity
from brainbox.behavior.dlc import get_dlc_everything
from brainbox.population.decode import xcorr
from iblutil.util import Bunch
from iblutil.numerical import ismember
from data_exploration_gui.utils import colours
from PyQt5 import QtGui
one = ONE()
probe ='probe00'
# load in the data that you need given eid and pid
EPOCH = [-0.4, 1]
TBIN = 0.02
AUTOCORR_WINDOW = 0.1
AUTOCORR_BIN = 0.001
FS = 30000

class DataModel:
    def __init__(self, pid='f8d0ecdc-b7bd-44cc-b887-3d544e24e561'):
        eid, probe = one.pid2eid(pid)

        self.spikes = one.load_object(eid, obj='spikes', collection=f'alf/{probe}',
                             attribute='clusters|times|amps|depths')
        self.clusters = one.load_object(eid, obj='clusters', collection=f'alf/{probe}',
                               attribute='metrics|waveforms')

        # Get everything we need for the clusters
        # need to get rid of nans in amps and depths
        self.clusters.clust_ids, self.clusters.depths, n_spikes = \
            compute_cluster_average(self.spikes.clusters[~np.isnan(self.spikes.depths)],
                                    self.spikes.depths[~np.isnan(self.spikes.depths)])
        _ , self.clusters.amps, _ = compute_cluster_average(
            self.spikes.clusters[~np.isnan(self.spikes.amps)],
            self.spikes.amps[~np.isnan(self.spikes.amps)])

        # KS2 good mua colours
        # Bug in the KS2 units, the clusters that do not exist in spikes.clusters have not been
        # filled with Nans like the other features in metrics
        colours_ks = np.array(self.clusters.metrics.ks2_label[:len(self.clusters.clust_ids)])
        #colours_ks = np.array(self.clusters.metrics.ks2_label[self.clusters.clust_ids])
        good_ks = np.where(colours_ks == 'good')[0]
        colours_ks[good_ks] = colours['KS good']
        mua_ks = np.where(colours_ks == 'mua')[0]
        colours_ks[mua_ks] = colours['KS mua']
        self.clusters.colours_ks = colours_ks

        # IBL good bad colours
        colours_ibl = np.array(self.clusters.metrics.ks2_label[self.clusters.clust_ids])
        good_ibl = np.where(self.clusters.metrics.label[self.clusters.clust_ids] == 1)[0]
        colours_ibl[good_ibl] = colours['IBL good']
        bad_ibl = np.where(self.clusters.metrics.label[self.clusters.clust_ids] != 1)[0]
        colours_ibl[bad_ibl] = colours['IBL bad']
        self.clusters.colours_ibl = colours_ibl


        # Some extra info for sorting clusters
        self.clusters['ids'] = np.arange(len(self.clusters.clust_ids))
        self.clusters['n spikes'] = np.argsort(n_spikes)[::-1]
        self.clusters['KS good'] = np.r_[good_ks, mua_ks]
        self.clusters['IBL good'] = np.r_[good_ibl, bad_ibl]

        #self.sort_clusters('ids')

        # Get trial data
        self.trials = one.load_object(eid, obj='trials', collection='alf')
        self.n_trials = self.trials['probabilityLeft'].shape[0]
        self.trial_events = [key for key in self.trials.keys() if 'time' in key]

        # Get behaviour data
        wheel = one.load_object(eid, obj='wheel', collection='alf')
        dlc_left = one.load_object(eid, obj='leftCamera', collection='alf',
                                   attribute='times|dlc')
        dlc_right = one.load_object(eid, obj='rightCamera', collection='alf',
                                    attribute='times|dlc')

        self.behav, self.behav_events = self.combine_behaviour_data(wheel, dlc_left, dlc_right)

        self.spikes_raster = Bunch()
        self.spikes_raster_psth = Bunch()
        self.behav_raster = Bunch()
        self.behav_raster_psth = Bunch()


    def combine_behaviour_data(self, wheel, dlc_left, dlc_right):
        behav = Bunch()
        behav_options = []
        if all([val in wheel.keys() for val in ['timestamps', 'position']]):
            behav['wheel'] = wheel
            behav_options = behav_options + ['wheel']

        if all([val in dlc_left.keys() for val in ['times', 'dlc']]):
            dlc = get_dlc_everything(dlc_left, 'left')
            behav['leftCamera'] = dlc
            keys = [f'leftCamera_{key}' for key in dlc.dlc.keys() if 'speed' in key]
            behav_options = behav_options + keys
            keys = [f'leftCamera_{key}' for key in dlc.keys() if
                    any([key in name for name in ['licks', 'sniffs']])]
            behav_options = behav_options + keys

        if all([val in dlc_right.keys() for val in ['times', 'dlc']]):
            dlc = get_dlc_everything(dlc_right, 'right')
            behav['rightCamera'] = dlc
            keys = [f'rightCamera_{key}' for key in dlc.dlc.keys() if 'speed' in key]
            behav_options = behav_options + keys
            keys = [f'rightCamera_{key}' for key in dlc.keys() if
                    any([key in name for name in ['licks', 'sniffs']])]
            behav_options = behav_options + keys

        return behav, behav_options

    def sort_clusters(self, sort):
        clust_sort = Bunch()
        self.clust_ids = self.clusters.clust_ids[self.clusters[sort]]
        clust_sort['ids'] = self.clust_ids
        clust_sort['amps'] = self.clusters.amps[self.clusters[sort]] * 1e6
        clust_sort['depths'] = self.clusters.depths[self.clusters[sort]]
        clust_sort['colours_ks'] = self.clusters.colours_ks[self.clusters[sort]]
        clust_sort['colours_ibl'] = self.clusters.colours_ibl[self.clusters[sort]]

        return clust_sort

    def _get_spike_data_for_selection(self, clust, trial_event='goCue_times'):
        """
        Triggered when one of the filters is changed, inputs should only be the filter options
        :return:
        """
        # Store the clust id and trial event
        self.clust = clust
        self.trial_event = trial_event
        print(clust)
        print(self.clust_ids[clust])
        raster, t = get_event_aligned_raster(self.spikes.times[self.spikes.clusters ==
                                                               self.clust_ids[clust]],
                                             self.trials[trial_event], tbin=TBIN, epoch=EPOCH)
        self.spikes_raster_psth['vals'] = raster
        self.spikes_raster_psth['time'] = t
        self.spikes_raster_psth['ylabel'] = 'Firing Rate / Hz'
#
        #
        #raster, t = get_event_aligned_raster(self.spikes.times[self.spikes.clusters ==
        #                                                       self.clust_ids[clust]],
        #                         self.trials[trial_event], tbin=0.02)
        #self.spikes_raster['vals'] = raster
        #self.spikes_raster['time'] = t
        #self.spikes_raster['cmap'] = 'binary'
        #self.spikes_raster['clevels'] = [0, 1]

#
    def _get_spike_raster(self, trial_ids):
        # Ain't most efficient but it will do for now!

        data = Bunch()

        epoch = [0.4, 1]
        spk_times = self.spikes.times[self.spikes.clusters == self.clust_ids[self.clust]]
        for idx, val in enumerate(self.trials[self.trial_event][trial_ids]):
            spks_to_include = np.bitwise_and(spk_times >= val - epoch[0],
                                             spk_times <= val + epoch[1])
            trial_spk_times = spk_times[spks_to_include] - val
            if idx == 0:
                x = trial_spk_times
                y = np.ones(len(trial_spk_times)) * idx
            else:
                x = np.r_[x, trial_spk_times]
                y = np.r_[y, np.ones(len(trial_spk_times)) * idx]

        data['raster'] = np.c_[x, y]
        data['time'] = EPOCH
        data['n_trials'] = self.n_trials

        return data

    def get_autocorr_for_selection(self):
        data = Bunch()
        x_corr = xcorr(self.spikes.times[self.spikes.clusters == self.clust_ids[self.clust]],
                       self.spikes.clusters[self.spikes.clusters == self.clust_ids[self.clust]],
                       AUTOCORR_BIN, AUTOCORR_WINDOW)
        t_corr = np.arange(0, AUTOCORR_WINDOW + AUTOCORR_BIN, AUTOCORR_BIN) - AUTOCORR_WINDOW/2
        #t_corr = np.arange((AUTOCORR_WINDOW / 2) - AUTOCORR_WINDOW,
        #                    (AUTOCORR_WINDOW / 2) + AUTOCORR_BIN, AUTOCORR_BIN)

        data['vals'] = x_corr[0, 0, :]
        data['time'] = t_corr

        return data

    def get_template_for_selection(self):
        data = Bunch()
        template = (self.clusters.waveforms[self.clust_ids[self.clust], :, 0]) * 1e6
        t_template = 1e3 * np.arange(template.shape[0]) / FS

        data['vals'] = template
        data['time'] = t_template

        return data

    def _get_behaviour_data_for_selection(self, behav, trial_event='goCue_times'):
        """
        Triggered when one of the filters is changed, inputs should only be the filter options
        :return:
        """
        if behav == 'wheel':
            v = velocity(self.behav[behav].timestamps, self.behav[behav].position)
            raster, t = get_event_aligned_raster(self.behav[behav].timestamps,
                                                 self.trials[trial_event], values=v, tbin=TBIN,
                                                 epoch=EPOCH)
            self.behav_raster['vals'] = raster
            self.behav_raster['time'] = t
            self.behav_raster['cmap'] = 'viridis'
            self.behav_raster['clevels'] = np.nanquantile(raster, [0, 1])
            self.behav_raster['ylabel'] = 'rad / s'

        else:
            camera = behav.split('_')[0]
            behav = '_'.join(behav.split('_')[1:])

            if behav == 'licks' or behav == 'sniffs':
                raster, t = get_event_aligned_raster(self.behav[camera][behav],
                                                     self.trials[trial_event], tbin=TBIN,
                                                     epoch=EPOCH)
                self.behav_raster['vals'] = raster
                self.behav_raster['time'] = t
                self.behav_raster['cmap'] = 'binary'
                self.behav_raster['clevels'] = [0, 0.1]
                self.behav_raster['ylabel'] = 'Rate (Hz)'

            else:
                raster, t = get_event_aligned_raster(self.behav[camera]['times'],
                                                     self.trials[trial_event],
                                                     values=self.behav[camera]['dlc'][behav],
                                                     tbin=TBIN, epoch=EPOCH)
                self.behav_raster['vals'] = raster
                self.behav_raster['time'] = t
                self.behav_raster['cmap'] = 'viridis'
                self.behav_raster['clevels'] = np.nanquantile(raster, [0, 0.9])
                self.behav_raster['ylabel'] = 'px / s'


    def _get_psth(self, raster, trials_id, tbin=1):
        data = Bunch()
        data['psth_mean'] = np.nanmean(raster.vals[trials_id], axis=0) / tbin
        data['psth_std'] = (np.nanstd(raster.vals[trials_id], axis=0) / tbin) \
                           / np.sqrt(trials_id.shape[0])
        data['time'] = raster.time
        data['ylabel'] = raster.ylabel


        return data

    def _get_raster(self, raster, trials_id):
        data = Bunch()
        data['raster'] = np.r_[raster.vals[trials_id],
                               np.full((self.n_trials-trials_id.shape[0],
                                        raster.vals.shape[1]), np.nan)]
        data['time'] = raster.time
        data['cmap'] = raster.cmap
        data['clevels'] = raster.clevels

        return data

    def get_psths_for_selection(self, side, choice, order, sort, contrast, event):
        trial_ids, _ = find_trial_ids(self.trials, side=side, choice=choice, order=order,
                                      sort=sort, contrast=contrast, event=event)
        spike_psth = self._get_psth(self.spikes_raster_psth, trial_ids, tbin=TBIN)
        behav_psth = self._get_psth(self.behav_raster, trial_ids)

        return spike_psth, behav_psth

    def get_rasters_for_selection(self, side, choice, order, sort, contrast, event):
        trial_ids, div = find_trial_ids(self.trials, side=side, choice=choice, order=order,
                                        sort=sort, contrast=contrast, event=event)
        dividers = self._get_dividers(div, trial_ids.shape[0])
        #spike_raster = self._get_raster(self.spikes_raster, trial_ids)
        spike_raster = self._get_spike_raster(trial_ids)
        spike_raster['dividers'] = dividers
        behav_raster = self._get_raster(self.behav_raster, trial_ids)
        behav_raster['dividers'] = dividers

        return spike_raster, behav_raster

    def _get_dividers(self, div, n_trials):
        if len(div) == 0:
            return div
        else:
            div.insert(0, 0)
            div.append(n_trials)
            dividers = []
            for i in range(len(div) - 1):
                dividers.append([div[i], div[i + 1]])

            return dividers
