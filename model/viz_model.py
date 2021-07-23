# if single session
from dataclasses import dataclass, field
from iblutil.io.parquet import np2str, str2np
from iblutil.io import parquet
from iblutil.util import Bunch
from one.api import ONE
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Session level
# trials
# dlc
# wheel
# video

# Probe level
# spikes
# clusters
# channels
# spikeglx

# Cluster level
# spike clusters
# cluster
# cluster waveform
# spike waveform
# cluster channels

# Spike level
# ????

# TODO write tests obassly
# TODO look into converting to dataclass?
# TODO am I just making a bad version of phylib? probably :/

# When dealing on a single session level we don't care about the cluster uuids - index is enough
# When dealing across sessions we need to care about cluster uuids


class VizModel:
    def __init__(self, use_tables=False, table_dir=None, one=None):

        if use_tables:
            assert table_dir
            self.single_session = False
            self.tables = self._load_tables(table_dir)
        else:
            self.single_session = True
            self.tables = None

        self.lazy = True
        self.eid = None
        self.pid = None
        self.one = one or ONE()

        self.ids = Bunch()

        # what is our entry point, pid or eid or cid (will depend on case)

    def _load_tables(self, table_dir):
        """
        Load in the trials, clusters, eid2pid and pid2cid tables (for now may be more)
        :param table_dir:
        :return:
        """
        # Super basic, in future make it like load_cache in one.api
        tables = Bunch()
        for table_file in Path(table_dir).glob('*.pqt'):
            table_type = table_file.stem
            table, _ = parquet.load(table_file)

            # sort the tables
            is_sorted = (table.index.is_monotonic_increasing
                         if isinstance(table.index, pd.MultiIndex)
                         else True)
            # Sorting makes MultiIndex indexing O(N) -> O(1)
            if not is_sorted:
                table.sort_index(inplace=True)

            tables[table_type] = table

        return tables


    def get_cluster_info(self, cid):
        """
        Get eid, pid, probe_name, cluster number for cluster uuid
        :return:
        """
        # return session, probe id, cluster number
        pass

    def find_cluster(self):
        # find clusters based on some search criterion
        # this can be xyz pos, atlas_id, probe insertion, cluster id, probe insertion and
        # and cluster no
        # returns you a list of clusters
        # and everything in the model is updated to reflect the current cluster
        # so this will be session level and probe level, cluster level
        # if just an id, need to know what context we are in
        pass

    def _get_meta(self, feature, keys=None):
        meta = self.ids.get(feature, None)
        if not meta:
            feature_meta = Bunch()
            for key in keys:
                feature_meta[key] = None
            self.ids[feature] = feature_meta
            return feature_meta
        else:
            return meta



    def get_trials(self, eid, as_frame=False):
        """
        Get trials for a given session
        :param eid: session id
        :param as_frame: return as dataframe or bunch
        :return:
        """
        trial_meta = self._get_meta('trials', keys=['eid'])

        if trial_meta['eid'] != eid:
            self.trials = self._load_trials(eid)
            # need a better way, because trials may be updated but perhaps dlc isn't
            self.ids.trials['eid'] = eid

        if as_frame:
            return self.trials
        else:
            return self.df_to_bunch(self.trials)

    def _load_trials(self, eid):
        """
        Load trials for a given session
        :param eid:
        :return:
        """
        #
        if self.single_session:
            trials = self.one.load_object(eid, obj='trials', collection='alf')
            # Maybe this isn't necessary, just to have consistency across single_session
            # vs multi_session but probs a waste of time
            np_eid = str2np(eid) # TODO make generic
            trials = trials.to_df()
            trials['eid0'] = np_eid[0, 0]
            trials['eid1'] = np_eid[0, 1]
            trials.set_index(['eid0', 'eid1'], inplace=True)
        else:
            trials = self.tables['trials'][self.session_mask(eid, self.tables.trials)]

        return trials

    def get_spikes(self, pid, as_frame=False, attributes=['times', 'clusters', 'depths', 'amps']):
        # load in spikes for a single probe
        spike_meta = self._get_meta('spikes', keys=['pid'])
        if spike_meta['pid'] != pid:
            self.spikes = self._load_spikes(pid, attributes=attributes)
            self.ids.spikes['pid'] = pid

        if as_frame:
            return self.spikes
        else:
            return self.df_to_bunch(self.spikes)


    def _load_spikes(self, pid, attributes):
        eid, probe = self.one.pid2eid(pid)
        spikes = self.one.load_object(eid, obj='spikes', collection=f'alf/{probe}',
                                      attribute=attributes)

        # Maybe this isn't necessary, just to have consistency across single_session
        # vs multi_session but probs a waste of time
        uuids = self.get_cluster_uuids(pid)
        np_id = str2np(uuids)
        spikes = spikes.to_df()
        spikes['cid0'] = np_id[:, 0][spikes.clusters]
        spikes['cid1'] = np_id[:, 1][spikes.clusters]
        spikes.set_index(['cid0', 'cid1'], inplace=True)

        return spikes

    def get_clusters(self, pid, as_frame=False, attributes=['amps', 'depths', 'uuids']):

        cluster_meta = self._get_meta('clusters', keys=['pid'])
        if cluster_meta['pid'] != pid:
            self.clusters = self._load_clusters(pid, attributes)
            self.ids.clusters['pid'] = pid

        if as_frame:
            return self.clusters
        else:
            return self.df_to_bunch(self.clusters)

    def _load_clusters(self, pid, attributes):
        if self.single_session:
            eid, probe = self.one.pid2eid(pid)
            clusters = self.one.load_object(eid, obj='clusters', collection=f'alf/{probe}',
                                            attribute=attributes)
            # Don't bother for single session
            self.uuids = clusters.pop('uuids').values.ravel()
            np_id = str2np(self.uuids)
            clusters['cid0'] = np_id[:, 0]
            clusters['cid1'] = np_id[:, 1]
            clusters = clusters.to_df()
            clusters.set_index(['cid0', 'cid1'], inplace=True)
        else:
            clusters = self.tables['clusters'][self.cluster_mask(pid, self.tables['pid2cid'])]

        return clusters

    def get_cluster_uuids(self, pid):
        # need this to be for single session or not single session
        # need this to be more clever yah!
        if self.single_session:
            # but this requires clusters to be loaded in before spikes
            return self.uuids
        else:

            uuids = np2str(np.array((self.tables['clusters']
                    [self.cluster_mask(pid, self.tables['pid2cid'])].index.values.tolist())))
        return uuids

    def get_cluster_spikes(self, cid, as_frame=False):

        # need to deal with some case where the cid is an int and a probe id or something
        self.cluster_spikes = self._load_cluster_spikes(cid)
        if as_frame:
            return self.cluster_spikes
        else:
            return self.df_to_bunch(self.cluster_spikes)

    def _load_cluster_spikes(self, cid):
        np_cid = str2np(cid)
        mask = (self.spikes.index.isin([np_cid[0, 0]], level='cid0') &
                self.spikes.index.isin([np_cid[0, 1]], level='cid1'))
        cluster_spikes = self.spikes[mask]

        return cluster_spikes


    def _id_to_cid(self, clust_idx):
        # something that transforms and id to a cid
        return self.uuids[clust_idx]




    def get_cluster_waveforms(self, cid, type='sample', chns='max'):

        # if in single session mode: assert cid is int



        # either template or sample
        # chns = 'max or 'all'
        # return the waveforms of the clusters, both the template and also the sample waveforms
        # if requested
        pass

    def change_default_attributes(self):
        pass


    def session_mask(self, eid, df, index=True):
        np_eid = str2np(eid)
        if index:
            mask = (df.index.isin([np_eid[0, 0]], level='eid0')
                    & df.index.isin([np_eid[0, 1]], level='eid1'))
        else:
            mask = (df.eid0.isin([np_eid[0, 0]]) & df.eid1.isin([np_eid[0, 1]])).values
        return mask

    def cluster_mask(self, pid, df, index=True):
        np_pid = str2np(pid)
        if index:
            mask = (df.index.isin([np_pid[0, 0]], level='pid0')
                    & df.index.isin([np_pid[0, 1]], level='pid1'))
        else:
            mask = (df.eid0.isin([np_pid[0, 0]]) & df.eid1.isin([np_pid[0, 1]])).values
        return mask

    def set_index(self, id, df, type='eid'):
        np_id = str2np(id)
        df[f'{type}0'] = np_id[0,0]
        df[f'{type}1'] = np_id[0,1]
        df.set_index([f'{type}0', f'{type}1'], inplace=True)
        return df

    def df_to_bunch(self, df):
        """
        convert dataframe to bunch
        :param df:
        :return:
        """

        data = Bunch()
        for col in df.columns:
            data[col] = df[col].values
        return data


    def get_dlc(self, session):
        pass



# just get out the spikes etc that you need for a single session









