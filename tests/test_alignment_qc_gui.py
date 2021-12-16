import unittest
from pathlib import Path
import re
from inspect import getmembers, isfunction

import numpy as np

from one.api import ONE
from ibllib.atlas import AllenAtlas
from atlaselectrophysiology.load_data import LoadData
from ibllib.pipes.ephys_alignment import EphysAlignment
from ibllib.pipes.misc import create_alyx_probe_insertions
from ibllib.pipes.histology import register_track


EPHYS_SESSION = 'b1c968ad-4874-468d-b2e4-5ffa9b9964e9'
one = ONE(base_url='https://test.alyx.internationalbrainlab.org')
brain_atlas = AllenAtlas(25)


class TestsAlignmentQcGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        probe = ['probe00', 'probe01']
        insertions = create_alyx_probe_insertions(session_path=EPHYS_SESSION, model='3B2',
                                                  labels=probe, one=one, force=True)
        print(insertions)
        cls.probe_id = one.alyx.rest('insertions', 'list', session=EPHYS_SESSION,
                                     name='probe00', no_cache=True)[0]['id']
        cls.probe_id2 = one.alyx.rest('insertions', 'list', session=EPHYS_SESSION,
                                      name='probe01', no_cache=True)[0]['id']
        data = np.load(Path(Path(__file__).parent.
                            joinpath('fixtures', 'data_alignmentqc_gui.npz')), allow_pickle=True)
        cls.xyz_picks = data['xyz_picks']
        cls.alignments = data['alignments'].tolist()
        cls.cluster_chns = data['cluster_chns']
        register_track(cls.probe_id, picks=cls.xyz_picks, one=one, overwrite=True,
                       channels=False, brain_atlas=brain_atlas)
        register_track(cls.probe_id2, picks=cls.xyz_picks, one=one, overwrite=True,
                       channels=False, brain_atlas=brain_atlas)

    def setUp(self) -> None:
        self.resolved_key = '2020-09-14T15:44:56_nate'
        self.ld = LoadData(one=one, brain_atlas=brain_atlas, testing=True, probe_id=self.probe_id)
        _ = self.ld.get_xyzpicks()
        self.ld.cluster_chns = self.cluster_chns
        _ = self.ld.get_previous_alignments()
        _ = self.ld.get_starting_alignment(0)
        self.ephysalign = EphysAlignment(self.ld.xyz_picks, self.ld.chn_depths,
                                         brain_atlas=self.ld.brain_atlas)
        traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                             provenance='Ephys aligned histology track', no_cache=True)
        if traj:
            self.prev_traj_id = traj[0]['id']

    def test_alignments(self):
        checks = getmembers(TestsAlignmentQcGUI,
                            lambda x: isfunction(x) and re.match(r'^_\d{2}_.*', x.__name__))
        # Run each function in order
        for name, fn in sorted(checks, key=lambda x: x[0]):
            if not name.startswith('_01_'):
                self.setUp()
            fn(self)

    def _01_no_alignment(self):

        prev_align = self.ld.get_previous_alignments()
        assert (len(prev_align) == 1)
        assert (prev_align[0] == 'original')
        feature, track = self.ld.get_starting_alignment(0)
        assert (not feature)
        assert (not track)
        assert (not self.ld.alignments)
        assert (self.ld.resolved == 0)

    def _02_one_alignment(self):
        key = '2020-07-26T17:06:58_alejandro'
        feature = self.alignments[key][0]
        track = self.alignments[key][1]
        xyz_channels = self.ephysalign.get_channel_locations(feature, track)
        self.ld.upload_data(xyz_channels, channels=False)
        self.ld.update_alignments(np.array(feature), np.array(track), key_info=key)
        _ = self.ld.get_previous_alignments()
        _ = self.ld.get_starting_alignment(0)
        assert (self.ld.current_align == key)

        traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                             provenance='Ephys aligned histology track', no_cache=True)
        assert (sorted(list(traj[0]['json'].keys()), reverse=True)[0] == key)
        assert (len(traj[0]['json']) == 1)
        # Not added user evaluation so should just contain feature and track lists
        assert(len(traj[0]['json'][key]) == 2)

        self.ld.update_qc(upload_flatiron=False)
        insertion = one.alyx.rest('insertions', 'read', id=self.probe_id, no_cache=True)
        assert (insertion['json']['extended_qc']['alignment_count'] == 1)
        assert (insertion['json']['extended_qc']['alignment_stored'] == key)
        assert (not insertion['json']['extended_qc'].get('experimenter', None))
        assert (insertion['json']['qc'] == 'NOT_SET')
        assert (self.ld.resolved == 0)

    def _03_same_user(self):
        key = '2020-08-26T17:06:58_alejandro'
        eval_str = 'PASS: Noise and artifact'
        feature = self.alignments[key][0]
        track = self.alignments[key][1]
        xyz_channels = self.ephysalign.get_channel_locations(feature, track)
        self.ld.upload_data(xyz_channels, channels=False)
        self.ld.update_alignments(np.array(feature), np.array(track), key_info=key,
                                  user_eval=eval_str)
        _ = self.ld.get_previous_alignments()
        _ = self.ld.get_starting_alignment(0)
        assert (self.ld.current_align == key)

        traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                             provenance='Ephys aligned histology track', no_cache=True)
        traj_id = traj[0]['id']
        assert (sorted(list(traj[0]['json'].keys()), reverse=True)[0] == key)
        assert (len(traj[0]['json']) == 1)
        assert(len(traj[0]['json'][key]) == 3)
        assert (traj_id != self.prev_traj_id)

        self.ld.update_qc(upload_flatiron=False)
        insertion = one.alyx.rest('insertions', 'read', id=self.probe_id, no_cache=True)
        assert (insertion['json']['extended_qc']['alignment_count'] == 1)
        assert (insertion['json']['extended_qc']['alignment_stored'] == key)
        assert (insertion['json']['extended_qc']['alignment_resolved'] == 0)
        assert(insertion['json']['extended_qc']['experimenter'] == 'PASS')
        assert (insertion['json']['qc'] == 'PASS')
        assert (self.ld.resolved == 0)

    def _04_two_alignments(self):
        key = '2020-09-14T15:42:22_guido'
        feature = self.alignments[key][0]
        track = self.alignments[key][1]
        xyz_channels = self.ephysalign.get_channel_locations(feature, track)
        self.ld.upload_data(xyz_channels, channels=False)
        self.ld.update_alignments(np.array(feature), np.array(track), key_info=key)
        _ = self.ld.get_previous_alignments()
        _ = self.ld.get_starting_alignment(0)

        assert (self.ld.current_align == key)

        traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                             provenance='Ephys aligned histology track', no_cache=True)
        traj_id = traj[0]['id']
        assert (sorted(list(traj[0]['json'].keys()), reverse=True)[0] == key)
        assert (len(traj[0]['json']) == 2)
        assert(len(traj[0]['json'][key]) == 2)
        assert (traj_id != self.prev_traj_id)
        # Also assert all the keys match

        self.ld.update_qc(upload_flatiron=False)
        insertion = one.alyx.rest('insertions', 'read', id=self.probe_id, no_cache=True)
        assert (insertion['json']['qc'] == 'PASS')
        assert (insertion['json']['extended_qc']['experimenter'] == 'PASS')
        assert (insertion['json']['extended_qc']['alignment_count'] == 2)
        assert (insertion['json']['extended_qc']['alignment_stored'] == key)
        assert (insertion['json']['extended_qc']['alignment_resolved'] == 0)
        assert (insertion['json']['extended_qc']['alignment_qc'] < 0.8)
        assert (self.ld.resolved == 0)

    def _05_three_alignments(self):

        key = '2020-09-14T15:44:56_nate'
        eval_str = 'WARNING: Drift'
        feature = self.alignments[key][0]
        track = self.alignments[key][1]
        xyz_channels = self.ephysalign.get_channel_locations(feature, track)
        self.ld.upload_data(xyz_channels, channels=False)
        self.ld.update_alignments(np.array(feature), np.array(track), key_info=key,
                                  user_eval=eval_str)
        _ = self.ld.get_previous_alignments()
        _ = self.ld.get_starting_alignment(0)

        assert (self.ld.current_align == key)

        traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                             provenance='Ephys aligned histology track', no_cache=True)
        traj_id = traj[0]['id']
        assert (len(traj[0]['json']) == 3)
        assert (sorted(list(traj[0]['json'].keys()), reverse=True)[0] == key)
        assert(len(traj[0]['json'][key]) == 3)
        assert (traj_id != self.prev_traj_id)

        self.ld.update_qc(upload_flatiron=False)
        insertion = one.alyx.rest('insertions', 'read', id=self.probe_id, no_cache=True)
        assert (insertion['json']['qc'] == 'WARNING')
        assert (insertion['json']['extended_qc']['experimenter'] == 'WARNING')
        assert (insertion['json']['extended_qc']['alignment_count'] == 3)
        assert (insertion['json']['extended_qc']['alignment_stored'] == key)
        assert (insertion['json']['extended_qc']['alignment_resolved'] == 1)
        assert (insertion['json']['extended_qc']['alignment_resolved_by'] == 'qc')
        assert (insertion['json']['extended_qc']['alignment_qc'] > 0.8)
        assert(self.ld.resolved == 1)

    def _06_new_user_after_resolved(self):
        key = '2020-09-16T15:44:56_mayo'
        eval_str = 'PASS: Drift'
        feature = self.alignments[key][0]
        track = self.alignments[key][1]
        xyz_channels = self.ephysalign.get_channel_locations(feature, track)
        self.ld.upload_data(xyz_channels, channels=False)
        self.ld.update_alignments(np.array(feature), np.array(track), key_info=key,
                                  user_eval=eval_str)
        _ = self.ld.get_previous_alignments()
        _ = self.ld.get_starting_alignment(0)

        assert (self.ld.current_align == key)

        traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                             provenance='Ephys aligned histology track', no_cache=True)
        traj_id = traj[0]['id']
        assert (len(traj[0]['json']) == 4)
        assert (sorted(list(traj[0]['json'].keys()), reverse=True)[0] == key)
        assert (traj_id == self.prev_traj_id)

        self.ld.update_qc(upload_flatiron=False)
        insertion = one.alyx.rest('insertions', 'read', id=self.probe_id, no_cache=True)
        assert (insertion['json']['qc'] == 'WARNING')
        assert (insertion['json']['extended_qc']['experimenter'] == 'WARNING')
        assert (insertion['json']['extended_qc']['alignment_count'] == 4)
        assert (insertion['json']['extended_qc']['alignment_stored'] == self.resolved_key)
        assert (insertion['json']['extended_qc']['alignment_resolved'] == 1)
        assert (insertion['json']['extended_qc']['alignment_resolved_by'] == 'qc')
        assert (insertion['json']['extended_qc']['alignment_qc'] > 0.8)
        assert (self.ld.resolved == 1)

    def _07_same_user_after_resolved(self):
        key = '2020-10-14T15:44:56_nate'
        eval_str = 'CRITICAL: Brain Damage'
        feature = self.alignments[key][0]
        track = self.alignments[key][1]
        xyz_channels = self.ephysalign.get_channel_locations(feature, track)
        self.ld.upload_data(xyz_channels, channels=False)
        self.ld.update_alignments(np.array(feature), np.array(track), key_info=key,
                                  user_eval=eval_str)
        _ = self.ld.get_previous_alignments()
        _ = self.ld.get_starting_alignment(0)

        assert (self.ld.current_align == key)

        traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                             provenance='Ephys aligned histology track', no_cache=True)
        traj_id = traj[0]['id']
        assert (sorted(list(traj[0]['json'].keys()), reverse=True)[0] == key)
        assert (len(traj[0]['json']) == 5)
        assert (traj_id == self.prev_traj_id)

        self.ld.update_qc(upload_flatiron=False)
        insertion = one.alyx.rest('insertions', 'read', id=self.probe_id, no_cache=True)
        assert (insertion['json']['qc'] == 'CRITICAL')
        assert (insertion['json']['extended_qc']['experimenter'] == 'CRITICAL')
        assert (insertion['json']['extended_qc']['alignment_count'] == 5)
        assert (insertion['json']['extended_qc']['alignment_stored'] == self.resolved_key)
        assert (insertion['json']['extended_qc']['alignment_resolved'] == 1)
        assert (insertion['json']['extended_qc']['alignment_resolved_by'] == 'qc')
        assert (insertion['json']['extended_qc']['alignment_qc'] > 0.8)
        assert (self.ld.resolved == 1)

    def _08_starting_alignments(self):
        key = '2020-10-14T15:44:56_nate'
        # Starting from original
        self.ld.probe_id = self.probe_id2
        prev_align = self.ld.get_previous_alignments()
        assert(len(prev_align) == 1)
        assert(prev_align[0] == 'original')
        assert(self.ld.alignments == {})

        # Loading in one with alignments
        self.ld.probe_id = self.probe_id
        prev_align = self.ld.get_previous_alignments()
        assert(len(prev_align) == 6)
        assert(prev_align[-1] == 'original')
        assert(len(self.ld.alignments) == 5)
        assert(sorted(self.ld.alignments, reverse=True)[0] == key)

        # Now loading back one with nothing
        self.ld.probe_id = self.probe_id2
        prev_align = self.ld.get_previous_alignments()
        assert(len(prev_align) == 1)
        assert(prev_align[0] == 'original')
        assert(self.ld.alignments == {})

    @classmethod
    def tearDownClass(cls) -> None:
        one.alyx.rest('insertions', 'delete', id=cls.probe_id)
        one.alyx.rest('insertions', 'delete', id=cls.probe_id2)


if __name__ == "__main__":
    unittest.main(exit=False)
