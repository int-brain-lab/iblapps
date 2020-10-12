import unittest
from oneibl.one import ONE
from ibllib.atlas import AllenAtlas
from atlaselectrophysiology.load_data import LoadData
from ibllib.pipes.ephys_alignment import EphysAlignment
from ibllib.pipes.misc import create_alyx_probe_insertions
from ibllib.qc.alignment_qc import AlignmentQC
from ibllib.pipes.histology import register_track
import numpy as np
import copy

EPHYS_SESSION = 'b1c968ad-4874-468d-b2e4-5ffa9b9964e9'
one = ONE(username='test_user', password='TapetesBloc18',
               base_url='https://test.alyx.internationalbrainlab.org')
brain_atlas = AllenAtlas(25)

class TestProbeInsertion(unittest.TestCase):

    def test_creation(self):
        probe = ['probe00', 'probe01']
        create_alyx_probe_insertions(session_path=EPHYS_SESSION, model='3B2', labels=probe,
                                     one=one, force=True)
        insertion = one.alyx.rest('insertions', 'list', session=EPHYS_SESSION)
        assert(len(insertion) == 2)
        assert (insertion[0]['json']['qc'] == 'NOT_SET')
        assert (len(insertion[0]['json']['extended_qc']) == 0)


class TestHistologyQc(unittest.TestCase):

    def test_session_creation(self):
        pass


    def test_probe_qc(self):
        pass

class TestTracingQc(unittest.TestCase):

    def setUp(self) -> None:
        self.probe00_id = one.alyx.rest('insertions', 'list', session=EPHYS_SESSION,
                                        name='probe00')[0]['id']
        self.probe01_id = one.alyx.rest('insertions', 'list', session=EPHYS_SESSION,
                                        name='probe01')[0]['id']
        data = np.load('data_alignmentqc_gui.npz', allow_pickle=True)
        self.xyz_picks = data['xyz_picks']

    def test_tracing_exists(self):
        register_track(self.probe00_id, picks=self.xyz_picks, one=one, overwrite=True,
                       channels=False)
        insertion = one.alyx.rest('insertions', 'read', id=self.probe00_id)
        # TODO figure this one out
        # assert(np.all(np.ravel(np.array(insertion['json']['xyz_picks'])) == np.ravel((xyz_picks * 1e6))))
        assert (insertion['json']['qc'] == 'NOT_SET')
        assert (insertion['json']['extended_qc']['_tracing_exists'] == 1)

    def test_tracing_not_exists(self):
        register_track(self.probe01_id, picks=None, one=one, overwrite=True,
                       channels=False)
        insertion = one.alyx.rest('insertions', 'read', id=self.probe01_id)
        assert (insertion['json']['qc'] == 'CRITICAL')
        assert (insertion['json']['extended_qc']['_tracing_exists'] == 0)

    def tearDown(self) -> None:
        one.alyx.rest('insertions', 'delete', id=self.probe01_id)
        traj = one.alyx.rest('trajectories', 'list', probe_insertion=self.probe01_id,
                             provenance='Histology track')
        one.alyx.rest('trajectories', 'delete', id=traj[0]['id'])
        traj = one.alyx.rest('trajectories', 'list', probe_insertion=self.probe00_id,
                             provenance='Histology track')
        one.alyx.rest('trajectories', 'delete', id=traj[0]['id'])



class TestsAlignmentQcGUI(unittest.TestCase):

    def setUp(self):
        self.probe_id = one.alyx.rest('insertions', 'list', session=EPHYS_SESSION,
                                      name='probe00')[0]['id']
        data = np.load('data_alignmentqc_gui.npz', allow_pickle=True)
        self.xyz_picks = data['alignments']
        self.alignments = data['xyz_picks'].tolist()
        self.cluster_chns = data['cluster_chns']
        self.ld = LoadData(one=one, brain_atlas=brain_atlas, testing=True, probe_id=self.probe_id)
        _ = self.ld.get_xyzpicks()
        self.ld.cluster_chns = self.cluster_chns
        self.ephysalign = EphysAlignment(self.ld.xyz_picks, self.ld.chn_depths,
                                         brain_atlas=self.ld.brain_atlas)
        self.all_keys = []

    def test_no_alignment(self):

        prev_align = self.ld.get_previous_alignments()
        assert (len(prev_align) == 1)
        assert (prev_align[0] == 'original')
        feature, track = self.ld.get_starting_alignment(0)
        assert (not feature)
        assert (not track)
        assert (not self.ld.alignments)
        assert (self.ld.resolved == 0)

    def test_one_alignment(self):
        key = '2020-07-26T17:06:58_alejandro'
        self.feature = self.alignments[key][0]
        self.track = self.alignments[key][1]
        self.xyz_channels = self.ephysalign.get_channel_locations(self.feature, self.track)
        self.ld.upload_data(self.xyz_channels, channels=False)
        self.ld.update_alignments(np.array(self.feature), np.array(self.track), key_info=key)
        _ = self.ld.get_previous_alignments()
        _ = self.ld.get_starting_alignment(0)
        assert (self.ld.current_align == key)

        traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                             provenance='Ephys aligned histology track')
        self.prev_traj_id = traj[0]['id']
        assert (traj[0]['json'].key() == key)
        assert (len(traj[0]['json']) == 1)

        self.ld.update_qc()
        insertion = one.alyx.rest('insertions', 'read', id=self.probe_id)
        assert (insertion['json']['extended_qc']['_alignment_number'] == 1)
        assert (insertion['json']['extended_qc']['_alignment_stored'] == key)
        assert (insertion['json']['qc'] == 'NOT_SET')
        assert (self.ld.resolved == 0)

    def test_same_user(self):
        key = '2020-08-26T17:06:58_alejandro'
        self.all_keys.append(key)
        self.ld.upload_data(self.xyz_channels, channels=False)
        self.ld.update_alignments(np.array(self.feature), np.array(self.track), key_info=key)
        _ = self.ld.get_previous_alignments()
        _ = self.ld.get_starting_alignment(0)
        assert (self.ld.current_align == key)

        traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                             provenance='Ephys aligned histology track')
        traj_id = traj[0]['id']
        assert (list(traj[0]['json'].keys()) == self.all_keys)
        assert (len(traj[0]['json']) == 1)
        assert (traj_id != self.prev_traj_id)
        self.prev_traj_id = traj_id

        self.ld.update_qc()
        insertion = one.alyx.rest('insertions', 'read', id=self.probe_id)
        assert (insertion['json']['extended_qc']['_alignment_number'] == 1)
        assert (insertion['json']['extended_qc']['_alignment_stored'] == key)
        assert (insertion['json']['qc'] == 'NOT_SET')
        assert (self.ld.resolved == 0)

    def test_two_alignments(self):
        key = '2020-09-14T15:42:22_guido'
        self.all_keys.append()
        feature = self.alignments[key][0]
        track = self.alignments[key][1]
        xyz_channels = self.ephysalign.get_channel_locations(feature, track)
        self.ld.upload_data(xyz_channels, channels=False)
        self.ld.update_alignments(np.array(feature), np.array(track), key_info=key)
        _ = self.ld.get_previous_alignments()
        _ = self.ld.get_starting_alignment(0)

        assert (self.ld.current_align == key)

        traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                             provenance='Ephys aligned histology track')
        traj_id = traj[0]['id']
        assert (list(traj[0]['json'].keys()) == self.all_keys)
        assert (len(traj[0]['json']) == 2)
        assert (traj_id != self.prev_traj_id)
        self.prev_traj_id = traj_id
        # Also assert all the keys match

        self.ld.update_qc()
        insertion = one.alyx.rest('insertions', 'read', id=self.probe_id)
        assert (insertion['json']['qc'] == 'WARNING')
        assert (insertion['json']['extended_qc']['_alignment_number'] == 2)
        assert (insertion['json']['extended_qc']['_alignment_stored'] == key)
        assert (insertion['json']['extended_qc']['_alignment_resolved'] == 0)
        assert (insertion['json']['extended_qc']['_alignment_qc'] < 0.8)
        assert (self.ld.resolved == 0)

    def test_three_alignments(self):

        self.key = '2020-09-14T15:44:56_nate'
        self.all_keys.append(self.key)
        feature = self.alignments[self.key][0]
        track = self.alignments[self.key][1]
        xyz_channels = self.ephysalign.get_channel_locations(feature, track)
        self.ld.upload_data(xyz_channels, channels=False)
        self.ld.update_alignments(np.array(feature), np.array(track), key_info=self.key)
        _ = self.ld.get_previous_alignments()
        _ = self.ld.get_starting_alignment(0)

        assert (self.ld.current_align == self.key)

        traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                             provenance='Ephys aligned histology track')
        traj_id = traj[0]['id']
        assert (len(traj[0]['json']) == 3)
        assert (list(traj[0]['json'].keys()) == self.all_keys)
        assert (traj_id != self.prev_traj_id)
        self.prev_traj_id = traj_id

        self.ld.update_qc()
        insertion = one.alyx.rest('insertions', 'read', id=self.probe_id)
        assert (insertion['json']['qc'] == 'PASS')
        assert (insertion['json']['extended_qc']['_alignment_number'] == 3)
        assert (insertion['json']['extended_qc']['_alignment_stored'] == self.key)
        assert (insertion['json']['extended_qc']['_alignment_resolved'] == 1)
        assert (insertion['json']['extended_qc']['_alignment_qc'] > 0.8)
        assert(self.ld.resolved == 1)

    def test_new_user_after_resolved(self):
        key = '2020-09-16T15:44:56_mayo'
        self.all_keys.append(key)
        self.ld.upload_data(self.xyz_channels, channels=False)
        self.ld.update_alignments(np.array(self.feature), np.array(self.track), key_info=self.key)
        _ = self.ld.get_previous_alignments()
        _ = self.ld.get_starting_alignment(0)

        assert (self.ld.current_align == key)

        traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                             provenance='Ephys aligned histology track')
        traj_id = traj[0]['id']
        assert (len(traj[0]['json']) == 4)
        assert (list(traj[0]['json'].keys()) == self.all_keys)
        assert (traj_id == self.prev_traj_id)

        self.ld.update_qc()
        insertion = one.alyx.rest('insertions', 'read', id=self.probe_id)
        assert (insertion['json']['qc'] == 'PASS')
        assert (insertion['json']['extended_qc']['_alignment_number'] == 4)
        assert (insertion['json']['extended_qc']['_alignment_stored'] == self.key)
        assert (insertion['json']['extended_qc']['_alignment_resolved'] == 1)
        assert (insertion['json']['extended_qc']['_alignment_qc'] > 0.8)
        assert (self.ld.resolved == 1)

    def test_same_user_after_resolved(self):
        key = '2020-10-14T15:44:56_nate'
        self.all_keys.append(key)
        self.ld.upload_data(self.xyz_channels, channels=False)
        self.ld.update_alignments(np.array(self.feature), np.array(self.track), key_info=self.key)
        _ = self.ld.get_previous_alignments()
        _ = self.ld.get_starting_alignment(0)

        assert (self.ld.current_align == key)

        traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                             provenance='Ephys aligned histology track')
        traj_id = traj[0]['id']
        assert (list(traj[0]['json'].keys()) == self.all_keys)
        assert (len(traj[0]['json']) == 5)
        assert (traj_id == self.prev_traj_id)

        self.ld.update_qc()
        insertion = one.alyx.rest('insertions', 'read', id=self.probe_id)
        assert (insertion['json']['qc'] == 'PASS')
        assert (insertion['json']['extended_qc']['_alignment_number'] == 5)
        assert (insertion['json']['extended_qc']['_alignment_stored'] == self.key)
        assert (insertion['json']['extended_qc']['_alignment_resolved'] == 1)
        assert (insertion['json']['extended_qc']['_alignment_qc'] > 0.8)
        assert (self.ld.resolved == 1)

    def tearDown(self) -> None:
        one.alyx.rest('trajectories', 'delete', id=self.prev_traj_id)
        one.alyx.rest('insertions', 'delete', id=self.probe_id)



class TestAlignmentQcExisting(unittest.TestCase):

        def setUp(self):
            probe = one.alyx.rest('insertions', 'list', session=EPHYS_SESSION, probe='probe00')
            json = self.probe_id
            # create insertion with xyz_picks
            # create trajectory ephys aligned with json fields
            data = np.load('data_alignmentqc_existing.npz', allow_pickle=True)
            self.xyz_picks = data['alignments']
            self.alignments = data['xyz_picks'].tolist()
            self.cluster_chns = data['cluster_chns']
            insertion = data['insertion']
            insertion['json'] = {'xyz_picks': self.xyz_picks}
            probe_insertion = one.alyx.rest('insertions', 'create', data=insertion)
            self.probe_id = probe_insertion['id']
            self.trajectory = data['trajectory']
            self.trajectory.update({'probe_insertion': self.probe_id})

        def test_alignments_disagree(self):
            alignments = {'2020-06-26T16:40:14_Karolina_Socha':
                              self.alignments['2020-06-26T16:40:14_Karolina_Socha'],
                          '2020-06-12T00:39:15_nate': self.alignments['2020-06-12T00:39:15_nate']}
            trajectory = copy.deepcopy(self.trajectory)
            trajectory.update({'json': alignments})
            traj = one.alyx.rest('trajectories', 'create', data=trajectory)
            self.traj_id = traj['id']

            align_qc = AlignmentQC(self.probe_id, one=one, brain_atlas=brain_atlas)
            insertion = one.alyx.rest('insertions', 'read', id=self.probe_id)
            # Make sure the qc fields have been added to the insertion object
            assert(insertion['json']['qc'] == 'NOT_SET')
            assert(len(insertion['json']['extended_qc']) == 0)

            align_qc.load_data(prev_alignments=traj['json'], xyz_picks=np.array(self.xyz_picks)/1e6,
                               cluster_chns=self.cluster_chns)
            align_qc.run(update=True, upload=True)

            insertion = one.alyx.rest('insertions', 'read', id=self.probe_id)
            assert (insertion['json']['qc'] == 'WARNING')
            assert (insertion['json']['extended_qc']['_alignment_number'] == 2)
            assert (insertion['json']['extended_qc']['_alignment_stored'] ==
                    '2020-06-26T16:40:14_Karolina_Socha')
            assert (insertion['json']['extended_qc']['_alignment_resolved'] == 0)
            assert (insertion['json']['extended_qc']['_alignment_qc'] < 0.8)

            traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                                 provenance='Ephys aligned histology track')
            assert(self.traj_id == traj[0]['id'])


        def test_alignments_agree(self):
            alignments = {'2020-06-19T10:52:36_noam.roth':
                              self.alignments['2020-06-19T10:52:36_noam.roth'],
                          '2020-06-12T00:39:15_nate': self.alignments['2020-06-12T00:39:15_nate']}
            trajectory = copy.deepcopy(self.trajectory)
            trajectory.update({'json': alignments})
            traj = one.alyx.rest('trajectories', 'update', id=self.traj_id, data=trajectory)
            assert(self.traj_id == traj['id'])

            align_qc = AlignmentQC(self.probe_id, one=one, brain_atlas=brain_atlas)
            align_qc.load_data(prev_alignments=traj['json'], xyz_picks=np.array(self.xyz_picks)/1e6,
                               cluster_chns=self.cluster_chns)
            align_qc.run(update=True, upload=True)

            insertion = one.alyx.rest('insertions', 'read', id=self.probe_id)
            assert (insertion['json']['qc'] == 'PASS')
            assert (insertion['json']['extended_qc']['_alignment_number'] == 2)
            assert (insertion['json']['extended_qc']['_alignment_stored'] ==
                    '2020-06-19T10:52:36_noam.roth')
            assert (insertion['json']['extended_qc']['_alignment_resolved'] == 1)
            assert (insertion['json']['extended_qc']['_alignment_qc'] > 0.8)

            traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                                 provenance='Ephys aligned histology track')
            assert(self.traj_id == traj[0]['id'])

        def test_not_latest_alignments_agree(self):
            alignments = copy.deepcopy(self.alignments)
            trajectory = copy.deepcopy(self.trajectory)
            trajectory.update({'json': alignments})
            traj = one.alyx.rest('trajectories', 'update', id=self.traj_id, data=trajectory)
            assert(self.traj_id == traj['id'])

            align_qc = AlignmentQC(self.probe_id, one=one, brain_atlas=brain_atlas)
            align_qc.load_data(prev_alignments=traj['json'], xyz_picks=np.array(self.xyz_picks)/1e6,
                               cluster_chns=self.cluster_chns)
            align_qc.run(update=True, upload=True)

            insertion = one.alyx.rest('insertions', 'read', id=self.probe_id)
            assert (insertion['json']['qc'] == 'PASS')
            assert (insertion['json']['extended_qc']['_alignment_number'] == 2)
            assert (insertion['json']['extended_qc']['_alignment_stored'] ==
                    '2020-06-19T10:52:36_noam.roth')
            assert (insertion['json']['extended_qc']['_alignment_resolved'] == 1)
            assert (insertion['json']['extended_qc']['_alignment_qc'] > 0.8)

            traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                                 provenance='Ephys aligned histology track')
            assert(self.traj_id != traj[0]['id'])
            self.traj_id = traj[0]['id']

        def tearDown(self) -> None:
            one.alyx.rest('insertions', 'delete', id=self.probe_id)
            one.alyx.rest('trajectories', 'delete', id=self.traj_id)




# class TestAlignmentQcManual(unittest.TestCase):












