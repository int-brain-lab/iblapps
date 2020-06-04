import scipy
import numpy as np
from datetime import datetime
import ibllib.pipes.histology as histology
import ibllib.atlas as atlas
from oneibl.one import ONE
from pathlib import Path
import alf.io
import glob
from atlaselectrophysiology.load_histology import download_histology_data, tif2nrrd
brain_atlas = atlas.AllenAtlas(25)
ONE_BASE_URL = "https://alyx.internationalbrainlab.org"
one = ONE(base_url=ONE_BASE_URL)


brain_regions = one.alyx.rest('brain-regions', 'list')
allen_id = np.empty((0, 1), dtype=int)
for br in brain_regions:
    allen_id = np.append(allen_id, br['id'])


class LoadData:
    def __init__(self):
        self.eid = []
        self.lab = []
        self.n_sess = []
        self.probe_label = []
        self.probe_id = []
        self.date = []
        self.subj = []
        self.chn_coords = []
        self.sess_path = []

    def get_subjects(self):
        """
        Finds all subjects that have a histology track trajectory registered
        :return subjects: list of subjects
        :type subjects: list of strings
        """
        sess_with_hist = one.alyx.rest('trajectories', 'list', provenance='Histology track')
        subjects = [sess['session']['subject'] for sess in sess_with_hist]
        self.subjects = np.unique(subjects)

        return self.subjects

    def get_sessions(self, idx):
        """
        Finds all sessions for a particular subject that have a histology track trajectory
        registered
        :param subject: subject name
        :type subject: string
        :return session: list of sessions associated with subject, displayed as date + probe
        :return session: list of strings
        """
        self.subj = self.subjects[idx]
        self.sess_with_hist = one.alyx.rest('trajectories', 'list', subject=self.subj,
                                            provenance='Histology track')
        session = [(sess['session']['start_time'][:10] + ' ' + sess['probe_name']) for sess in
                   self.sess_with_hist]
        return session

    def get_info(self, idx):
        """
        """
        self.n_sess = self.sess_with_hist[idx]['session']['number']
        self.date = self.sess_with_hist[idx]['session']['start_time'][:10]
        self.probe_label = self.sess_with_hist[idx]['probe_name']
        self.probe_id = self.sess_with_hist[idx]['probe_insertion']
        self.lab = self.sess_with_hist[idx]['session']['lab']

        ephys_traj_prev = one.alyx.rest('trajectories', 'list', probe_insertion=self.probe_id,
                                        provenance='Ephys aligned histology track')

        if ephys_traj_prev:
            self.alignments = ephys_traj_prev[0]['json']
            self.prev_align = []
            if self.alignments:
                self.prev_align = [*self.alignments.keys()]
            # To make sure they are ordered by date added, default to latest fit
            self.prev_align.reverse()
            self.prev_align.append('original')
        else:
            self.prev_align = ['original']

        return self.prev_align

    def get_eid(self):
        eids = one.search(subject=self.subj, date=self.date, number=self.n_sess,
                          task_protocol='ephys')
        self.eid = eids[0]
        print(self.subj)
        print(self.probe_label)
        print(self.date)
        print(self.eid)

    def get_starting_alignment(self, idx):
        align = self.prev_align[idx]

        if align == 'original':
            feature = None
            track = None
        else:
            feature = np.array(self.alignments[align][0])
            track = np.array(self.alignments[align][1])

        return feature, track

    def get_data(self):
        # Load in all the data required
        dtypes = [
            'spikes.depths',
            'spikes.amps',
            'spikes.times',
            'spikes.clusters',
            'channels.localCoordinates',
            'channels.rawInd',
            'clusters.metrics',
            'clusters.peakToTrough',
            'clusters.waveforms',
            '_iblqc_ephysTimeRms.rms',
            '_iblqc_ephysTimeRms.timestamps',
            '_iblqc_ephysSpectralDensity.freqs',
            '_iblqc_ephysSpectralDensity.amps',
            '_iblqc_ephysSpectralDensity.power'
        ]

        _ = one.load(self.eid, dataset_types=dtypes, download_only=True)
        self.sess_path = one.path_from_eid(self.eid)
        alf_path = Path(self.sess_path, 'alf', self.probe_label)
        ephys_path = Path(self.sess_path, 'raw_ephys_data', self.probe_label)
        self.chn_coords = np.load(Path(alf_path, 'channels.localCoordinates.npy'))
        chn_depths = self.chn_coords[:, 1]

        sess = one.alyx.rest('sessions', 'read', id=self.eid)
        if sess['notes']:
            sess_notes = sess['notes'][0]['text']
        else:
            sess_notes = 'No notes for this session'

        return alf_path, ephys_path, chn_depths, sess_notes

    def get_allen_csv(self):
        allen_path = Path(Path(atlas.__file__).parent, 'allen_structure_tree.csv')
        allen = alf.io.load_file_content(allen_path)

        return allen

    def get_xyzpicks(self):
        insertion = one.alyx.rest('insertions', 'list', session=self.eid, name=self.probe_label)
        xyz_picks = np.array(insertion[0]['json']['xyz_picks']) / 1e6

        return xyz_picks

    def get_slice_images(self, xyz_channels):
        # First see if the histology file exists before attempting to connect with FlatIron and
        # download
        hist_dir = Path(self.sess_path.parent.parent, 'histology')
        if hist_dir.exists():
            path_to_image = glob.glob(str(hist_dir) + '/*RD.tif')
            if path_to_image:
                hist_path = tif2nrrd(Path(path_to_image[0]))
            else:
                hist_path = download_histology_data(self.subj, self.lab)
        else:
            hist_path = download_histology_data(self.subj, self.lab)

        ccf_slice, width, height, _ = brain_atlas.tilted_slice(xyz_channels, axis=1)
        ccf_slice = np.swapaxes(np.flipud(ccf_slice), 0, 1)
        label_slice, _, _, _ = brain_atlas.tilted_slice(xyz_channels, volume='annotation', axis=1)
        label_slice = np.swapaxes(np.flipud(label_slice), 0, 1)

        if hist_path:
            hist_atlas = atlas.AllenAtlas(hist_path=hist_path)
            hist_slice, _, _, _ = hist_atlas.tilted_slice(xyz_channels, axis=1)
            hist_slice = np.swapaxes(np.flipud(hist_slice), 0, 1)
        else:
            print('Could not find histology image for this subject')
            hist_slice = np.copy(ccf_slice)

        slice_data = {
            'hist': hist_slice,
            'ccf': ccf_slice,
            'label': label_slice,
            'scale': np.array([(width[-1] - width[0])/hist_slice.shape[0],
                               (height[-1] - height[0])/hist_slice.shape[1]]),
            'offset': np.array([width[0], height[0]])
        }

        return slice_data

    @staticmethod
    def get_region_description(region_idx):
        struct_idx = np.where(allen_id == region_idx)[0][0]
        description = brain_regions[struct_idx]['description']
        region_lookup = brain_regions[struct_idx]['acronym'] + ': ' + \
                        brain_regions[struct_idx]['name']

        if region_lookup == 'void: void':
            region_lookup = 'root: root'

        if not description:
            description = region_lookup + '\nNo information available on Alyx for this region'
        else:
            description = region_lookup + '\n' + description

        return description, region_lookup

    def upload_data(self, feature, track, xyz_channels, overwrite=False):

        if overwrite:
            # Get the original stored trajectory
            ephys_traj_prev = one.alyx.rest('trajectories', 'list', probe_insertion=self.probe_id,
                                            provenance='Ephys aligned histology track')
            # Save the json field in memory
            original_json = []
            if np.any(ephys_traj_prev):
                original_json = ephys_traj_prev[0]['json']

            # Create new trajectory and overwrite previous one
            insertion = atlas.Insertion.from_track(xyz_channels, brain_atlas)
            # NEEED TO ADD TIP TO DEPTH?
            brain_regions = brain_atlas.regions.get(brain_atlas.get_labels
                                                         (xyz_channels))
            brain_regions['xyz'] = xyz_channels
            brain_regions['lateral'] = self.chn_coords[:, 0]
            brain_regions['axial'] = self.chn_coords[:, 1]
            assert np.unique([len(brain_regions[k]) for k in brain_regions]).size == 1
            histology.register_aligned_track(self.probe_id, insertion, brain_regions, one=one,
                                             overwrite=overwrite)

            # Get the new trajectoru
            ephys_traj = one.alyx.rest('trajectories', 'list', probe_insertion=self.probe_id,
                                       provenance='Ephys aligned histology track')

            name = one._par.ALYX_LOGIN
            date = datetime.now().replace(microsecond=0).isoformat()
            data = {date + '_' + name: [feature.tolist(), track.tolist()]}
            if original_json:
                original_json.update(data)
            else:
                original_json = data
            patch_dict = {'json': original_json}
            one.alyx.rest('trajectories', 'partial_update', id=ephys_traj[0]['id'],
                          data=patch_dict)


