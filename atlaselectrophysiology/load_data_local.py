import numpy as np
from datetime import datetime
import ibllib.atlas as atlas
from pathlib import Path
import alf.io
import glob
import json

brain_atlas = atlas.AllenAtlas(25)


class LoadDataLocal:
    def __init__(self):
        self.folder_path = []
        self.chn_coords = []
        self.sess_path = []

    def get_info(self, folder_path):
        """
        Read in the local json file to see if any previous alignments exist
        """
        self.folder_path = folder_path

        # If previous alignment json file exists, read in previous alignments
        if Path(self.folder_path, 'prev_alignments.json').exists():
            with open(Path(self.folder_path, 'prev_alignments.json'), "r") as f:
                self.alignments = json.load(f)
                self.prev_align = []
                if self.alignments:
                    self.prev_align = [*self.alignments.keys()]
                self.prev_align.reverse()
                self.prev_align.append('original')
        else:
            self.alignments = []
            self.prev_align = ['original']

        return self.prev_align

    def get_starting_alignment(self, idx):
        """
        Find out the starting alignmnet
        """
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

        # Define alf_path and ephys_path (a bit redundant but so it is compatible with plot data)
        alf_path = self.folder_path
        ephys_path = self.folder_path
        self.chn_coords = np.load(Path(alf_path, 'channels.localCoordinates.npy'))
        chn_depths = self.chn_coords[:, 1]

        # Read in notes for this experiment see if file exists in directory
        if Path(self.folder_path, 'session_notes.txt').exists():
            with open(Path(self.folder_path, 'session_notes.txt'), "r") as f:
                sess_notes = f.read()
        else:
            sess_notes = 'No notes for this session'

        return alf_path, ephys_path, chn_depths, sess_notes

    def get_allen_csv(self):
        allen_path = Path(Path(atlas.__file__).parent, 'allen_structure_tree.csv')
        self.allen = alf.io.load_file_content(allen_path)

        return self.allen

    def get_xyzpicks(self):
        # Read in local xyz_picks file
        # This file must exist, otherwise we don't know where probe was
        assert(Path(self.folder_path, 'xyz_picks.json').exists())
        with open(Path(self.folder_path, 'xyz_picks.json'), "r") as f:
            user_picks = json.load(f)

        xyz_picks = np.array(user_picks['xyz_picks']) / 1e6

        return xyz_picks

    def get_slice_images(self, xyz_channels):
        # First see if the histology file exists before attempting to connect with FlatIron and
        # download

        path_to_rd_image = glob.glob(str(self.folder_path) + '/*RD.nrrd')
        if path_to_rd_image:
            hist_path_rd = Path(path_to_rd_image[0])
        else:
            hist_path_rd = []

        path_to_gr_image = glob.glob(str(self.folder_path) + '/*GR.nrrd')
        if path_to_gr_image:
            hist_path_gr = Path(path_to_gr_image[0])
        else:
            hist_path_gr = []

        index = brain_atlas.bc.xyz2i(xyz_channels)[:, brain_atlas.xyz2dims]
        ccf_slice = brain_atlas.image[index[:, 0], index[:, 1], :]
        ccf_slice = np.swapaxes(ccf_slice, 0, 1)

        label_slice = brain_atlas._label2rgb(brain_atlas.label[index[:, 0], index[:, 1], :])
        label_slice = np.swapaxes(label_slice, 0, 1)

        width = [brain_atlas.bc.i2x(0), brain_atlas.bc.i2x(456)]
        height = [brain_atlas.bc.i2z(index[0, 0]), brain_atlas.bc.i2z(index[-1, 0])]

        if hist_path_rd:
            hist_atlas_rd = atlas.AllenAtlas(hist_path=hist_path_rd)
            hist_slice_rd = hist_atlas_rd.image[index[:, 0], index[:, 1], :]
            hist_slice_rd = np.swapaxes(hist_slice_rd, 0, 1)
        else:
            print('Could not find red histology image for this subject')
            hist_slice_rd = np.copy(ccf_slice)

        if hist_path_gr:
            hist_atlas_gr = atlas.AllenAtlas(hist_path=hist_path_gr)
            hist_slice_gr = hist_atlas_gr.image[index[:, 0], index[:, 1], :]
            hist_slice_gr = np.swapaxes(hist_slice_gr, 0, 1)
        else:
            print('Could not find histology image for this subject')
            hist_slice_gr = np.copy(ccf_slice)

        slice_data = {
            'hist_rd': hist_slice_rd,
            'hist_gr': hist_slice_gr,
            'ccf': ccf_slice,
            'label': label_slice,
            'scale': np.array([(width[-1] - width[0])/ccf_slice.shape[0],
                               (height[-1] - height[0])/ccf_slice.shape[1]]),
            'offset': np.array([width[0], height[0]])
        }

        return slice_data

    def get_region_description(self, region_idx):
        struct_idx = np.where(self.allen['id'] == region_idx)[0][0]
        # Haven't yet incorporated how to have region descriptions when not on Alyx
        # For now always have this as blank
        description = ''
        region_lookup = self.allen['acronym'][struct_idx] + ': ' + \
                        self.allen['name'][struct_idx]

        if region_lookup == 'void: void':
            region_lookup = 'root: root'

        if not description:
            description = region_lookup + '\nNo information available for this region'
        else:
            description = region_lookup + '\n' + description

        return description, region_lookup

    def upload_data(self, feature, track, xyz_channels, overwrite=False):

            brain_regions = brain_atlas.regions.get(brain_atlas.get_labels
                                                         (xyz_channels))
            brain_regions['xyz'] = xyz_channels
            brain_regions['lateral'] = self.chn_coords[:, 0]
            brain_regions['axial'] = self.chn_coords[:, 1]
            assert np.unique([len(brain_regions[k]) for k in brain_regions]).size == 1
            channel_dict = self.create_channel_dict(brain_regions)
            bregma = atlas.ALLEN_CCF_LANDMARKS_MLAPDV_UM['bregma'].tolist()
            origin = {'origin': {'bregma': bregma}}
            channel_dict.update(origin)

            # Save the channel locations
            with open(Path(self.folder_path, 'channel_locations.json'), "w") as f:
                json.dump(channel_dict, f,  indent=2, separators=(',', ': '))

            original_json = self.alignments
            date = datetime.now().replace(microsecond=0).isoformat()
            data = {date: [feature.tolist(), track.tolist()]}
            if original_json:
                original_json.update(data)
            else:
                original_json = data

            # Save the new alignment
            with open(Path(self.folder_path, 'prev_alignments.json'), "w") as f:
                json.dump(original_json, f,  indent=2, separators=(',', ': '))

    @staticmethod
    def create_channel_dict(brain_regions):
        """
        Create channel dictionary in form to write to json file
        :param brain_regions: information about location of electrode channels in brain atlas
        :type brain_regions: Bunch
        :return channel_dict:
        :type channel_dict: dictionary of dictionaries
        """
        channel_dict = {}
        for i in np.arange(brain_regions.id.size):
            channel = {
                'x': brain_regions.xyz[i, 0] * 1e6,
                'y': brain_regions.xyz[i, 1] * 1e6,
                'z': brain_regions.xyz[i, 2] * 1e6,
                'axial': brain_regions.axial[i],
                'lateral': brain_regions.lateral[i],
                'brain_region_id': int(brain_regions.id[i]),
                'brain_region': brain_regions.acronym[i]
            }
            data = {'channel_' + str(i): channel}
            channel_dict.update(data)

        return channel_dict
