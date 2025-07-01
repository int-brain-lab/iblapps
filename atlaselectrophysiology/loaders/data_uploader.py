import json
from ibllib.pipes import histology
from one import params
from datetime import datetime
from abc import ABC, abstractmethod
import numpy as np
from iblatlas import atlas
from ibllib.qc.alignment_qc import AlignmentQC
import ibllib.qc.critical_reasons as usrpmt

class DataUploader(ABC):
    def __init__(self, brain_atlas):
        self.brain_atlas = brain_atlas

    @abstractmethod
    def upload_data(self, data): pass




class DataUploaderLocal(DataUploader):
    def __init__(self, data_path, shank_idx, n_shanks, brain_atlas, user=None):
        self.data_path = data_path
        self.shank_idx = shank_idx
        self.n_shanks = n_shanks
        self.user = user
        # TODO figure this out
        self.orig_idx = None
        super().__init__(brain_atlas)

    def upload_data(self, data):
        self.data = data
        brain_regions = self.brain_atlas.regions.get(self.brain_atlas.get_labels
                                                     (self.data['xyz_channels']))
        brain_regions['xyz'] = self.data['xyz_channels']
        brain_regions['lateral'] = self.data['chn_coords'][:, 0]
        brain_regions['axial'] = self.data['chn_coords'][:, 1]
        assert np.unique([len(brain_regions[k]) for k in brain_regions]).size == 1
        channel_dict = self.create_channel_dict(brain_regions)
        bregma = atlas.ALLEN_CCF_LANDMARKS_MLAPDV_UM['bregma'].tolist()
        origin = {'origin': {'bregma': bregma}}
        channel_dict.update(origin)
        # Save the channel locations
        chan_loc_filename = 'channel_locations.json' if self.n_shanks == 1 else \
            f'channel_locations_shank{self.shank_idx + 1}.json'

        with open(self.data_path.joinpath(chan_loc_filename), "w") as f:
            json.dump(channel_dict, f, indent=2, separators=(',', ': '))

        self.update_alignments()

        info = 'Channels locations saved'
        return info

    def update_alignments(self):
        original_json = self.data['alignments']
        date = datetime.now().replace(second=0, microsecond=0).isoformat()
        align_key = date + '_' + self.user if self.user else date
        data = {align_key: [self.data['feature'], self.data['track']]}
        if original_json:
            original_json.update(data)
        else:
            original_json = data
        # Save the new alignment
        self.write_alignments_to_disk(original_json)

    def write_alignments_to_disk(self, data):
        prev_align_filename = 'prev_alignments.json' if self.n_shanks == 1 else \
            f'prev_alignments_shank{self.shank_idx + 1}.json'
        with open(self.data_path.joinpath(prev_align_filename), "w") as f:
            json.dump(data, f, indent=2, separators=(',', ': '))

    def create_channel_dict(self, brain_regions):
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
                'x': np.float64(brain_regions.xyz[i, 0] * 1e6),
                'y': np.float64(brain_regions.xyz[i, 1] * 1e6),
                'z': np.float64(brain_regions.xyz[i, 2] * 1e6),
                'axial': np.float64(brain_regions.axial[i]),
                'lateral': np.float64(brain_regions.lateral[i]),
                'brain_region_id': int(brain_regions.id[i]),
                'brain_region': brain_regions.acronym[i]
            }
            if self.orig_idx is not None:
                channel['original_channel_idx'] = int(self.orig_idx[i])

            data = {'channel_' + str(i): channel}
            channel_dict.update(data)

        return channel_dict


class DataUploaderONE(DataUploader):
    def __init__(self, insertion, one, brain_atlas):
        self.one = one
        self.insertion = insertion
        self.probe_id = self.insertion['id']
        self.resolved = self.insertion['json'].get('extended_qc', {}).get('alignment_resolved', False)
        super().__init__(brain_atlas)

    def upload_data(self, data):

        self.data = data
        # Upload channels
        channels = self.upload_channels()
        # Upload alignments
        self.update_alignments()
        # Update alignment qc
        resolved = self.update_qc()
        # Update probe qc
        upload_info = self.get_upload_info(channels, resolved)

        return upload_info

    @staticmethod
    def get_upload_info(channels, resolved):
        if channels and resolved == 0:
            # Channels saved alignment not resolved
            info = "Channels locations saved to Alyx. \nAlignment not resolved"

        if channels and resolved == 1:
            # channels saved alignment resolved, writen to flatiron
            info = "Channel locations saved to Alyx. \nAlignment resolved and channels datasets written to flatiron"

        if not channels and resolved == 1:
            # alignment already resolved, save alignment but channels not written
            info = ("Channel locations not saved to Alyx as alignment has already been resolved. "
                    "\nNew user reference lines have been saved")

        return info

    def upload_channels(self, channels=True):
        if not self.resolved:

            channel_upload = True
            # Create new trajectory and overwrite previous one
            histology.register_aligned_track(self.probe_id, self.data['xyz_channels'],
                                             chn_coords=self.data['chn_coords'], one=self.one,
                                             overwrite=True, channels=channels, brain_atlas=self.brain_atlas)
        else:
            channel_upload = False

        return channel_upload

    def update_alignments(self, key_info=None, user_eval=None):

        feature = self.data['feature']
        track = self.data['track']

        if not key_info:
            user = params.get().ALYX_LOGIN
            date = datetime.now().replace(second=0, microsecond=0).isoformat()
            data = {date + '_' + user: [feature, track, self.qc_str, self.confidence_str]}
        else:
            user = key_info[20:]
            if user_eval:
                data = {key_info: [feature, track, user_eval]}
            else:
                data = {key_info: [feature, track]}

        old_user = [key for key in self.data['alignments'].keys() if user in key]
        # Only delete duplicated if trajectory is not resolved
        if len(old_user) > 0 and not self.resolved:
            for old in old_user:
                self.data['alignments'].pop(old)

        self.data['alignments'].update(data)
        self.write_alignments_to_disk(self.data['alignments'])
        self.update_json(self.data['alignments'])

    def write_alignments_to_disk(self, data):
        prev_align_filename = 'prev_alignments.json'
        if self.data['probe_path'] is not None:
            with open(self.data['probe_path'].joinpath(prev_align_filename), "w") as f:
                json.dump(data, f, indent=2, separators=(',', ': '))

    def update_json(self, json_data):
        # Get the new trajectory
        ephys_traj = self.one.alyx.rest('trajectories', 'list', probe_insertion=self.probe_id,
                                        provenance='Ephys aligned histology track', no_cache=True)
        patch_dict = {'probe_insertion': self.probe_id, 'json': json_data}
        self.one.alyx.rest('trajectories', 'partial_update', id=ephys_traj[0]['id'],
                           data=patch_dict)

    def get_qc_string(self, align_qc, ephys_qc, ephys_desc):

        if len(ephys_desc) == 0:
            ephys_desc_str = 'None'
        else:
            ephys_desc_str = ", ".join(ephys_desc)

        self.qc_str = ephys_qc.upper() + ': ' + ephys_desc_str
        self.confidence_str = f'Confidence: {align_qc}'


        if ephys_qc.upper() == 'CRITICAL':
            usrpmt.main_gui(self.probe_id, reasons_selected=ephys_desc, alyx=self.one.alyx)

    def update_qc(self, upload_alyx=True, upload_flatiron=False):
        # if resolved just update the alignment_number
        align_qc = AlignmentQC(self.probe_id, one=self.one, brain_atlas=self.brain_atlas,
                               collection=self.data['probe_collection'])
        align_qc.load_data(prev_alignments=self.data['alignments'], xyz_picks=self.data['xyz_picks'],
                           depths=self.data['chn_depths'], cluster_chns=self.data['cluster_chns'],
                           chn_coords=self.data['chn_coords'])
        results = align_qc.run(update=True, upload_alyx=upload_alyx,
                               upload_flatiron=upload_flatiron)
        align_qc.update_experimenter_evaluation(prev_alignments=self.data['alignments'])

        self.resolved = results['alignment_resolved']

        return self.resolved



