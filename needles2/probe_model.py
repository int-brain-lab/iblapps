import numpy as np
from brainbox.numerical import ismember
from oneibl.one import ONE
from ibllib.pipes import histology
from ibllib.atlas import AllenAtlas
from ibllib.ephys.neuropixel import TIP_SIZE_UM, SITES_COORDINATES
from ibllib.pipes.ephys_alignment import EphysAlignment
import time

PROV_2_VAL = {
    'Resolved': 90,
    'Ephys aligned histology track': 70,
    'Histology track': 50,
    'Micro-manipulator': 30,
    'Planned': 10}

VAL_2_PROV = {v: k for k, v in PROV_2_VAL.items()}


class ProbeModel:
    def __init__(self, one=None, ba=None):

        self.one = one or ONE()
        self.ba = ba or AllenAtlas(25)
        self.traj = {'Planned': {},
                     'Micro-manipulator': {},
                     'Histology track': {},
                     'Ephys aligned histology track': {},
                     'Resolved': {},
                     'Best': {}}

        self.get_traj_for_provenance(provenance='Histology track', django=['x__isnull,False'])
        self.get_traj_for_provenance(provenance='Ephys aligned histology track')
        self.get_traj_for_provenance(provenance='Ephys aligned histology track',
                                     django=['probe_insertion__json__extended_qc__'
                                             'alignment_resolved,True'], prov_dict='Resolved')
        self.find_traj_is_best(provenance='Histology track')
        self.find_traj_is_best(provenance='Ephys aligned histology track')
        self.traj['Resolved']['is_best'] = np.arange(len(self.traj['Resolved']['traj']))

    @staticmethod
    def get_traj_info(traj):
        return traj['probe_insertion'], traj['x'], traj['y']

    def get_traj_for_provenance(self, provenance='Histology track', django=None,
                                prov_dict=None):

        if django is None:
            django = []

        if prov_dict is None:
            prov_dict = provenance

        django_base = ['probe_insertion__session__project__name__icontains,'
                       'ibl_neuropixel_brainwide_01']
        django_str = ','.join(django_base + django)

        self.traj[prov_dict]['traj'] = np.array(self.one.alyx.rest('trajectories', 'list',
                                                                   provenance=provenance,
                                                                   django=django_str))
        ins_ids, x, y = zip(*[self.get_traj_info(traj) for traj in self.traj[prov_dict]['traj']])
        self.traj[prov_dict]['ins'] = np.array(ins_ids)
        self.traj[prov_dict]['x'] = np.array(x)
        self.traj[prov_dict]['y'] = np.array(y)

    def compute_best_for_provenance(self, provenance='Histology track'):
        val = PROV_2_VAL[provenance]
        prov_to_include = []
        for k, v in VAL_2_PROV.items():
            if k >= val:
                prov_to_include.append(v)

        for iP, prov in enumerate(prov_to_include):
            if not 'is_best' in self.traj[prov].keys():
                self.find_traj_is_best(prov)

            if iP == 0:
                self.traj['Best']['traj'] = self.traj[prov]['traj'][self.traj[prov]['is_best']]
                self.traj['Best']['ins'] = self.traj[prov]['ins'][self.traj[prov]['is_best']]
                self.traj['Best']['x'] = self.traj[prov]['x'][self.traj[prov]['is_best']]
                self.traj['Best']['y'] = self.traj[prov]['y'][self.traj[prov]['is_best']]
            else:
                self.traj['Best']['traj'] = np.r_[self.traj['Best']['traj'],
                                                  (self.traj[prov]['traj']
                                                  [self.traj[prov]['is_best']])]
                self.traj['Best']['ins'] = np.r_[self.traj['Best']['ins'],
                                                 (self.traj[prov]['ins']
                                                 [self.traj[prov]['is_best']])]
                self.traj['Best']['x'] = np.r_[self.traj['Best']['x'],
                                               self.traj[prov]['x'][self.traj[prov]['is_best']]]
                self.traj['Best']['y'] = np.r_[self.traj['Best']['y'],
                                               self.traj[prov]['y'][self.traj[prov]['is_best']]]

    def find_traj_is_best(self, provenance='Histology track'):
        val = PROV_2_VAL[provenance]
        next_provenance = VAL_2_PROV[val + 20]

        if not 'traj' in self.traj[provenance].keys():
            self.get_traj_for_provenance(provenance)
        if not 'traj' in self.traj[next_provenance].keys():
            self.get_traj_for_provenance(next_provenance)

        isin, _ = ismember(self.traj[provenance]['ins'],
                           self.traj[next_provenance]['ins'])
        self.traj[provenance]['is_best'] = np.where(np.invert(isin))[0]

        # Special exception for planned provenance
        if provenance == 'Planned':
            next_provenance = VAL_2_PROV[val + 40]
            if not 'traj' in self.traj[next_provenance].keys():
                self.get_traj_for_provenance(next_provenance)
            isin, _ = ismember(self.traj[provenance]['ins'][self.traj[provenance]['is_best']],
                               self.traj[next_provenance]['ins'])
            self.traj[provenance]['is_best'] = (self.traj[provenance]['is_best']
                                                [np.where(np.invert(isin))[0]])

    def get_channels(self, provenance):
        
        depths = SITES_COORDINATES[:, 1]
        start = time.time()

        # Need to account for case when no insertions for planned and micro can skip the insertions
        insertions = []
        step = 150
        for i in range(np.ceil(self.traj[provenance]['ins'].shape[0] / step).astype(np.int)):
            insertions += self.one.alyx.rest(
                'insertions', 'list', django=f"id__in,{list(self.traj[provenance]['ins'][i * step:(i + 1) * step])}")

        ins_id = np.ravel([np.where(ins['id'] == self.traj[provenance]['ins'])
                           for ins in insertions])

        end = time.time()
        print(end-start)

        start1 = time.time()
        for iT, (traj, ins) in enumerate(zip(self.traj[provenance]['traj'][ins_id], insertions)):
            try:
                xyz_picks = np.array(ins['json']['xyz_picks']) / 1e6
                if traj['provenance'] == 'Histology track':
                    xyz_picks = xyz_picks[np.argsort(xyz_picks[:, 2]), :]
                    xyz_channels = histology.interpolate_along_track(xyz_picks, (depths + TIP_SIZE_UM) / 1e6)
                else:
                    align_key = ins['json']['extended_qc']['alignment_stored']
                    feature = traj['json'][align_key][0]
                    track = traj['json'][align_key][1]
                    ephysalign = EphysAlignment(xyz_picks, depths, track_prev=track,
                                                feature_prev=feature,
                                                brain_atlas=self.ba, speedy=True)
                    xyz_channels = ephysalign.get_channel_locations(feature, track)

                if iT == 0:
                    all_channels = xyz_channels
                else:
                    all_channels = np.r_[all_channels, xyz_channels]
            except Exception as err:
                print(err)
                print(traj['id'])

        end = time.time()
        print(end-start1)

        return all_channels

from scipy.signal import fftconvolve
cvol = np.zeros(ba.image.shape, dtype=np.float)
val, counts = np.unique(ba._lookup(all_channels), return_counts=True)
cvol[np.unravel_index(val, cvol.shape)] = counts
#
from ibllib.dsp import fcn_cosine
DIST_FCN = np.array([100, 150]) / 1e6
dx = ba.bc.dx
template = np.arange(- np.max(DIST_FCN) - dx, np.max(DIST_FCN) + 2 * dx, dx) ** 2
kernel = sum(np.meshgrid(template, template, template))
kernel = 1 - fcn_cosine(DIST_FCN)(np.sqrt(kernel))
#
cvol = fftconvolve(cvol, kernel)
#
#
# cvol[np.unravel_index(ba._lookup(all_channels), cvol.shape)] = 1

# from ibllib.atlas import AllenAtlas
# ba = AllenAtlas()
# import vedo
# import numpy as np
#
# actor = vedo.Volume(ba.image, c='bone', spacing = np.array([25]*3), mapper='smart', mode=0, alphaGradient=0.5)
# plt = vedo.Plotter()
# plt.add(actor)
# plt.show()


