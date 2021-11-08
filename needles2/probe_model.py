import time
import copy
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from one.api import ONE
from iblutil.numerical import ismember

from ibllib.pipes import histology
from ibllib.atlas import AllenAtlas, atlas
from ibllib.ephys.neuropixel import TIP_SIZE_UM, SITES_COORDINATES
from ibllib.pipes.ephys_alignment import EphysAlignment
from ibllib.dsp import fcn_cosine

PROV_2_VAL = {
    'Resolved': 90,
    'Ephys aligned histology track': 70,
    'Histology track': 50,
    'Micro-manipulator': 30,
    'Planned': 10}

VAL_2_PROV = {v: k for k, v in PROV_2_VAL.items()}


class ProbeModel:
    def __init__(self, one=None, ba=None, lazy=False):

        self.one = one or ONE()
        self.ba = ba or AllenAtlas(25)
        self.traj = {'Planned': {},
                     'Micro-manipulator': {},
                     'Histology track': {},
                     'Ephys aligned histology track': {},
                     'Resolved': {},
                     'Best': {}}
        self.ins = {}
        self.cvol = None
        self.cvol_flat = None
        self.initialised = False
        self.mirror = False

        if not lazy:
            self.initialise()

    def initialise(self):
        self.get_traj_for_provenance(provenance='Histology track', django=['x__isnull,False'])
        self.get_traj_for_provenance(provenance='Ephys aligned histology track')
        self.get_traj_for_provenance(provenance='Ephys aligned histology track',
                                     django=['probe_insertion__json__extended_qc__'
                                             'alignment_resolved,True'], prov_dict='Resolved')
        self.find_traj_is_best(provenance='Histology track')
        self.find_traj_is_best(provenance='Ephys aligned histology track')
        self.traj['Resolved']['is_best'] = np.arange(len(self.traj['Resolved']['traj']))

        self.get_insertions_with_xyz()
        self.initialised = True

    @staticmethod
    def get_traj_info(traj):
        return traj['probe_insertion'], traj['x'], traj['y']

    def get_traj_for_provenance(self, provenance='Histology track', django=None,
                                prov_dict=None):
        start = time.time()
        if django is None:
            django = []

        if prov_dict is None:
            prov_dict = provenance

        django_base = ['probe_insertion__session__project__name__icontains,'
                       'ibl_neuropixel_brainwide_01']
        django = ['probe_insertion__session__qc__lt,50',
                  '~probe_insertion__json__qc,CRITICAL',
                  'probe_insertion__json__extended_qc__tracing_exists,True',
                  'probe_insertion__session__extended_qc__behavior,1']


        django_str = ','.join(django_base + django)

        self.traj[prov_dict]['traj'] = np.array(self.one.alyx.rest('trajectories', 'list',
                                                                   provenance=provenance,
                                                                   django=django_str))

        if self.mirror:
            # if we want to mirror all insertions onto one hemisphere
            for ip, p in enumerate(self.traj[prov_dict]['traj']):
               if p['x'] < 0:
                   continue
               elif p['y'] < -4400:
                   self.traj[prov_dict]['traj'][ip]['x'] = -1 * p['x']
                   if p['phi'] == 180:
                       self.traj[prov_dict]['traj'][ip]['phi'] = 0
                   else:
                       self.traj[prov_dict]['traj'][ip]['phi'] = 180


        ins_ids, x, y = zip(*[self.get_traj_info(traj) for traj in self.traj[prov_dict]['traj']])
        self.traj[prov_dict]['ins'] = np.array(ins_ids)
        self.traj[prov_dict]['x'] = np.array(x)
        self.traj[prov_dict]['y'] = np.array(y)
        end = time.time()
        print(end-start)

    def get_insertions_with_xyz(self):
        start = time.time()
        django_str = 'session__project__name__icontains,ibl_neuropixel_brainwide_01,' \
                     'json__has_key,xyz_picks'
        self.ins['insertions'] = self.one.alyx.rest('insertions', 'list', django=django_str)
        self.ins['ids'] = np.array([ins['id'] for ins in self.ins['insertions']])

        end = time.time()
        print(end-start)

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

    def get_all_channels(self, provenance):

        depths = SITES_COORDINATES[:, 1]
        for iT, traj in enumerate(self.traj[provenance]['traj']):
            try:
                xyz_channels = self.get_channels(traj, depths=depths)
                if iT == 0:
                    all_channels = xyz_channels
                else:
                    all_channels = np.r_[all_channels, xyz_channels]
            except Exception as err:
                print(err)
                print(traj['id'])

        iii = self.ba.bc.xyz2i(all_channels)
        keep_idx = np.setdiff1d(np.arange(all_channels.shape[0]), np.unique(np.where(iii < 0)[0]))
        return all_channels[keep_idx, :]

    def compute_channel_coverage(self, all_channels):

        start = time.time()
        cvol = np.zeros(self.ba.image.shape, dtype=np.float)
        val, counts = np.unique(self.ba._lookup(all_channels), return_counts=True)
        #cvol[np.unravel_index(val, cvol.shape)] = counts
        cvol[np.unravel_index(val, cvol.shape)] = 1

        DIST_FCN = np.array([100, 150]) / 1e6
        dx = self.ba.bc.dx
        template = np.arange(- np.max(DIST_FCN) - dx, np.max(DIST_FCN) + 2 * dx, dx) ** 2
        kernel = sum(np.meshgrid(template, template, template))
        kernel = 1 - fcn_cosine(DIST_FCN)(np.sqrt(kernel))
        #
        cvol = fftconvolve(cvol, kernel, mode='same')
        end = time.time()
        print(end-start)
        self.cvol = cvol
        self.cvol_flat = cvol.flatten()

        return cvol

    def grid_coverage(self, all_channels, spacing):
        cov, bc = histology.coverage_grid(all_channels, spacing, self.ba)

        return cov, bc

    def add_coverage(self, traj):

        cov, xyz, flatixyz = histology.coverage([traj], self.ba)
        if self.cvol_flat is not None:
            idx = np.where(cov.flatten()[flatixyz] > 0.1)[0]
            idx_sig = np.where(self.cvol_flat[flatixyz][idx] > 0.1)[0].shape[0]
            per_new_coverage = (1 - idx_sig / idx.shape[0]) * 100
        else:
            per_new_coverage = np.nan

        return cov, xyz, per_new_coverage

    def insertion_by_id(self, ins_id):
        traj = self.one.alyx.rest('trajectories', 'list', probe_insertion=ins_id)
        ins = self.one.alyx.rest('insertions', 'list', id=ins_id)[0]
        val = [PROV_2_VAL[tr['provenance']] for tr in traj]
        best_traj = traj[np.argmax(val)]

        return best_traj, ins

    def get_channels(self, traj, ins=None, depths=None):
        if depths is None:
            depths = SITES_COORDINATES[:, 1]
        if traj['provenance'] == 'Planned' or traj['provenance'] == 'Micro-manipulator':
            ins = atlas.Insertion.from_dict(traj)
            # Deepest coordinate first
            xyz = np.c_[ins.tip, ins.entry].T
            xyz_channels = histology.interpolate_along_track(xyz, (depths +
                                                                   TIP_SIZE_UM) / 1e6)
        else:
            if ins is None:
                ins_idx = np.where(traj['probe_insertion'] == self.ins['ids'])[0][0]
                xyz = np.array(self.ins['insertions'][ins_idx]['json']['xyz_picks']) / 1e6
            else:
                xyz = np.array(ins['json']['xyz_picks']) / 1e6
            if traj['provenance'] == 'Histology track':
                xyz = xyz[np.argsort(xyz[:, 2]), :]
                xyz_channels = histology.interpolate_along_track(xyz, (depths +
                                                                       TIP_SIZE_UM) / 1e6)
            else:
                if ins is None:
                    align_key = (self.ins['insertions'][ins_idx]['json']['extended_qc']
                                 ['alignment_stored'])
                else:
                    align_key = ins['json']['extended_qc']['alignment_stored']
                feature = traj['json'][align_key][0]
                track = traj['json'][align_key][1]
                ephysalign = EphysAlignment(xyz, depths, track_prev=track,
                                            feature_prev=feature,
                                            brain_atlas=self.ba, speedy=True)
                xyz_channels = ephysalign.get_channel_locations(feature, track)

        return xyz_channels

    def get_brain_regions(self, traj, ins=None, mapping='Allen'):
        depths = SITES_COORDINATES[:, 1]
        xyz_channels = self.get_channels(traj, ins=ins, depths=depths)
        region_ids = self.ba.get_labels(xyz_channels, mapping=mapping)
        if all(region_ids == 0):
            region = [[0, 3840]]
            region_label = [[3840/2, 'VOID']]
            region_colour = [[0, 0, 0]]
        else:
            (region, region_label,
             region_colour, _) = EphysAlignment.get_histology_regions(xyz_channels, depths,
                                                                      brain_atlas=self.ba,
                                                                      mapping=mapping)
        return region, region_label, region_colour


    def report_coverage(self, provenance, dist):
        coverage, _ = self.compute_coverage(self.traj[provenance]['traj'],
                                            dist_fcn=[dist, dist + 1])
        coverage[coverage > 2] = 2
        return coverage


    def compute_coverage(self, trajs, dist_fcn=[50, 100], limit=True, coverage=None, pl_voxels=None, factor=1):
        """
        Computes a coverage volume from
        :param trajs: dictionary of trajectories from Alyx rest endpoint (one.alyx.rest...)
        :param ba: ibllib.atlas.BrainAtlas instance
        :return: 3D np.array the same size as the volume provided in the brain atlas
        """
        # in um. Coverage = 1 below the first value, 0 after the second, cosine taper in between
        ACTIVE_LENGTH_UM = 3.84 * 1e3
        MAX_DIST_UM = dist_fcn[1]  # max distance around the probe to be searched for
        ba = self.ba

        def crawl_up_from_tip(ins, d):
            return (ins.entry - ins.tip) * (d[:, np.newaxis] /
                                            np.linalg.norm(ins.entry - ins.tip)) + ins.tip
        if coverage is None:
            full_coverage = np.zeros(ba.image.shape, dtype=np.float32)
        else:
            full_coverage = copy.deepcopy(coverage)

        full_coverage[ba.label == 0] = np.nan
        full_coverage = full_coverage.flatten()

        per2 = []
        per1 = []
        per0 = []

        for p in np.arange(len(trajs)):
            if len(trajs) > 20:
                if p % 20 == 0:
                    print(p / len(trajs))
            traj = trajs[p]
            ins = atlas.Insertion.from_dict(traj)
            # those are the top and bottom coordinates of the active part of the shank extended
            # to maxdist
            d = (np.array([ACTIVE_LENGTH_UM + MAX_DIST_UM * np.sqrt(2),
                           -MAX_DIST_UM * np.sqrt(2)]) + TIP_SIZE_UM)
            top_bottom = crawl_up_from_tip(ins, d / 1e6)
            # this is the axis that has the biggest deviation. Almost always z
            axis = np.argmax(np.abs(np.diff(top_bottom, axis=0)))
            if axis != 2:
                continue
            # sample the active track path along this axis
            tbi = ba.bc.xyz2i(top_bottom)
            nz = tbi[1, axis] - tbi[0, axis] + 1
            ishank = np.round(np.array(
                [np.linspace(tbi[0, i], tbi[1, i], nz) for i in np.arange(3)]).T).astype(
                np.int32)
            # creates a flattened "column" of candidate volume indices around the track
            # around each sample get an horizontal square slice of nx *2 +1 and ny *2 +1 samples
            # flatten the corresponding xyz indices and  compute the min distance to the track only
            # for those
            nx = int(
                np.floor(MAX_DIST_UM / 1e6 / np.abs(ba.bc.dxyz[0]) * np.sqrt(2) / 2)) * 2 + 1
            ny = int(
                np.floor(MAX_DIST_UM / 1e6 / np.abs(ba.bc.dxyz[1]) * np.sqrt(2) / 2)) * 2 + 1
            ixyz = np.stack([v.flatten() for v in np.meshgrid(
                np.arange(-nx, nx + 1), np.arange(-ny, ny + 1), np.arange(nz))]).T
            ixyz[:, 0] = ishank[ixyz[:, 2], 0] + ixyz[:, 0]
            ixyz[:, 1] = ishank[ixyz[:, 2], 1] + ixyz[:, 1]
            ixyz[:, 2] = ishank[ixyz[:, 2], 2]
            # if any, remove indices that lie outside of the volume bounds
            iok = np.logical_and(0 <= ixyz[:, 0], ixyz[:, 0] < ba.bc.nx)
            iok &= np.logical_and(0 <= ixyz[:, 1], ixyz[:, 1] < ba.bc.ny)
            iok &= np.logical_and(0 <= ixyz[:, 2], ixyz[:, 2] < ba.bc.nz)
            ixyz = ixyz[iok, :]
            # get the minimum distance to the trajectory, to which is applied the cosine taper
            xyz = np.c_[
                ba.bc.xscale[ixyz[:, 0]], ba.bc.yscale[ixyz[:, 1]], ba.bc.zscale[ixyz[:, 2]]]
            sites_bounds = crawl_up_from_tip(
                ins, (np.array([ACTIVE_LENGTH_UM, 0]) + TIP_SIZE_UM) / 1e6)
            mdist = ins.trajectory.mindist(xyz, bounds=sites_bounds)
            coverage = 1 - fcn_cosine(np.array(dist_fcn) / 1e6)(mdist)
            if limit:
                coverage[coverage != 1] = 0
            else:
                coverage[coverage > 0] = 1
            # remap to the coverage volume
            flat_ind = ba._lookup_inds(ixyz)
            full_coverage[flat_ind] += (coverage * factor)

            if pl_voxels is not None:
                n_pl_voxels = pl_voxels.shape[0]
                fp_voxels_2 = np.where(full_coverage[pl_voxels] >= 2)[0].shape[0]
                fp_voxels_1 = np.where(full_coverage[pl_voxels] == 1)[0].shape[0]
                fp_voxels_0 = np.where(full_coverage[pl_voxels] == 0)[0].shape[0]

                per2.append((fp_voxels_2 / n_pl_voxels) * 100)
                per1.append((fp_voxels_1 / n_pl_voxels) * 100)
                per0.append((fp_voxels_0 / n_pl_voxels) * 100)

        full_coverage = full_coverage.reshape(ba.image.shape)
        # full_coverage[ba.label == 0] = np.nan

        if pl_voxels is not None:
            return full_coverage, per0, per1, per2
        else:
            return full_coverage, np.mean(xyz, 0)


def coverage_with_insertions(csv_file):
    pr = ProbeModel()
    pr.initialise()
    pr.compute_best_for_provenance(provenance='Histology track')
    first_pass_coverage = pr.report_coverage(provenance='Best', dist=354)

    second_pass_coverage = np.load(r'C:\Users\Mayo\iblenv\volume.npy')
    second_pass_coverage = second_pass_coverage.flatten()
    second_pass_coverage[second_pass_coverage == 0] = np.nan
    ixyz_second = np.where(~np.isnan(second_pass_coverage.flatten()))[0]

    insertions = pd.read_csv(csv_file).to_dict(orient='records')

    # want to make it twice for each insertion

    new_coverage, per0, per1, per2 = pr.compute_coverage(insertions, dist_fcn=[354, 355], coverage=first_pass_coverage,
                                                         pl_voxels=ixyz_second, factor=2)

    return new_coverage, per0, per1, per2









