import time
import copy
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from one.api import ONE
from iblutil.numerical import ismember

from ibllib.pipes import histology
from ibllib.atlas import AllenAtlas, atlas
from neuropixel import TIP_SIZE_UM, trace_header
from ibllib.pipes.ephys_alignment import EphysAlignment
from neurodsp.utils import fcn_cosine

PROV_2_VAL = {
    'Resolved': 90,
    'Ephys aligned histology track': 70,
    'Histology track': 50,
    'Micro-manipulator': 30,
    'Planned': 10}
SITES_COORDINATES = np.c_[trace_header()['x'], trace_header()['y']]
VAL_2_PROV = {v: k for k, v in PROV_2_VAL.items()}


class ProbeModel:
    def __init__(self, one=None, ba=None, lazy=False, res=25, verbose=False):

        self.one = one or ONE()
        self.ba = ba or AllenAtlas(res_um=res)
        self.ba.compute_surface()
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
        self.verbose = verbose

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
                       'ibl_neuropixel_brainwide_01,probe_insertion__session__json__IS_MOCK,False',
                       'probe_insertion__session__qc__lt,50',
                       '~probe_insertion__json__qc,CRITICAL',
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
        if self.verbose is True:
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
            ins = atlas.Insertion.from_dict(traj, brain_atlas=self.ba)
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

    def compute_coverage(self, trajs, dist_fcn=[50, 100], limit=True, coverage=None, pl_voxels=None, factor=1,
                         entire_voxel=True):
        """
        Computes coverage of the brain atlas associated with the probe model given a set of trajectories.

        Parameters
        ----------
        trajs: list of dictionary of trajectories from Alyx rest endpoint
        dist_fcn: list of two values in micro meter, the first one marks the distance to the probe, below which
                  voxels will be marked covered (=1). Coverage will then decrease with a cosine taper until the
                  second value, after which coverage will be 0.
        limit: boolean, if True, the effect of the cosine taper will be negated, i.e. all coverage values below 1
               will be set to 0.
        coverage: np.array of the same size as the brain atlas image associated with the probe insertion.
                  If given, the coverage computed by this function will be added to this input.
        pl_voxels: np.array with flat indices into the brain atlas image volume. If given, only these voxels are
                   considered for the computation of coverage percentages. Note that the coverage itself is always
                   computed for the entire volume
        factor: Number by which to multiply coverage of each trajectory, e.g. to count trajectories twice
        entire_voxel: boolean, if True, consider every voxel as covered, that is touched by the radius if dist_fcn[0]
                      around the probe. If False, only consider those as covered where the center of the voxel falls in
                      this radius.

        Returns
        3D np.array the same size as the brain atlas with coverage values for every voxel in the brain

        If pl_voxels is given:
        Three lists of values indicating the percent coverage of these voxels after adding each trajectory (increases
        incrementally). The first list indicates percent covered by 0 probes, second list percent covered by 1 probe,
        and the third list percent covered by 2 probes.
        """

        ba = self.ba
        ACTIVE_LENGTH_UM = 3.84 * 1e3  # This is the length of the NP1 probe with electrodes
        MAX_DIST_UM = dist_fcn[1]  # max distance around the probe to be searched for

        # Covered_length_um are two values which indicate on the path from tip to entry of a given insertion
        # where the region that is considered to be covered by this insertion begins and ends in micro meter
        # Note that the second value is negative, because the covered regions extends beyond the tip of the probe
        # We multiply by sqrt(2) to translate the radius given by dist_fcn[1] into the side length of a square
        # that is contained in the circle with radius dist_fcn[1]
        covered_length_um = TIP_SIZE_UM + np.array([ACTIVE_LENGTH_UM + MAX_DIST_UM * np.sqrt(2),
                                                    -MAX_DIST_UM * np.sqrt(2)])

        # Horizontal slice of voxels to be considered around each trajectory is only dependent on MAX_DIST_UM
        # and the voxel resolution, so can be defined here. We translate max dist in voxels and add 1 for safety
        # Unit of atlas is m so divide um by 1e6
        NX = int(np.ceil(MAX_DIST_UM / 1e6 / np.abs(ba.bc.dxyz[0]))) + 1
        NY = int(np.ceil(MAX_DIST_UM / 1e6 / np.abs(ba.bc.dxyz[1]))) + 1

        def crawl_up_from_tip(ins, covered_length):
            straight_trajectory = ins.entry - ins.tip  # Straight line from entry to tip of the probe
            # scale generic covered region to the length of the trajectory
            covered_length_scaled = covered_length[:, np.newaxis] / np.linalg.norm(straight_trajectory)
            # starting from tip crawl scaled covered region
            covered_region = ins.tip + (straight_trajectory * covered_length_scaled)
            return covered_region

        # Coverage that we start from, is either given or instantiated with all zeros. Values outside brain set to nan
        if coverage is None:
            full_coverage = np.zeros(ba.image.shape, dtype=np.float32)
        else:
            full_coverage = copy.deepcopy(coverage)
        full_coverage[ba.label == 0] = np.nan
        full_coverage = full_coverage.flatten()
        # There are lists to collect the updated coverage with 0, 1 and 2 probes after each trajectory
        per2 = []
        per1 = []
        per0 = []

        for p, traj in enumerate(trajs):
            if len(trajs) > 20 and self.verbose is True:
                if p % 20 == 0:
                    print(p / len(trajs))
            # Get one trajectory from the list and create an insertion in the brain atlas
            # x and y coordinates of entry are translated to the atlas voxel space
            # z is locked to surface of the brain at these x,y coordinates (disregarding actual z value of trajectory)

            '''
            # Compute x-y distance taking into account angle of probe
            nx = int(np.ceil(NX / np.cos(traj['theta'] / 180 * np.pi)))
            ny = int(np.ceil(NY / np.cos(traj['phi'] / 180 * np.pi)))
            '''
            nx = NX
            ny = NY

            # Reset probes that have depth negative ; warning
            if traj['depth'] < 0:
                traj['depth'] = -traj['depth']
                print(f"Depth negative for {traj['probe_insertion']}, "
                      f"provenance {traj['provenance']}. "
                      "Sign flip for coverage computation. Please change on Alyx.")

            ins = atlas.Insertion.from_dict(traj, brain_atlas=ba)

            # Don't use probes that have same entry and tip, something is wrong
            set_nan = False
            if np.linalg.norm(ins.entry - ins.tip) == 0:
                print(f"Insertion entry and tip are identical for insertion {traj['probe_insertion']}, "
                      f"provenance {traj['provenance']}. "
                      "Skipping for coverage computation.")
                set_nan = True
            else:
                # Translate the top and bottom of the region considered covered from abstract to current insertion
                # Unit of atlas is m so divide um by 1e6
                top_bottom = crawl_up_from_tip(ins, covered_length_um / 1e6)
                # Check that z is the axis with the biggest deviation, don't use probes that are more shallow than deep
                axis = np.argmax(np.abs(np.diff(top_bottom, axis=0)))
                if axis != 2:
                    print(f"Z is not the longest axis for insertion {traj['probe_insertion']}. "
                          "Skipping for coverage computation.")
                    set_nan = True
            # Skip insertions with length zero or where z is not the longest axis
            if set_nan is True:
                if pl_voxels is not None:
                    per2.append(np.nan)
                    per1.append(np.nan)
                    per0.append(np.nan)
                continue
            # To sample the active track path along the longest axis, first get top and bottom voxel
            # If these lay outside of the atlas volume, clip to the nearest voxel in the volume
            tbi = ba.bc.xyz2i(top_bottom, mode='clip')
            # Number of voxels along the longest axis between top and bottom
            nz = tbi[1, axis] - tbi[0, axis] + 1
            # Create a set of nz voxels that track the path between top and bottom by equally spacing between the
            # x, y and z coordinates and then rounding
            ishank = np.round(
                np.array([np.linspace(tbi[0, i], tbi[1, i], nz) for i in np.arange(3)]).T).astype(np.int32)
            # Around each of the voxels along this shank, get a horizontal slice of voxels to consider
            # nx and ny are defined outside the loop as they don't depend on the trajectory
            # Instead of a set of slices, flatten these voxels. ixyz is of size (n_voxels, 3)
            ixyz = np.stack([v.flatten() for v in np.meshgrid(
                np.arange(-nx, nx + 1), np.arange(-ny, ny + 1), np.arange(nz))]).T
            # Add the x and y coordinates of the respective slice center voxel (defined by z coordinate)
            # So that the slice actually is at the right spot in the volume
            ixyz[:, 0] = ishank[ixyz[:, 2], 0] + ixyz[:, 0]
            ixyz[:, 1] = ishank[ixyz[:, 2], 1] + ixyz[:, 1]
            ixyz[:, 2] = ishank[ixyz[:, 2], 2]
            # If any, remove indices that lie outside the volume bounds
            iok = np.logical_and(0 <= ixyz[:, 0], ixyz[:, 0] < ba.bc.nx)
            iok &= np.logical_and(0 <= ixyz[:, 1], ixyz[:, 1] < ba.bc.ny)
            iok &= np.logical_and(0 <= ixyz[:, 2], ixyz[:, 2] < ba.bc.nz)
            ixyz = ixyz[iok, :]

            # Get the minimum distance of each of the considered voxels to the insertion
            # Translate voxel indices into distance from origin in atlas space
            xyz = np.c_[ba.bc.xscale[ixyz[:, 0]], ba.bc.yscale[ixyz[:, 1]], ba.bc.zscale[ixyz[:, 2]]]
            # This is the active region that we want to calculate the distance TO (here without MAX_DIST_UM)
            sites_bounds = crawl_up_from_tip(ins, (np.array([ACTIVE_LENGTH_UM, 0]) + TIP_SIZE_UM) / 1e6)
            # Calculate the minimum distance of each voxel in the search voxels to the active region
            mdist = ins.trajectory.mindist(xyz, bounds=sites_bounds)

            # Now we calculate the actual coverage
            # mdist gives us for each voxel in the search volume the distance of the CENTER of that voxel to the probe
            # If we want to include voxels where only part of the voxel, but not the center, falls into the radius
            # we consider covered, we need to increase the radius by the distance of a voxel's center to its corners
            # which is given by (sqrt(3) * side_length / 2).
            if entire_voxel is True:
                dist_adjusted = np.array(dist_fcn) / 1e6 + (np.abs(ba.bc.dxyz).max() * np.sqrt(3) / 2)
            else:
                dist_adjusted = np.array(dist_fcn) / 1e6
            # We then compute coverage using a cosine taper
            # Anything below the minimum distance will be 1
            # Anything above the maximum distance will be 0
            # Anything between minimum and maximum distance will slowly decrease from 1 to 0 with cosine taper
            coverage = 1 - fcn_cosine(dist_adjusted)(mdist)
            # If limit is set to True, remove effect of cosine taper and set everything under 1 to 0
            if limit:
                coverage[coverage < 1] = 0
            # Translate the flat coverage values to the volume
            flat_ind = ba._lookup_inds(ixyz)
            full_coverage[flat_ind] += (coverage * factor)

            # If a mask in which coverage should be calculated is given, restrict to this mask
            if pl_voxels is not None:
                n_pl_voxels = pl_voxels.shape[0]
                fp_voxels_2 = np.where(full_coverage[pl_voxels] >= 2)[0].shape[0]
                fp_voxels_1 = np.where(full_coverage[pl_voxels] == 1)[0].shape[0]
                fp_voxels_0 = np.where(full_coverage[pl_voxels] == 0)[0].shape[0]

                per2.append((fp_voxels_2 / n_pl_voxels) * 100)
                per1.append((fp_voxels_1 / n_pl_voxels) * 100)
                per0.append((fp_voxels_0 / n_pl_voxels) * 100)

        full_coverage = full_coverage.reshape(ba.image.shape)
        full_coverage[ba.label == 0] = np.nan

        if pl_voxels is not None:
            return full_coverage, per0, per1, per2
        else:
            return full_coverage, np.mean(xyz, 0)


    def compute_coverage_dict(self, trajs, dist_fcn=[50, 100], limit=True, coverage=None,
                         entire_voxel=True):
        """
        Computes coverage of the brain atlas associated with the probe model given a set of trajectories.

        Parameters
        ----------
        trajs: list of dictionary of trajectories from Alyx rest endpoint
        dist_fcn: list of two values in micro meter, the first one marks the distance to the probe, below which
                  voxels will be marked covered (=1). Coverage will then decrease with a cosine taper until the
                  second value, after which coverage will be 0.
        limit: boolean, if True, the effect of the cosine taper will be negated, i.e. all coverage values below 1
               will be set to 0.
        coverage: np.array of the same size as the brain atlas image associated with the probe insertion.
                  If given, the coverage computed by this function will be added to this input.
        pl_voxels: np.array with flat indices into the brain atlas image volume. If given, only these voxels are
                   considered for the computation of coverage percentages. Note that the coverage itself is always
                   computed for the entire volume
        factor: Number by which to multiply coverage of each trajectory, e.g. to count trajectories twice
        entire_voxel: boolean, if True, consider every voxel as covered, that is touched by the radius if dist_fcn[0]
                      around the probe. If False, only consider those as covered where the center of the voxel falls in
                      this radius.

        Returns
        3D np.array the same size as the brain atlas with coverage values for every voxel in the brain

        If pl_voxels is given:
        Three lists of values indicating the percent coverage of these voxels after adding each trajectory (increases
        incrementally). The first list indicates percent covered by 0 probes, second list percent covered by 1 probe,
        and the third list percent covered by 2 probes.
        """

        ba = self.ba
        ACTIVE_LENGTH_UM = 3.84 * 1e3  # This is the length of the NP1 probe with electrodes
        MAX_DIST_UM = dist_fcn[1]  # max distance around the probe to be searched for

        # Covered_length_um are two values which indicate on the path from tip to entry of a given insertion
        # where the region that is considered to be covered by this insertion begins and ends in micro meter
        # Note that the second value is negative, because the covered regions extends beyond the tip of the probe
        # We multiply by sqrt(2) to translate the radius given by dist_fcn[1] into the side length of a square
        # that is contained in the circle with radius dist_fcn[1]
        covered_length_um = TIP_SIZE_UM + np.array([ACTIVE_LENGTH_UM + MAX_DIST_UM * np.sqrt(2),
                                                    -MAX_DIST_UM * np.sqrt(2)])

        # Horizontal slice of voxels to be considered around each trajectory is only dependent on MAX_DIST_UM
        # and the voxel resolution, so can be defined here. We translate max dist in voxels and add 1 for safety
        # Unit of atlas is m so divide um by 1e6
        nx = int(np.ceil(MAX_DIST_UM / 1e6 / np.abs(ba.bc.dxyz[0]))) + 1
        ny = int(np.ceil(MAX_DIST_UM / 1e6 / np.abs(ba.bc.dxyz[1]))) + 1

        def crawl_up_from_tip(ins, covered_length):
            straight_trajectory = ins.entry - ins.tip  # Straight line from entry to tip of the probe
            # scale generic covered region to the length of the trajectory
            covered_length_scaled = covered_length[:, np.newaxis] / np.linalg.norm(straight_trajectory)
            # starting from tip crawl scaled covered region
            covered_region = ins.tip + (straight_trajectory * covered_length_scaled)
            return covered_region

        for p in np.arange(len(trajs)):
            if len(trajs) > 20 and self.verbose is True:
                if p % 20 == 0:
                    print(p / len(trajs))
            # Get one trajectory from the list and create an insertion in the brain atlas
            # x and y coordinates of entry are translated to the atlas voxel space
            # z is locked to surface of the brain at these x,y coordinates (disregarding actual z value of trajectory)
            traj = trajs[p]
            ins = atlas.Insertion.from_dict(traj, brain_atlas=ba)
            # Don't use probes that have same entry and tip, something is wrong
            set_nan = False
            if np.linalg.norm(ins.entry - ins.tip) == 0:
                print(f"Insertion entry and tip are identical for insertion {traj['probe_insertion']}. "
                      "Skipping for coverage computation.")
                set_nan = True
            else:
                # Translate the top and bottom of the region considered covered from abstract to current insertion
                # Unit of atlas is m so divide um by 1e6
                top_bottom = crawl_up_from_tip(ins, covered_length_um / 1e6)
                # Check that z is the axis with the biggest deviation, don't use probes that are more shallow than deep
                axis = np.argmax(np.abs(np.diff(top_bottom, axis=0)))
                if axis != 2:
                    print(f"Z is not the longest axis for insertion. "
                          "Skipping for coverage computation.")
                    set_nan = True
            # Skip insertions with length zero or where z is not the longest axis
            if set_nan is True:
                continue
            # To sample the active track path along the longest axis, first get top and bottom voxel
            # If these lay outside of the atlas volume, clip to the nearest voxel in the volume
            tbi = ba.bc.xyz2i(top_bottom, mode='clip')
            # Number of voxels along the longest axis between top and bottom
            nz = tbi[1, axis] - tbi[0, axis] + 1
            # Create a set of nz voxels that track the path between top and bottom by equally spacing between the
            # x, y and z coordinates and then rounding
            ishank = np.round(
                np.array([np.linspace(tbi[0, i], tbi[1, i], nz) for i in np.arange(3)]).T).astype(np.int32)
            # Around each of the voxels along this shank, get a horizontal slice of voxels to consider
            # nx and ny are defined outside the loop as they don't depend on the trajectory
            # Instead of a set of slices, flatten these voxels. ixyz is of size (n_voxels, 3)
            ixyz = np.stack([v.flatten() for v in np.meshgrid(
                np.arange(-nx, nx + 1), np.arange(-ny, ny + 1), np.arange(nz))]).T
            # Add the x and y coordinates of the respective slice center voxel (defined by z coordinate)
            # So that the slice actually is at the right spot in the volume
            ixyz[:, 0] = ishank[ixyz[:, 2], 0] + ixyz[:, 0]
            ixyz[:, 1] = ishank[ixyz[:, 2], 1] + ixyz[:, 1]
            ixyz[:, 2] = ishank[ixyz[:, 2], 2]
            # If any, remove indices that lie outside the volume bounds
            iok = np.logical_and(0 <= ixyz[:, 0], ixyz[:, 0] < ba.bc.nx)
            iok &= np.logical_and(0 <= ixyz[:, 1], ixyz[:, 1] < ba.bc.ny)
            iok &= np.logical_and(0 <= ixyz[:, 2], ixyz[:, 2] < ba.bc.nz)
            ixyz = ixyz[iok, :]

            # Get the minimum distance of each of the considered voxels to the insertion
            # Translate voxel indices into distance from origin in atlas space
            xyz = np.c_[ba.bc.xscale[ixyz[:, 0]], ba.bc.yscale[ixyz[:, 1]], ba.bc.zscale[ixyz[:, 2]]]
            # This is the active region that we want to calculate the distance TO (here without MAX_DIST_UM)
            sites_bounds = crawl_up_from_tip(ins, (np.array([ACTIVE_LENGTH_UM, 0]) + TIP_SIZE_UM) / 1e6)
            # Calculate the minimum distance of each voxel in the search voxels to the active region
            mdist = ins.trajectory.mindist(xyz, bounds=sites_bounds)

            # Now we calculate the actual coverage
            # mdist gives us for each voxel in the search volume the distance of the CENTER of that voxel to the probe
            # If we want to include voxels where only part of the voxel, but not the center, falls into the radius
            # we consider covered, we need to increase the radius by the distance of a voxel's center to its corners
            # which is given by (sqrt(3) * side_length / 2).
            if entire_voxel is True:
                dist_adjusted = np.array(dist_fcn) / 1e6 + (np.abs(ba.bc.dxyz).max() * np.sqrt(3) / 2)
            else:
                dist_adjusted = np.array(dist_fcn) / 1e6
            # We then compute coverage using a cosine taper
            # Anything below the minimum distance will be 1
            # Anything above the maximum distance will be 0
            # Anything between minimum and maximum distance will slowly decrease from 1 to 0 with cosine taper
            coverage = 1 - fcn_cosine(dist_adjusted)(mdist)
            # If limit is set to True, remove effect of cosine taper and set everything under 1 to 0
            if limit:
                coverage[coverage < 1] = 0
            # Translate the flat coverage values to the volume
            flat_ind = ba._lookup_inds(ixyz)

        return flat_ind[coverage != 0]


def coverage_with_insertions(csv_file, second_pass_volume, res=25):
    pr = ProbeModel(res=res)
    pr.initialise()
    pr.compute_best_for_provenance(provenance='Histology track')
    first_pass_coverage = pr.report_coverage(provenance='Best', dist=354)

    second_pass_coverage = np.load(second_pass_volume)
    second_pass_coverage = second_pass_coverage.flatten()
    second_pass_coverage[second_pass_coverage == 0] = np.nan
    ixyz_second = np.where(~np.isnan(second_pass_coverage.flatten()))[0]

    insertions = pd.read_csv(csv_file).to_dict(orient='records')

    # want to make it twice for each insertion

    new_coverage, per0, per1, per2 = pr.compute_coverage(insertions, dist_fcn=[354, 355], coverage=first_pass_coverage,
                                                         pl_voxels=ixyz_second, factor=1)

    return new_coverage, per0, per1, per2









