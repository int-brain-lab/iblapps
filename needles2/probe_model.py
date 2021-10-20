import numpy as np
from iblutil.numerical import ismember
from one.api import ONE
from ibllib.pipes import histology
from ibllib.atlas import AllenAtlas, atlas
from ibllib.ephys.neuropixel import TIP_SIZE_UM, SITES_COORDINATES
from ibllib.pipes.ephys_alignment import EphysAlignment
import time
from scipy.signal import fftconvolve
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
                  'probe_insertion__session__extended_qc__behavior,1,'
                  'probe_insertion__session__json__IS_MOCK,False']


        django_str = ','.join(django_base + django)


        ## TODO GC REMOVE
        list_pid = \
            ['134e8241-f5d7-4ad7-a8c0-11b34f2eedc3',
             '6d3b68e0-3efd-4b03-b747-16e44118a0a9',
             '6a098711-5423-4072-8909-7cff0e2d4531',
             '3ccb2d59-9e94-48e6-9e72-0b7b96bd3f9b',
             'a69887c2-f35d-497c-a2db-d9c44406369a',
             'c0c3c95d-43c3-4e30-9ce7-0519d0473911',
             '05ccc92c-fcb0-4e92-84d3-de033890c7a8',
             '1a276285-8b0e-4cc9-9f0a-a3a002978724',
             '54c8f5de-83d7-49b9-9b4c-67fe0f07289b',
             '7dcd74a9-fc7a-4c3c-85b9-769d869629ec',
             '2a7caaa4-1710-49a1-ade4-db0a986ae6f5',
             '37137b1d-34d4-4183-9796-abfc9ffb6abe',
             '450cea3b-9289-4708-9df2-a2518bdc4b59',
             '1a6a17cc-ba8c-4d79-bf20-cc897c9500dc',
             '91fb4ed3-ee27-4d7e-ba5d-f61202ed9884',
             '4c04120d-523a-4795-ba8f-49dbb8d9f63a',
             '8dfb86c8-d45c-46c4-90ec-33078014d434',
             '3e7618b8-34ca-4e48-ba3a-0e0f88a43131',
             '72274871-80e5-4fb9-be1a-bb04acebb1de',
             'f8ef3dfc-f789-43a1-9015-c82ffa8e7df0',
             '491540f7-7878-410f-894e-7ac4766b5105',
             '180d278e-1a49-42da-a624-24aa14b57a4e',
             '3c15054d-9a26-40cc-af3b-d11b1cfe2212',
             '396a41bc-78fc-4fc5-b559-b83a5cdb86fd',
             '4037a618-e6ba-4142-a6d2-3e32ef25fd1b',
             '262337f9-2cff-4dc0-8b1b-65b34bb41d5e',
             '31f0400a-71f5-436c-ada4-dce50627dc73',
             '5a9f8899-556a-43ea-892d-5e35b969ff38',
             '1d1f11e4-fa84-43b4-9b89-f2ac777b8d96',
             '30dfb8c6-9202-43fd-a92d-19fe68602b6f',
             'd14f70e6-bf7b-4d6d-a380-bfd0a46ed7a1',
             '64d04585-67e7-4320-baad-8d4589fd18f7',
             'f698f893-da01-49a1-ae4b-16f7e42e5ff5',
             '4f46198d-c54d-4627-9e4e-3c8fd4c3c070',
             '0ece5c6a-7d1e-4365-893d-ac1cc04f1d7b']
        test = self.one.alyx.rest('trajectories', 'list', provenance=provenance, django=django_str)
        test_2 = [item for item in test if (item['probe_insertion'] not in list_pid)]
        self.traj[prov_dict]['traj'] = np.array(test_2)

        ##TODO GC UNCOMMENT
        # self.traj[prov_dict]['traj'] = np.array(self.one.alyx.rest('trajectories', 'list',
        #                                                            provenance=provenance,
        #                                                            django=django_str))

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


    def compute_coverage(self, trajs, dist_fcn=[50, 100], limit=True):
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
        full_coverage = np.zeros(ba.image.shape, dtype=np.float32).flatten()

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
            full_coverage[flat_ind] += coverage

        full_coverage = full_coverage.reshape(ba.image.shape)
        full_coverage[ba.label == 0] = np.nan

        return full_coverage, np.mean(xyz, 0)
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

#from vedo import *
#from ibllib.atlas import AllenAtlas
#ba = AllenAtlas()
#import numpy as np
#import vtk
#la = ba.label
#la[la != 0] = 1
#vol2 = Volume(la).alpha([0, 0, 0.5])
#vol = Volume(ba.image).alpha([0, 0, 0.8]).c('bone').pickable(False)
##vol = Volume(ba.image).c('bone').pickable(False)
#
#plane = vtk.vtkPlane()
#clipping_planes = vtk.vtkPlaneCollection()
#clipping_planes.AddItem(plane)
#vol2.mapper().SetClippingPlanes(clipping_planes)
##
#plane.SetOrigin(vol.center() + np.array([0, -100, 0]))
#plane.SetNormal(0.5, 0.866, 0)
##
#sl = vol.slicePlane(origin=vol.center() + np.array([0, 100, 50]), normal=(0.5, 0.866, 0))
#s2 = vol.slicePlane(origin=vol.center(), normal=(0.5, 0, 0.866))
#s3 = vol.slicePlane(origin=vol.center() + np.array([0, -100, -50]), normal=(1, 0, 0)) # this is 30 degree in coronal
##s3 = vol.slicePlane(origin=vol.center(), normal=(0.5, 0, 0.866)) # this is 30 degree in coronal
#
#sl.cmap('Purples_r').lighting('off').addScalarBar(title='Slice', c='w')
#s2.cmap('Blues_r').lighting('off')
#s3.cmap('Greens_r').lighting('off')
#def func(evt):
#    if not evt.actor:
#        return
#    pid = evt.actor.closestPoint(evt.picked3d, returnPointId=True)
#    txt = f"Probing:\n{precision(evt.actor.picked3d, 3)}\nvalue = {pid}"
#    sph = Sphere(evt.actor.points(pid), c='orange7').pickable(False)
#    vig = sph.vignette(txt, s=7, offset=(-150,15), font=2).followCamera()
#    plt.remove(plt.actors[-2:]).add([sph, vig]) #remove old 2 & add the new 2
#
#plt = show(vol, sl, s2, s3, __doc__, axes=9, bg='k', bg2='bb', interactive=False)
#plt.actors += [None, None]  # 2 placeholders for [sphere, vignette]
#plt.addCallback('as my mouse moves please call', func)
#interactive()
#
#from vedo.utils import versor
#
#
#from vedo import *
#
#vol = Volume(dataurl+'embryo.slc').alpha([0,0,0.5]).c('k')
#
#slices = []
#for i in range(4):
#    sl = vol.slicePlane(origin=[150,150,i*50+50], normal=(-1,0,1))
#    slices.append(sl)
#
#amap = [0, 1, 1, 1, 1]  # hide low value points giving them alpha 0
#mslices = merge(slices) # merge all slices into a single Mesh
#mslices.cmap('hot_r', alpha=amap).lighting('off').addScalarBar3D()
#
#show(vol, mslices, __doc__, axes=1)