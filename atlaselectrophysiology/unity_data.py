import oursin as urchin

from matplotlib.colors import Normalize, rgb2hex
from matplotlib import cm
import matplotlib
import numpy as np

import time

SHANK_COLOURS = {
    '0': '#000000',
    '1': '#000000',
    'a': '#000000',
    'b': '#30B666',
    'c': '#ff0044',
    'd': '#0000ff'
}


class UnityData:
    def __init__(self):
        urchin.setup()
        time.sleep(8)
        urchin.ccf25.load()
        time.sleep(8)
        self.add_root()
        self.init()

    def init(self):

        self.particles = None
        self.probes = None
        self.regions = []
        self.text_list = []


    def add_root(self):

        urchin.ccf25.root.set_visibility(True)
        urchin.ccf25.root.set_material('transparent-lit')
        urchin.ccf25.root.set_alpha(0.5)

    def add_regions(self, regions):

        regions = [r for r in regions if r not in ['void', 'root']]
        self.regions = urchin.ccf25.get_areas(regions)
        urchin.ccf25.set_visibilities(self.regions, True, urchin.utils.Side.LEFT)
        urchin.ccf25.set_materials(self.regions, 'transparent-lit', 'left')
        urchin.ccf25.set_alphas(self.regions, 0.25, 'left')

    def toggle_regions(self, display):
        if len(self.regions) > 0:
            urchin.ccf25.set_visibilities(self.regions, display, urchin.utils.Side.LEFT)

    def delete_text(self):
        for text in self.text_list:
            text.delete()

    def add_data(self, data, point_type, feature, filter_type, point_size, ba):

        urchin.particles.clear()
        points, probes = self.prepare_data(data, point_type, feature, filter_type, ba)
        self.set_points(points)
        self.set_point_size(point_size)
        self.set_probes(probes)
        self.set_text(probes)

    def set_points(self, points):

        self.particles = urchin.particles.ParticleSystem(n=len(points['pos']))
        self.particles.set_material('circle')
        self.particles.set_positions(points['pos'])
        self.particles.set_colors(points['col'])

    def set_point_size(self, point_size):

        self.particles.set_sizes(list(np.ones(self.particles.data.n) * point_size * 1000))

    def set_text(self, text):

        if len(self.text_list) == 0 and len(text) > 0:
            text = sorted(text, key=lambda x: x['name'])
            self.text_list = urchin.text.create(len(text))
            urchin.text.set_texts(self.text_list, [t['name'] for t in text])
            urchin.text.set_positions(self.text_list, [[-0.95, 0.95], [-0.95, 0.9], [-0.95, 0.85], [-0.95, 0.8]])
            urchin.text.set_font_sizes(self.text_list, [24, 24, 24, 24])
            urchin.text.set_colors(self.text_list, [t['col'] for t in text])

    def set_probes(self, probes):

        self.probes = urchin.particles.ParticleSystem(n=len(probes))
        self.probes.set_material('circle')
        self.probes.set_positions([p['pos'] for p in probes])
        self.probes.set_colors([p['col'] for p in probes])
        self.probes.set_sizes(list(np.ones(self.probes.data.n) * 250))

    def prepare_data(self, data, point_type, feature, filter_type, ba):

        colours = []
        positions = []
        probes = []

        for shank, shank_data in data.items():
            sh_data = shank_data[point_type]
            if point_type == 'clusters':
                clu_idx = shank_data['cluster_idx'][filter_type]
            else:
                clu_idx = np.arange(sh_data['x'].size)

            feat = sh_data.get(feature, None)

            if feat is None:
                continue

            cols = self.data_to_colors(feat['data'][clu_idx], feat['cmap'], feat['levels'][0], feat['levels'][1])
            xyz = np.c_[sh_data['x'][clu_idx], sh_data['y'][clu_idx], sh_data['z'][clu_idx]]
            mlapdv = ba.xyz2ccf(xyz, mode='clip')

            for i, loc in enumerate(mlapdv):
                colours.append(cols[i])
                # convert to ap ml dv order
                positions.append([loc[1], loc[0], loc[2]])

            # Find the position to put the probe indicators
            min_idx = np.argmin(mlapdv[:, 2])
            sh_info = {'name': shank,
                       'pos': [mlapdv[min_idx, 1], mlapdv[min_idx, 0], mlapdv[min_idx, 2] - 200],
                       'col': SHANK_COLOURS[shank[-1]]}
            probes.append(sh_info)

        return {'pos': positions, 'col': colours}, probes

    @staticmethod
    def data_to_colors(data, cmap, vmin, vmax):

        cmap = cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=matplotlib.colormaps[cmap])
        cvals = cmap.to_rgba(data)
        chex = [rgb2hex(c) for c in cvals]
        return chex


