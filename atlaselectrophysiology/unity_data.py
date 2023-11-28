import oursin as urchin

from matplotlib.colors import Normalize, rgb2hex
from matplotlib import cm
import matplotlib
import numpy as np

import time


class UnityData:
    def __init__(self):
        urchin.setup()
        time.sleep(5)
        urchin.ccf25.load()
        time.sleep(5)
        self.add_root()
        self.particles = None

    def add_root(self):

        urchin.ccf25.root.set_visibility(True)
        urchin.ccf25.root.set_material('transparent-lit')
        urchin.ccf25.root.set_alpha(0.5)

    def add_data(self, data, points, feature, ba):

        urchin.particles.clear()

        positions, colours = self.prepare_data(data, points, feature, ba)
        particles = urchin.particles.create(len(positions))
        urchin.particles.set_positions(particles, positions)
        urchin.particles.set_colors(particles, colours)
        urchin.particles.set_material('circle')

        self.particles = particles

    def prepare_data(self, data, points, feature, ba):

        colours = []
        positions = []

        for shank, shank_data in data.items():
            shank_data = shank_data[points]

            feat = shank_data.get(feature, None)

            if feat is None:
                continue

            cols = self.data_to_colors(feat['data'], feat['cmap'], feat['levels'][0], feat['levels'][1])
            xyz = np.c_[shank_data['x'], shank_data['y'], shank_data['z']]
            mlapdv = ba.xyz2ccf(xyz, mode='clip')

            for i, loc in enumerate(mlapdv):
                colours.append(cols[i])
                positions.append(list(loc))

        return positions, colours

    @staticmethod
    def data_to_colors(data, cmap, vmin, vmax):

        cmap = cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=matplotlib.colormaps[cmap])
        cvals = cmap.to_rgba(data)
        chex = [rgb2hex(c) for c in cvals]
        return chex


