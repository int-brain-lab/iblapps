import oursin as urchin
import numpy as np
import time

from matplotlib.colors import Normalize, rgb2hex
from matplotlib import cm
import matplotlib
from atlaselectrophysiology.qt_utils.utils import shank_loop
from iblutil.util import Bunch
from iblatlas.atlas import AllenAtlas
from qtpy import QtCore, QtGui, QtWidgets
ba = AllenAtlas()

PLUGIN_NAME = "Unity 3D"

SHANK_COLOURS = {
    '0': '#000000',
    '1': '#000000',
    'a': '#000000',
    'b': '#30B666',
    'c': '#ff0044',
    'd': '#0000ff'
}


def setup(parent):

    parent.plugins[PLUGIN_NAME]= Bunch()
    parent.plugins[PLUGIN_NAME]['loader'] = Unity3d(parent)
    parent.plugins[PLUGIN_NAME]['loader'].init()
    parent.plugins[PLUGIN_NAME]['data_button_pressed'] = data_button_pressed
    parent.plugins[PLUGIN_NAME]['on_config_selected'] = update_plots
    parent.plugins[PLUGIN_NAME]['filter_unit_pressed'] = update_plots
    parent.plugins[PLUGIN_NAME]['update_plots'] = update_plots
    parent.plugins[PLUGIN_NAME]['plot_probe_panels'] = plot_probe_panels
    parent.plugins[PLUGIN_NAME]['plot_scatter_panels'] = plot_scatter_panels


    # Add a submenu to the main menu
    plugin_menu = QtWidgets.QMenu(PLUGIN_NAME, parent)
    parent.plugin_options.addMenu(plugin_menu)

    action = QtWidgets.QAction(f'Show Regions', parent, checkable=True, checked=False)
    action.triggered.connect(lambda a=action : parent.plugins[PLUGIN_NAME]['loader'].toggle_regions(action.isChecked()))
    plugin_menu.addAction(action)
    parent.plugins[PLUGIN_NAME]['toggle_action'] = action

    # TODO clean up
    widget = QtWidgets.QWidget()
    glayout3 = QtWidgets.QGridLayout()
    glayout3.setVerticalSpacing(0)
    min_label = QtWidgets.QLabel("0.1")
    max_label = QtWidgets.QLabel("1")
    max_label.setAlignment(QtCore.Qt.AlignRight)
    unity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    unity_slider.setMinimum(1)
    unity_slider.setMaximum(10)
    unity_slider.setValue(5)
    unity_slider.setTickPosition(QtWidgets.QSlider.TicksAbove)
    unity_slider.setTickInterval(1)
    unity_slider.sliderReleased.connect(lambda p=parent : on_point_size_changed(p))

    parent.plugins[PLUGIN_NAME]['unity_slider'] = unity_slider

    glayout3.addWidget(unity_slider, 0, 0, 1, 10)
    glayout3.addWidget(min_label, 1, 0, 1, 1)
    glayout3.addWidget(max_label, 1, 9, 1, 1)

    widget.setLayout(glayout3)

    slider_action = QtWidgets.QWidgetAction(parent)
    slider_action.setDefaultWidget(widget)
    plugin_menu.addAction(slider_action)


def on_point_size_changed(parent):
    point_size = parent.plugins[PLUGIN_NAME].unity_slider.value() / 100
    parent.plugins[PLUGIN_NAME]['loader'].set_point_size(point_size)

def update_plots(parent):
    if parent.plugins[PLUGIN_NAME]['loader'].plot == 'channels':
        parent.plugins[PLUGIN_NAME]['loader'].plot_channels(parent.plugins[PLUGIN_NAME]['loader'].plot_type)
    else:
        parent.plugins[PLUGIN_NAME]['loader'].plot_clusters(parent.plugins[PLUGIN_NAME]['loader'].plot_type)


def plot_probe_panels(parent, plot_type):
    parent.plugins[PLUGIN_NAME]['loader'].plot_channels(plot_type)


def plot_scatter_panels(parent, plot_type):
    if plot_type == 'Amplitude':
        return
    parent.plugins[PLUGIN_NAME]['loader'].plot_clusters(plot_type)


def data_button_pressed(parent):

    parent.plugins[PLUGIN_NAME]['loader'].delete_text()
    parent.plugins[PLUGIN_NAME]['loader'].toggle_regions(False)
    parent.plugins[PLUGIN_NAME]['loader'].init()
    regions = get_regions(parent)
    regions = np.unique(np.concatenate(regions))
    parent.plugins[PLUGIN_NAME]['loader'].add_regions(regions, parent.loaddata.hemisphere)
    parent.plugins[PLUGIN_NAME]['loader'].plot_channels(parent.probe_init)
    parent.plugins[PLUGIN_NAME]['toggle_action'].setChecked(True)


class Unity3d:
    def __init__(self, parent):
        self.parent = parent
        urchin.setup()
        time.sleep(5)
        urchin.ccf25.load()
        time.sleep(5)
        self.add_root()
        self.init()
        self.point_size = 0.05
        self.plot = 'channels'
        self.plot_type = None

    def init(self):

        self.particles = None
        self.probes = None
        self.regions = []
        self.text_list = []


    def add_root(self):

        urchin.ccf25.root.set_visibility(True)
        urchin.ccf25.root.set_material('transparent-lit')
        urchin.ccf25.root.set_alpha(0.5)

    def add_regions(self, regions, hemisphere):

        regions = [r for r in regions if r not in ['void', 'root']]
        self.regions = urchin.ccf25.get_areas(regions)
        self.side = urchin.utils.Side.LEFT if hemisphere == -1 else urchin.utils.Side.RIGHT

        urchin.ccf25.set_visibilities(self.regions, True, self.side)
        urchin.ccf25.set_materials(self.regions, 'transparent-lit')
        urchin.ccf25.set_alphas(self.regions, 0.25)


    def toggle_regions(self, display):
        if len(self.regions) > 0:
            urchin.ccf25.set_visibilities(self.regions, display, self.side)

    def delete_text(self):
        for text in self.text_list:
            text.delete()


    def set_points(self, points):

        self.particles = urchin.particles.ParticleSystem(n=len(points['pos']))
        self.particles.set_material('circle')
        self.particles.set_positions(points['pos'])
        self.particles.set_colors(points['col'])

    def set_point_size(self, point_size):

        self.point_size = point_size
        self.particles.set_sizes(list(np.ones(self.particles.data.n) * self.point_size * 1000))

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

    def plot_channels(self, plot_type):

        self.plot_type = plot_type
        self.plot = 'channels'

        if self.parent.loaddata.selected_config == 'both' or not self.parent.loaddata.selected_config:
            results = update_channels(self.parent, plot_type)
        else:
            results = update_channels(self.parent, plot_type, configs=[self.parent.loaddata.selected_config])

        self.add_data(results)

    def plot_clusters(self, plot_type):

        self.plot_type = plot_type
        self.plot = 'clusters'

        if self.parent.loaddata.selected_config == 'both' or not self.parent.loaddata.selected_config:
            results = update_clusters(self.parent, plot_type)
        else:
            results = update_clusters(self.parent, plot_type, configs=[self.parent.loaddata.selected_config])

        self.add_data(results)


    def add_data(self, results):

        colours = []
        positions = []
        probes = []

        for res in results:
            cols = res['values']
            xyz = res['xyz']
            mlapdv = ba.xyz2ccf(xyz, mode='clip')
            shank = res['shank']


            for i, loc in enumerate(mlapdv):
                colours.append(cols[i])
                # convert to ap ml dv order
                positions.append([loc[1], loc[0], loc[2]])

            # Find the position to put the probe indicators
            min_idx = np.argmin(mlapdv[:, 2])
            sh_info = {'name': shank,
                       'pos': [mlapdv[min_idx, 1], mlapdv[min_idx, 0], mlapdv[min_idx, 2] - 200],
                       'col': SHANK_COLOURS[shank[-1]]}
            if self.parent.loaddata.selected_config == 'both':
                if res['config'] == 'quarter':
                    probes.append(sh_info)
            else:
                probes.append(sh_info)


        urchin.particles.clear()
        self.set_points({'pos': positions, 'col': colours})
        self.set_point_size(self.point_size)
        self.set_probes(probes)
        self.set_text(probes)



def data_to_colors(data, cmap, vmin, vmax):

    cmap = cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=matplotlib.colormaps[cmap])
    cvals = cmap.to_rgba(data)
    chex = [rgb2hex(c) for c in cvals]
    return chex


@shank_loop
def get_regions(parent, items, **kwargs):
    return np.unique(items.hist_data['axis_label'][:, 1])

@shank_loop
def update_clusters(parent, items, plot_type, **kwargs):

    xyz = parent.loaddata.xyz_clusters
    data = parent.loaddata.scatter_plots.get(plot_type, None)

    values = data_to_colors(data.colours, data.cmap, data.levels[0], data.levels[1])

    return {'xyz': xyz, 'values': values, 'shank': kwargs['shank'], 'config': kwargs['config']}

@shank_loop
def update_channels(parent, items, plot_type, **kwargs):

    xyz = parent.loaddata.xyz_channels
    data = parent.loaddata.probe_plots.get(plot_type, None)
    vals = np.concatenate(data.img, axis=1)[0]

    # We need to do this because the probe plots are split by bank
    idx = np.concatenate(data.idx)
    vals = vals[idx]

    values = data_to_colors(vals, data.cmap, data.levels[0], data.levels[1])

    return {'xyz': xyz, 'values': values, 'shank': kwargs['shank'], 'config': kwargs['config']}