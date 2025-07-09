import datoviz as dvz
import numpy as np
from pywavefront import Wavefront
import urllib.request
import numpy as np
from one import params
from atlaselectrophysiology.qt_utils.utils import shank_loop
from iblatlas.atlas import AllenAtlas
atlas = AllenAtlas()

PLUGIN_NAME = "Datoviz 3D"

def setup(parent):

    parent.plugins[PLUGIN_NAME]['loader'] = Datoviz3d(parent)
    parent.plugins[PLUGIN_NAME]['data_button_pressed'] = data_button_pressed


def on_config_selected(parent):
    # change the number of points that are displayed

    pass


def update_plots():
    # change the xyz pos of the points
    pass





def data_button_pressed(parent):

    print('in datoviz data buttong pressed')

    # get the points for the selected config
    if parent.loaddata.selected_config == 'both' or not parent.loaddata.selected_config:
        results = get_n_points(parent)
    else:
        results = get_n_points(parent, configs=[parent.loaddata.selected_config])

    n_clusters = 0
    n_channels = 0
    for res in results:
        n_clusters += res['clusters']
        n_channels += res['channels']


    parent.plugins[PLUGIN_NAME]['loader'].add_channels(n_channels)
    parent.plugins[PLUGIN_NAME]['loader'].add_clusters(n_clusters)

    parent.plugins[PLUGIN_NAME]['loader'].plot_channels(parent.probe_init)

    parent.plugins[PLUGIN_NAME]['loader'].app.run()
    parent.plugins[PLUGIN_NAME]['loader'].app.destroy()


class Datoviz3d:
    def __init__(self, parent):
        self.parent = parent
        self.channel_points = None
        self.cluster_points = None
        self.plot = 'channels'
        self.plot_type = None
        self.is_setup = False

    def setup(self):

        print('in this setup')
        region_idx = 997
        m = load_mesh(region_idx=region_idx)
        mesh_pos = m['pos']
        mesh_idx = m['idx']
        mesh_color = m['color']
        mesh_pos = atlas.ccf2xyz(mesh_pos, ccf_order='apdvml')

        self.center = mesh_pos.mean(axis=0)
        mesh_pos -= self.center
        mesh_pos *= 200

        # Mesh.
        nv = mesh_pos.shape[0]
        ni = mesh_idx.size

        mesh_pos = np.ascontiguousarray(mesh_pos, dtype=np.float32)
        mesh_color = np.tile(mesh_color, (nv, 1)).astype(np.uint8)
        mesh_color[:, 3] = 32
        mesh_idx = mesh_idx.astype(np.uint32).ravel()

        self.app = dvz.App(background='white')
        figure = self.app.figure(gui=True)
        self.panel = figure.panel()
        arcball = self.panel.arcball()

        visual = self.app.mesh(indexed=True, lighting=True, cull='back')
        visual.set_data(
            position=mesh_pos,
            color=mesh_color,
            index=mesh_idx,
            compute_normals=True,
        )
        self.panel.add(visual)
        self.is_setup = True

        # self.app.run()
        # self.app.destroy()

    def add_channels(self, n_points):

        if self.channel_points is not None:
            self.panel.remove(self.channel_points)

        clust_size = np.full(n_points, 6, dtype=np.float32)

        self.channel_points = self.app.point(
            depth_test=False,
            size=clust_size,
        )
        self.panel.add(self.channel_points)

    def add_clusters(self, n_points):

        if self.cluster_points is not None:
            self.panel.remove(self.cluster_points)

        clust_size = np.full(n_points, 6, dtype=np.float32)

        self.cluster_points = self.app.point(
            depth_test=False,
            size=clust_size,
        )

        # Don't add as initialise with channels
        # self.panel.add(self.cluster_points)


    def plot_channels(self, plot_type):
        if self.plot == 'clusters':
            self.panel.remove(self.cluster_points)
            self.panel.add(self.channel_points)
            self.plot = 'channels'


        self.plot_type = plot_type

        if self.parent.loaddata.selected_config == 'both' or not self.parent.loaddata.selected_config:
            results = update_channels(self.parent, plot_type)
        else:
            results = update_channels(self.parent, plot_type, configs=[self.parent.loaddata.selected_config])

        values = []
        points = []
        for res in results:
            values.append(res['values'])
            points.append(res['xyz'])

        pos = np.concatenate(points)
        pos -= self.center
        pos *= 200

        self.channel_points.set_position(pos)
        self.channel_points.set_color(np.concatenate(values))


    def plot_clusters(self, plot_type):
        if self.plot == 'channels':
            self.panel.remove(self.channel_points)
            self.panel.add(self.cluster_points)
            self.plot = 'clusters'

        self.plot_type = plot_type

        if self.parent.loaddata.selected_config == 'both' or not self.parent.loaddata.selected_config:
            results = update_clusters(self.parent, plot_type)
        else:
            results = update_clusters(self.parent, plot_type, configs=[self.parent.loaddata.selected_config])
        values = []
        points = []
        for res in results:
            values.append(res['values'])
            points.append(res['xyz'])

        pos = np.concatenate(points)
        pos -= self.center
        pos *= 200

        self.cluster_points.set_position(pos)
        self.cluster_points.set_color(np.concatenate(values))



@shank_loop
def update_clusters(parent, items, plot_type, **kwargs):

    xyz = parent.loaddata.xyz_clusters
    data = parent.loaddata.scatter_plots.get(plot_type, None)
    values = dvz.cmap(data.cmap, data.colours, data.levels[0], data.levels[1])

    return {'xyz': xyz, 'values': values}

@shank_loop
def update_channels(parent, items, plot_type, **kwargs):

    xyz = parent.loaddata.xyz_channels
    data = parent.loaddata.probe_plots.get(plot_type, None)
    vals = np.concatenate(data.img)[0]

    idx = np.concatenate(data.idx)
    sort_idx = np.argsort(idx)
    vals= vals[sort_idx]
    values = dvz.cmap(data.cmap, vals, data.levels[0], data.levels[1])

    return {'xyz': xyz, 'values': values}


@shank_loop
def get_n_points(parent, items, **kwargs):

    n_clusters = parent.loaddata.xyz_clusters.shape[0]
    n_channels = parent.loaddata.xyz_channels.shape[0]

    return {'channels': n_channels, 'clusters': n_clusters}



CCF_URL = 'http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/structure_meshes/'
MESH_PATH = params.get_cache_dir().joinpath('histology', 'ATLAS', 'mesh')
MESH_PATH.mkdir(parents=True, exist_ok=True)
OBJ_PATH = params.get_cache_dir().joinpath('histology', 'ATLAS', 'obj')
OBJ_PATH.mkdir(parents=True, exist_ok=True)

def dl(atlas_id):
    mesh_url = CCF_URL + str(atlas_id) + '.obj'
    fn = OBJ_PATH.joinpath(f"{atlas_id}.obj")
    try:
        urllib.request.urlretrieve(mesh_url, fn)
    except Exception as e:
        print(f"Error: {str(e)}")


def get_color(atlas_id, alpha=255, br=None):
    _, idx = br.id2index(atlas_id)
    color = br.rgb[idx[0][0], :]
    return np.hstack((color, [alpha])).astype(np.uint8)


def load_obj(path):
    obj = Wavefront(path, collect_faces=True, create_materials=True)
    pos = np.array(obj.vertices)
    idx = np.array(obj.mesh_list[0].faces)
    return pos, idx


def load_region(atlas_id, br=None):
    obj_path = OBJ_PATH.joinpath(f"{atlas_id}.obj")
    if not obj_path.exists():
        dl(atlas_id)
    pos, idx = load_obj(OBJ_PATH.joinpath(f"{atlas_id}.obj"))
    color = get_color(atlas_id, br=br)
    return pos, idx, color


def load_mesh(region_idx=315):
    fn = MESH_PATH.joinpath(f"mesh-{region_idx:03d}.npz")
    try:
        m = np.load(fn)
        # print(f"Loaded mesh {fn}")
    except (IOError, FileNotFoundError):
        br = atlas.regions
        pos, idx, color = load_region(region_idx, br=br)

        m = dict(pos=pos, idx=idx, color=color)
        np.savez(fn, **m)
        print(f"Saved {fn}")
    return m

