from iblviewer.atlas_controller import AtlasController
import vedo
from iblviewer.atlas_model import AtlasModel, AtlasUIModel, CameraModel
from iblviewer.slicer_model import SlicerModel

from iblviewer.atlas_view import AtlasView
from iblviewer.volume_view import VolumeView
from iblviewer.slicer_view import SlicerView
import iblviewer.utils as utils

from ipyvtk_simple.viewer import ViewInteractiveWidget


class NeedlesViewer(AtlasController):
    def __init__(self):
        super(NeedlesViewer, self).__init__()

    def initialize(self, plot, resolution=25, mapping='Allen', volume_mode=None, num_windows=1, render=False):
        vedo.settings.allowInteraction = False
        self.plot = plot
        self.plot_window_id = 0
        self.model = AtlasModel()
        self.model.initialize(resolution)
        self.model.load_allen_volume(mapping, volume_mode)
        self.model.initialize_slicers()

        self.view = AtlasView(self.plot, self.model)
        self.view.initialize()
        self.view.volume = VolumeView(self.plot, self.model.volume, self.model)

        #pn = SlicerModel.NAME_XYZ_POSITIVE
        nn = SlicerModel.NAME_XYZ_NEGATIVE

        #pxs_model = self.model.find_model(pn[0], self.model.slicers)
        #self.px_slicer = SlicerView(self.plot, self.view.volume, pxs_model, self.model)
        #pys_model = self.model.find_model(pn[1], self.model.slicers)
        #self.py_slicer = SlicerView(self.plot, self.view.volume, pys_model, self.model)
        #pzs_model = self.model.find_model(pn[2], self.model.slicers)
        #self.pz_slicer = SlicerView(self.plot, self.view.volume, pzs_model, self.model)

        nxs_model = self.model.find_model(nn[0], self.model.slicers)
        self.nx_slicer = SlicerView(self.plot, self.view.volume, nxs_model, self.model)
        nys_model = self.model.find_model(nn[1], self.model.slicers)
        self.ny_slicer = SlicerView(self.plot, self.view.volume, nys_model, self.model)
        nzs_model = self.model.find_model(nn[2], self.model.slicers)
        self.nz_slicer = SlicerView(self.plot, self.view.volume, nzs_model, self.model)

        self.slicers = [self.nx_slicer, self.ny_slicer, self.nz_slicer]

        vedo.settings.defaultFont = self.model.ui.font
        self.initialize_embed_ui(slicer_target=self.view.volume)

        self.plot.show(interactive=False)
        self.handle_transfer_function_update()
        # By default, the atlas volume is our target
        self.model.camera.target = self.view.volume.actor
        # We start with a sagittal view
        self.set_left_view()

        self.render()

    def initialize_embed_ui(self, slicer_target=None):
        s_kw = self.model.ui.slider_config
        d = self.view.volume.model.dimensions
        if d is None:
            return

        #self.add_slider('px', self.update_px_slicer, 0, int(d[0]), 0, pos=(0.05, 0.065, 0.12),
        #                title='+X', **s_kw)
        #self.add_slider('py', self.update_py_slicer, 0, int(d[1]), 0, pos=(0.2, 0.065, 0.12),
        #                title='+Y', **s_kw)
        #self.add_slider('pz', self.update_pz_slicer, 0, int(d[2]), 0, pos=(0.35, 0.065, 0.12),
        #                title='+Z', **s_kw)

    def update_slicer(self, slicer_view, value):
        """
        Update a given slicer with the given value
        :param slicer_view: SlicerView instance
        :param value: Value
        """
        volume = self.view.volume
        model = slicer_view.model
        model.set_value(value)
        model.clipping_planes = volume.get_clipping_planes(model.axis)
        slicer_view.update(add_to_scene=self.model.slices_visible)


