from dataclasses import dataclass

import numpy as np

import datoviz as dviz

# -------------------------------------------------------------------------------------------------
# Raster viewer
# -------------------------------------------------------------------------------------------------


class RasterView:
    def __init__(self):
        self.canvas = dviz.canvas(show_fps=True)
        self.panel = self.canvas.panel(controller='axes')
        self.visual = self.panel.visual('point')
        self.pvars = {'ms': 2., 'alpha': .03}
        self.gui = self.canvas.gui('XY')
        self.gui.control("label", "Coords", value="(0, 0)")

    def set_spikes(self, spikes):
        pos = np.c_[spikes.times, spikes.depths, np.zeros_like(spikes.times)]
        color = dviz.colormap(20 * np.log10(spikes.amps), cmap='cividis', alpha=self.pvars['alpha'])
        self.visual.data('pos', pos)
        self.visual.data('color', color)
        self.visual.data('ms', np.array([self.pvars['ms']]))


class RasterController:
    _time_select_cb = None

    def __init__(self, model, view):
        self.m = model
        self.v = view
        self.v.canvas.connect(self.on_mouse_move)
        self.v.canvas.connect(self.on_key_press)
        self.redraw()

    def redraw(self):
        print('redraw', self.v.pvars)
        self.v.set_spikes(self.m.spikes)

    def on_mouse_move(self, x, y, modifiers=()):
        p = self.v.canvas.panel_at(x, y)
        if not p:
            return
        # Then, we transform into the data coordinate system
        # Supported coordinate systems:
        #   target_cds='data' / 'scene' / 'vulkan' / 'framebuffer' / 'window'
        xd, yd = p.pick(x, y)
        self.v.gui.set_value("Coords", f"({xd:0.2f}, {yd:0.2f})")

    def on_key_press(self, key, modifiers=()):
        print(key, modifiers)
        if key == 'a' and modifiers == ('control',):
            self.v.pvars['alpha'] = np.minimum(self.v.pvars['alpha'] + 0.1, 1.)
        elif key == 'z' and modifiers == ('control',):
            self.v.pvars['alpha'] = np.maximum(self.v.pvars['alpha'] - 0.1, 0.)
        elif key == 'page_up':
            self.v.pvars['ms'] = np.minimum(self.v.pvars['ms'] * 1.1, 20)
        elif key == 'page_down':
            self.v.pvars['ms'] = np.maximum(self.v.pvars['ms'] / 1.1, 1)
        else:
            return
        self.redraw()


@dataclass
class RasterModel:
    spikes: dict


def raster(spikes):
    rm = RasterController(RasterModel(spikes), RasterView())
    dviz.run()
