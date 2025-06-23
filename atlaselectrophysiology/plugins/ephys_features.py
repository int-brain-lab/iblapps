import numpy as np
from PyQt5 import QtWidgets
from ephysfeatures.features_across_region import RegionFeatureWindow

PLUGIN_NAME = 'Ephys features'

def setup(parent):
    parent.plugins[PLUGIN_NAME] = dict()
    # parent.plugins[PLUGIN_NAME]['loader'] = LoadData(parent.shank.one, parent.shank.brain_atlas)

    action = QtWidgets.QAction(PLUGIN_NAME, parent)
    action.triggered.connect(lambda: callback(parent))
    parent.plugin_options.addAction(action)

def callback(parent):
    parent.region_win = RegionFeatureWindow(parent.shank.one, np.unique(np.array(parent.shank.align.ephysalign.region_id).ravel()),
                                            parent.shank.brain_atlas, download=True)
    parent.region_win.show()