from pathlib import Path
from oneibl.one import ONE
import qt


import atlaselectrophysiology.ephys_atlas_gui as alignment_window
import data_exploration_gui.gui_main as trial_window


# some extra controls

class AlignmentWindow(alignment_window.MainWindow):
    def __init__(self, offline=False, probe_id=None, one=None):
        super(AlignmentWindow, self).__init__()
        self.trial_gui = None


    def cluster_clicked(self, item, point):
        clust = super().cluster_clicked(item, point)
        print(clust)



class TrialWindow(trial_window.MainWindow):
    def __init__(self):
        super(TrialWindow, self).__init__()
        self.alignment_gui = None


def viewer(probe_id=None, one=None):
    """
    """
    qt.create_app()
    av = AlignmentWindow()
    bv = TrialWindow()

    av.show()
    bv.show()
    return av

