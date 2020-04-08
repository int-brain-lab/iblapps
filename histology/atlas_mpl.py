from pathlib import Path
import matplotlib
import numpy as np
from PyQt5 import QtCore, QtWidgets, uic

from iblapps import qt
from iblapps.qt_matplotlib import BaseMplCanvas
import ibllib.atlas as atlas

# Make sure that we are using QT5
matplotlib.use('Qt5Agg')


class Model:
    """
    Container for Data and variables of the application
    """
    brain_atlas: atlas.BrainAtlas
    ap_um: float

    def __init__(self, brain_atlas=None, ap_um=0):
        self.ap_um = ap_um
        # load the brain atlas
        if brain_atlas is None:
            self.brain_atlas = atlas.AllenAtlas(res_um=25)
        else:
            self.brain_atlas = brain_atlas


class MyStaticMplCanvas(BaseMplCanvas):

    def __init__(self, *args, **kwargs):
        super(MyStaticMplCanvas, self).__init__(*args, **kwargs)
        self.mpl_connect("motion_notify_event", self.on_move)

    def on_move(self, event):
        mw = qt.get_main_window()
        ap = mw._model.ap_um
        xlab = '' if event.xdata is None else '{: 6.0f}'.format(event.xdata)
        ylab = '' if event.ydata is None else '{: 6.0f}'.format(event.ydata)
        mw.label_ml.setText(xlab)
        mw.label_dv.setText(ylab)
        mw.label_ap.setText('{: 6.0f}'.format(ap))
        if event.xdata is None or event.ydata is None:
            return
        # if whithin the bounds of the Atlas, label with the current strucure hovered onto
        xyz = np.array([[event.xdata, ap, event.ydata]]) / 1e6
        regions = mw._model.brain_atlas.regions
        id_label = mw._model.brain_atlas.get_labels(xyz)
        il = np.where(regions.id == id_label)[0]
        mw.label_acronym.setText(regions.acronym[il][0])
        mw.label_structure.setText(regions.name[il][0])
        mw.label_id.setText(str(id_label))

    def update_slice(self, volume='image'):
        mw = qt.get_main_window()
        ap_um = mw._model.ap_um
        im = mw._model.brain_atlas.slice(ap_um / 1e6, axis=1, volume=volume)
        im = np.swapaxes(im, 0, 1)
        self.axes.images[0].set_data(im)
        self.draw()


class AtlasViewer(QtWidgets.QMainWindow):
    def __init__(self, brain_atlas=None, ap_um=0):
        # init the figure
        super(AtlasViewer, self).__init__()
        uic.loadUi(str(Path(__file__).parent.joinpath('atlas_mpl.ui')), self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)
        self.pb_toggle.clicked.connect(self.update_slice)

        # init model
        self._model = Model(brain_atlas=brain_atlas, ap_um=ap_um)
        # display the coronal slice in the mpl widget
        self._model.brain_atlas.plot_cslice(ap_coordinate=ap_um / 1e6,
                                            ax=self.mpl_widget.axes)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def update_slice(self, ce):
        volume = self.pb_toggle.text()
        if volume == 'Annotation':
            self.pb_toggle.setText('Image')
        elif volume == 'Image':
            self.pb_toggle.setText('Annotation')
        self.mpl_widget.update_slice(volume.lower())


def viewatlas(brain_atlas=None, title=None, ap_um=0):
    qt.create_app()
    av = AtlasViewer(brain_atlas, ap_um=ap_um)
    av.setWindowTitle(title)
    av.show()
    return av, av.mpl_widget.axes


if __name__ == "__main__":
    w = viewatlas(title="Allen Atlas - IBL")
    qt.run_app()
