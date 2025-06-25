from atlaselectrophysiology.qt_utils.utils import PopupWindow, set_axis
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets
import numpy as np

PLUGIN_NAME = "Cluster Popup"

def setup(parent):

    parent.plugins[PLUGIN_NAME] = dict()
    parent.plugins[PLUGIN_NAME]['loader'] = ClusterPopups(parent)
    parent.plugins[PLUGIN_NAME]['callback'] = callback

    # Add a submenu to the main menu
    plugin_menu = QtWidgets.QMenu(PLUGIN_NAME, parent)
    parent.plugin_options.addMenu(plugin_menu)

    action = QtWidgets.QAction(f'Minimise/Show {PLUGIN_NAME}', parent)
    action.setShortcut('M')
    action.triggered.connect(parent.plugins[PLUGIN_NAME]['loader'].minimise_popups)
    plugin_menu.addAction(action)

    action = QtWidgets.QAction(f'Close {PLUGIN_NAME}', parent)
    action.setShortcut('Alt+X')
    action.triggered.connect(parent.plugins[PLUGIN_NAME]['loader'].close_popups)
    plugin_menu.addAction(action)


def callback(parent, _ ,point):

    point_pos = point[0].pos()
    clust_idx = np.argwhere(parent.cluster_data == point_pos.x())[0][0]
    data = {}
    data['t_autocorr'] = parent.shank.plotdata.t_autocorr
    data['autocorr'], clust_no = parent.shank.plotdata.get_autocorr(clust_idx)
    data['t_template'] = parent.shank.plotdata.t_template
    data['template_wf'] = parent.shank.plotdata.get_template_wf(clust_idx)

    parent.plugins[PLUGIN_NAME]['loader'].add_popup(clust_no, data)


class ClusterPopups:
    def __init__(self, parent):
        self.parent = parent
        self.cluster_popups = []
        self.popup_status = True


    def add_popup(self, clust_no, data):
        clust_popup = ClusterPopup(f'Cluster {clust_no}', data=data, parent=self.parent)
        clust_popup.closed.connect(self.popup_closed)
        clust_popup.moved.connect(self.popup_moved)
        self.cluster_popups.append(clust_popup)
        self.parent.activateWindow()

    def minimise_popups(self):
        self.popup_status = not self.popup_status
        if self.popup_status:
            for pop in self.cluster_popups:
                pop.showNormal()
                pop.activateWindow()
        else:
            for pop in self.cluster_popups:
                pop.showMinimized()
        self.parent.activateWindow()

    def close_popups(self):
        for pop in self.cluster_popups:
            pop.blockSignals(True)
            pop.close()
        self.cluster_popups = []

    def popup_closed(self, popup):
        popup_idx = [iP for iP, pop in enumerate(self.cluster_popups) if pop == popup][0]
        self.cluster_popups.pop(popup_idx)

    def popup_moved(self):
        self.parent.activateWindow()

    def reset(self):
        self.close_popups()


class ClusterPopup(PopupWindow):
    def __init__(self, title, data=None, parent=None):
        self.data = data
        super().__init__(title, parent=parent, size=(300, 300), graphics=True)

    def setup(self):

        autocorr_plot = pg.PlotItem()
        autocorr_plot.setXRange(min=np.min(self.data['t_autocorr']), max=np.max(self.data['t_autocorr']))
        autocorr_plot.setYRange(min=0, max=1.05 * np.max(self.data['autocorr']))
        set_axis(autocorr_plot, 'bottom', label='T (ms)')
        set_axis(autocorr_plot, 'left', label='Number of spikes')
        plot = pg.BarGraphItem(x=self.data['t_autocorr'], height=self.data['autocorr'], width=0.24, brush=QtGui.QColor(160, 160, 160))
        autocorr_plot.addItem(plot)


        template_plot = pg.PlotItem()
        plot = pg.PlotCurveItem()
        template_plot.setXRange(min=np.min(self.data['t_template']), max=np.max(self.data['t_template']))
        set_axis(template_plot, 'bottom', label='T (ms)')
        set_axis(template_plot, 'left', label='Amplitude (a.u.)')
        plot.setData(x=self.data['t_template'], y=self.data['template_wf'], pen=pg.mkPen(color='k', style=QtCore.Qt.SolidLine, width=2))
        template_plot.addItem(plot)

        self.popup_widget.addItem(autocorr_plot, 0, 0)
        self.popup_widget.addItem(template_plot, 1, 0)

