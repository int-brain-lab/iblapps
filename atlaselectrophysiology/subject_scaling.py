import pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets
import numpy as np
import atlaselectrophysiology.ColorBar as cb
from ibllib.pipes.ephys_alignment import EphysAlignment
from atlaselectrophysiology.AdaptedAxisItem import replace_axis
from neuropixel import trace_header

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class ScalingWindow(QtWidgets.QMainWindow):

    def __init__(self, pid, subject, one, ba, size=(1600, 800)):
        super(ScalingWindow, self).__init__()

        self.resize(size[0], size[1])
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout()
        main_widget.setLayout(main_layout)

        self.data = []
        depths = trace_header()['y']
        idx = None

        insertions = one.alyx.rest('insertions', 'list', subject=subject,
                                   django='json__extended_qc__has_key,alignment_resolved')

        for i, ins in enumerate(insertions):
            if ins['id'] == pid:
                idx = i
                continue

            xyz = np.array(ins['json']['xyz_picks']) / 1e6
            traj = one.alyx.rest('trajectories', 'list', probe_insertion=ins['id'],
                                 provenance='Ephys aligned histology track')[0]
            align_key = ins['json']['extended_qc']['alignment_stored']
            feature = traj['json'][align_key][0]
            track = traj['json'][align_key][1]
            resolved = ins['json']['extended_qc']['alignment_resolved']
            ephysalign = EphysAlignment(xyz, depths, track_prev=track,
                                        feature_prev=feature,
                                        brain_atlas=ba, speedy=True)

            region_colour = ephysalign.region_colour
            region, region_label = ephysalign.scale_histology_regions(feature, track)
            scale_region, scale = ephysalign.get_scale_factor(region)
            region_orig, label_orig = ephysalign.scale_histology_regions(ephysalign.track_extent, ephysalign.track_extent)

            data = {'regions_orig': region_orig,
                    'labels_orig': label_orig,
                    'regions': region,
                    'labels': region_label,
                    'colors': region_colour,
                    'scaled_regions': scale_region,
                    'scale': scale,
                    'resolved': resolved
                    }

            self.data.append(data)

        if idx is not None:
            insertions.pop(idx)

        info_widget = QtWidgets.QWidget()
        info_layout = QtWidgets.QHBoxLayout()
        info_widget.setLayout(info_layout)

        for ins in insertions:
            info_label = QtWidgets.QLabel()
            info_label.setText(f'{ins["session_info"]["subject"]}/{ins["session_info"]["start_time"][:10]}'
                               f'/00{ins["session_info"]["number"]} \n {ins["name"]}')
            info_layout.addWidget(info_label)

        plot_widget = QtWidgets.QWidget()
        plot_layout = QtWidgets.QHBoxLayout()
        plot_widget.setLayout(plot_layout)

        self.plots_hist = []
        self.plots_hist_ref = []
        self.plots_scale = []
        self.plots_cbar = []

        for _ in insertions:
            fig_area = pg.GraphicsLayoutWidget()
            fig_area.setMouseTracking(True)
            fig_area.scene().sigMouseHover.connect(self.on_mouse_hover)
            fig_layout = pg.GraphicsLayout()
            fig_area.addItem(fig_layout)

            fig_hist = pg.PlotItem()
            fig_hist.setContentsMargins(0, 0, 0, 0)
            fig_hist.setMouseEnabled(x=False)
            self.set_axis(fig_hist, 'bottom', pen='w')

            replace_axis(fig_hist)
            ax_hist = self.set_axis(fig_hist, 'left', pen='k')
            ax_hist.setWidth(30)
            self.plots_hist.append(fig_hist)

            fig_scale = pg.PlotItem()
            fig_scale.setMaximumWidth(50)
            fig_scale.setMouseEnabled(x=False)
            self.set_axis(fig_scale, 'bottom', pen='w')
            self.set_axis(fig_scale, 'left', show=False)
            fig_scale.setYLink(fig_hist)
            self.plots_scale.append(fig_scale)

            # Figure that will show scale factor of histology boundaries
            fig_scale_cb = pg.PlotItem()
            fig_scale_cb.setMouseEnabled(x=False, y=False)
            fig_scale_cb.setMaximumHeight(70)
            self.set_axis(fig_scale_cb, 'bottom', show=False)
            self.set_axis(fig_scale_cb, 'left', show=False)
            self.set_axis(fig_scale_cb, 'top', pen='w')
            self.set_axis(fig_scale_cb, 'right', show=False)
            self.plots_cbar.append(fig_scale_cb)

            # Histology figure that will remain at initial state for reference
            fig_hist_ref = pg.PlotItem()
            fig_hist_ref.setMouseEnabled(x=False)
            self.set_axis(fig_hist_ref, 'bottom', pen='w')
            self.set_axis(fig_hist_ref, 'left', show=False)
            replace_axis(fig_hist_ref, orientation='right', pos=(2, 2))
            ax_hist_ref = self.set_axis(fig_hist_ref, 'right', pen=None)
            ax_hist_ref.setWidth(0)
            self.plots_hist_ref.append(fig_hist_ref)

            fig_layout.addItem(fig_scale_cb, 0, 0, 1, 3)
            fig_layout.addItem(fig_hist, 1, 0)
            fig_layout.addItem(fig_scale, 1, 1)
            fig_layout.addItem(fig_hist_ref, 1, 2)
            fig_layout.layout.setColumnStretchFactor(0, 4)
            fig_layout.layout.setColumnStretchFactor(1, 1)
            fig_layout.layout.setColumnStretchFactor(2, 4)
            fig_layout.layout.setRowStretchFactor(0, 1)
            fig_layout.layout.setRowStretchFactor(1, 10)

            plot_layout.addWidget(fig_area)

        main_layout.addWidget(info_widget)
        main_layout.addWidget(plot_widget)
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 20)
        self.plot_scale_factor()
        self.plot_histology()
        self.plot_histology(ref=True)

        self.show()

    def set_axis(self, fig, ax, show=True, label=None, pen='k', ticks=True):

        if not label:
            label = ''
        if type(fig) == pg.PlotItem:
            axis = fig.getAxis(ax)
        else:
            axis = fig.plotItem.getAxis(ax)
        if show:
            axis.show()
            axis.setPen(pen)
            axis.setTextPen(pen)
            axis.setLabel(label)
            if not ticks:
                axis.setTicks([[(0, ''), (0.5, ''), (1, '')]])
        else:
            axis.hide()

        return axis

    def plot_scale_factor(self):
        """
        Plots the scale factor applied to brain regions along probe track, displayed
        alongside histology figure
        """
        self.scale_regions = []
        for iD, data in enumerate(self.data):
            fig_scale = self.plots_scale[iD]
            fig_scale_cb = self.plots_cbar[iD]

            fig_scale.clear()
            fig_scale_cb.clear()

            scale_regions = np.empty((0, 1))
            scale_factor = data['scale'] - 0.5
            color_bar = cb.ColorBar('seismic')
            cbar = color_bar.makeColourBar(20, 5, fig_scale_cb, min=0.5, max=1.5,
                                           label='Scale Factor')
            colours = color_bar.map.mapToQColor(scale_factor)

            for ir, reg in enumerate(data['scaled_regions']):
                region = pg.LinearRegionItem(values=(reg[0], reg[1]),
                                             orientation=pg.LinearRegionItem.Horizontal,
                                             brush=colours[ir], movable=False)
                bound = pg.InfiniteLine(pos=reg[0], angle=0, pen=colours[ir])

                fig_scale.addItem(region)
                fig_scale.addItem(bound)
                scale_regions = np.vstack([scale_regions, region])

            bound = pg.InfiniteLine(pos=data['scaled_regions'][-1][1], angle=0,
                                    pen=colours[-1])

            fig_scale.addItem(bound)

            fig_scale.setYRange(min=0, max=3840)
            self.set_axis(fig_scale, 'bottom', pen='w', label='blank')
            fig_scale_cb.addItem(cbar)
            self.scale_regions.append(scale_regions)

    def plot_histology(self, ref=False):

        for iD, data in enumerate(self.data):

            if ref:
                fig = self.plots_hist_ref[iD]
                axis = 'right'
                labels = data['labels_orig']
                regions = data['regions_orig']
            else:
                fig = self.plots_hist[iD]
                axis = 'left'
                labels = data['labels']
                regions = data['regions']

            fig.clear()

            axis = fig.getAxis(axis)
            axis.setTicks([labels])
            axis.setZValue(10)
            axis.setPen('k')

            # Plot each histology region
            for ir, (reg, col) in enumerate(zip(regions, data['colors'])):
                colour = QtGui.QColor(*col)

                region = pg.LinearRegionItem(values=(reg[0], reg[1]),
                                             orientation=pg.LinearRegionItem.Horizontal,
                                             brush=colour, movable=False)
                # Add a white line at the boundary between regions
                bound = pg.InfiniteLine(pos=reg[0], angle=0, pen='w')
                fig.addItem(region)
                fig.addItem(bound)

            fig.setYRange(min=0, max=3840)

    def on_mouse_hover(self, items):
        if len(items) > 1:
            scale_plot = False
            for idx, pl in enumerate(self.plots_scale):
                if items[0] == pl:
                    scale_plot = True
                    break

            if scale_plot:
                reg_idx = np.where(self.scale_regions[idx] == items[1])[0]
                if len(reg_idx) > 0:
                    ax = self.plots_cbar[idx].getAxis('top')
                    ax.setLabel('Scale Factor = ' + str(np.around(self.data[idx]['scale'][reg_idx[0]], 2)))
