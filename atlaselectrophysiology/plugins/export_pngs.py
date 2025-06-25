import matplotlib.pyplot as plt
from pathlib import Path
import glob
import atlaselectrophysiology.qt_utils.utils as utils
import pyqtgraph as pg
from qtpy import QtWidgets
import numpy as np


PLUGIN_NAME = 'Export PNGs'

def setup(parent):

    parent.plugins[PLUGIN_NAME] = dict()

    action = QtWidgets.QAction(PLUGIN_NAME, parent)
    action.triggered.connect(lambda: save_plots(parent))
    parent.plugin_options.addAction(action)



def save_plots(self, save_path=None):
    # TODOOOOO improve
    """
    Saves all plots from the GUI into folder
    """
    # make folder to save plots to
    try:
        sess_info = (self.shank.subj + '_' + str(self.shank.date) + '_' +
                     self.shank.probe_label + '_')
        image_path_overview = self.shank.probe_path.joinpath('GUI_plots')
        image_path = image_path_overview.joinpath(sess_info[:-1])
    except Exception as e:
        print(e)
        sess_info = ''
        image_path_overview = self.shank.probe_path.joinpath('GUI_plots')
        image_path = image_path_overview

    if save_path:
        image_path_overview = Path(save_path)

    image_path.mkdir(parents=True, exist_ok=True)
    image_path_overview.mkdir(parents=True, exist_ok=True)

    # Reset all axis, put view back to 1 and remove any reference lines
    self.reset_axis_button_pressed()
    self.set_view(view=1, configure=False)
    self.remove_reference_lines_from_display()

    xlabel_img = self.fig_img.getAxis('bottom').label.toPlainText()
    xlabel_line = self.fig_line.getAxis('bottom').label.toPlainText()

    # First go through all the image plots
    self.fig_data_layout.removeItem(self.fig_probe)
    self.fig_data_layout.removeItem(self.fig_probe_cb)
    self.fig_data_layout.removeItem(self.fig_line)

    width1 = self.fig_data_area.width()
    height1 = self.fig_data_area.height()
    ax_width = self.fig_img.getAxis('left').width()
    ax_height = self.fig_img_cb.getAxis('top').height()

    utils.set_font(self.fig_img, 'left', ptsize=15, width=ax_width + 20)
    utils.set_font(self.fig_img, 'bottom', ptsize=15)
    utils.set_font(self.fig_img_cb, 'top', ptsize=15, height=ax_height + 15)

    self.fig_data_area.resize(700, height1)

    plot = None
    start_plot = self.img_options_group.checkedAction()

    while plot != start_plot:
        utils.set_font(self.fig_img_cb, 'top', ptsize=15, height=ax_height + 15)
        exporter = pg.exporters.ImageExporter(self.fig_data_layout.scene())
        exporter.export(str(image_path.joinpath(sess_info + 'img_' +
                                                self.img_options_group.checkedAction()
                                                .text() + '.png')))
        self.toggle_plots(self.img_options_group)
        self.remove_reference_lines_from_display()
        plot = self.img_options_group.checkedAction()

    utils.set_font(self.fig_img, 'left', ptsize=8, width=ax_width)
    utils.set_font(self.fig_img, 'bottom', ptsize=8)
    utils.set_font(self.fig_img_cb, 'top', ptsize=8, height=ax_height)
    utils.set_axis(self.fig_img, 'bottom', label=xlabel_img)
    self.fig_data_layout.removeItem(self.fig_img)
    self.fig_data_layout.removeItem(self.fig_img_cb)

    # Next go over probe plots
    self.fig_data_layout.addItem(self.fig_probe_cb, 0, 0, 1, 2)
    self.fig_data_layout.addItem(self.fig_probe, 1, 0)
    utils.set_axis(self.fig_probe, 'left', label='Distance from probe tip (uV)')
    self.fig_probe.setFixedWidth(self.fig_probe_width + self.fig_ax_width + 20)
    utils.set_font(self.fig_probe, 'left', ptsize=15, width=ax_width + 20)
    utils.set_font(self.fig_probe_cb, 'top', ptsize=15, height=ax_height + 15)
    self.fig_data_area.resize(250, height1)

    plot = None
    start_plot = self.probe_options_group.checkedAction()

    while plot != start_plot:
        utils.set_font(self.fig_probe_cb, 'top', ptsize=15, height=ax_height + 15)
        exporter = pg.exporters.ImageExporter(self.fig_data_layout.scene())
        exporter.export(str(image_path.joinpath(sess_info + 'probe_' +
                                                self.probe_options_group.checkedAction().
                                                text() + '.png')))
        self.toggle_plots(self.probe_options_group)
        plot = self.probe_options_group.checkedAction()

    self.fig_probe.setFixedWidth(self.fig_probe_width + self.fig_ax_width)
    utils.set_font(self.fig_probe, 'left', ptsize=8, width=ax_width)
    utils.set_font(self.fig_probe_cb, 'top', ptsize=8, height=ax_height)
    utils.set_axis(self.fig_probe, 'bottom', pen='w', label='blank')
    self.fig_data_layout.removeItem(self.fig_probe)
    self.fig_data_layout.removeItem(self.fig_probe_cb)

    # Next go through the line plots
    self.fig_data_layout.addItem(self.fig_probe_cb, 0, 0, 1, 2)
    self.fig_probe_cb.clear()
    text = self.fig_probe_cb.getAxis('top').label.toPlainText()
    utils.set_axis(self.fig_probe_cb, 'top', pen='w')
    self.fig_data_layout.addItem(self.fig_line, 1, 0)

    utils.set_axis(self.fig_line, 'left', label='Distance from probe tip (um)')
    utils.set_font(self.fig_line, 'left', ptsize=15, width=ax_width + 20)
    utils.set_font(self.fig_line, 'bottom', ptsize=15)
    self.fig_data_area.resize(200, height1)

    plot = None
    start_plot = self.line_options_group.checkedAction()
    while plot != start_plot:
        exporter = pg.exporters.ImageExporter(self.fig_data_layout.scene())
        exporter.export(str(image_path.joinpath(sess_info + 'line_' +
                                                self.line_options_group.checkedAction().
                                                text() + '.png')))
        self.toggle_plots(self.line_options_group)
        plot = self.line_options_group.checkedAction()

    [self.fig_probe_cb.addItem(cbar) for cbar in self.probe_cbars]
    utils.set_axis(self.fig_probe_cb, 'top', pen='k', label=text)
    utils.set_font(self.fig_line, 'left', ptsize=8, width=ax_width)
    utils.set_font(self.fig_line, 'bottom', ptsize=8)
    utils.set_axis(self.fig_line, 'bottom', label=xlabel_line)
    self.fig_data_layout.removeItem(self.fig_line)
    self.fig_data_layout.removeItem(self.fig_probe_cb)
    self.fig_data_area.resize(width1, height1)
    self.fig_data_layout.addItem(self.fig_probe_cb, 0, 0, 1, 2)
    self.fig_data_layout.addItem(self.fig_img_cb, 0, 2)
    self.fig_data_layout.addItem(self.fig_probe, 1, 0)
    self.fig_data_layout.addItem(self.fig_line, 1, 1)
    self.fig_data_layout.addItem(self.fig_img, 1, 2)

    self.set_view(view=1, configure=False)

    # Save slice images
    plot = None
    start_plot = self.slice_options_group.checkedAction()
    while plot != start_plot:
        self.toggle_channels()
        self.traj_line.setData(x=self.xyz_channels[:, 0], y=self.xyz_channels[:, 2],
                               pen=self.rpen_dot)
        self.fig_slice.addItem(self.traj_line)
        slice_name = self.slice_options_group.checkedAction().text()
        exporter = pg.exporters.ImageExporter(self.fig_slice)
        exporter.export(str(image_path.joinpath(sess_info + 'slice_' + slice_name + '.png')))
        self.toggle_plots(self.slice_options_group)
        plot = self.slice_options_group.checkedAction()

    plot = None
    start_plot = self.slice_options_group.checkedAction()
    while plot != start_plot:
        self.toggle_channels()
        self.traj_line.setData(x=self.xyz_channels[:, 0], y=self.xyz_channels[:, 2],
                               pen=self.rpen_dot)
        self.fig_slice.addItem(self.traj_line)
        slice_name = self.slice_options_group.checkedAction().text()
        self.fig_slice.setXRange(min=np.min(self.xyz_channels[:, 0]) - 200 / 1e6,
                                 max=np.max(self.xyz_channels[:, 0]) + 200 / 1e6)
        self.fig_slice.setYRange(min=np.min(self.xyz_channels[:, 2]) - 500 / 1e6,
                                 max=np.max(self.xyz_channels[:, 2]) + 500 / 1e6)
        self.fig_slice.resize(50, self.slice_height)
        exporter = pg.exporters.ImageExporter(self.fig_slice)
        exporter.export(
            str(image_path.joinpath(sess_info + 'slice_zoom_' + slice_name + '.png')))
        self.fig_slice.resize(self.slice_width, self.slice_height)
        self.fig_slice.setRange(rect=self.slice_rect)
        self.toggle_plots(self.slice_options_group)
        plot = self.slice_options_group.checkedAction()

    # Save the brain regions image
    utils.set_axis(self.fig_hist_extra_yaxis, 'left')
    # Add labels to show which ones are aligned
    utils.set_axis(self.fig_hist, 'bottom', label='aligned')
    utils.set_font(self.fig_hist, 'bottom', ptsize=12)
    utils.set_axis(self.fig_hist_ref, 'bottom', label='original')
    utils.set_font(self.fig_hist_ref, 'bottom', ptsize=12)
    exporter = pg.exporters.ImageExporter(self.fig_hist_layout.scene())
    exporter.export(str(image_path.joinpath(sess_info + 'hist.png')))
    utils.set_axis(self.fig_hist_extra_yaxis, 'left', pen=None)
    utils.set_font(self.fig_hist, 'bottom', ptsize=8)
    utils.set_axis(self.fig_hist, 'bottom', pen='w', label='blank')
    utils.set_font(self.fig_hist_ref, 'bottom', ptsize=8)
    utils.set_axis(self.fig_hist_ref, 'bottom', pen='w', label='blank')

    make_overview_plot(image_path, sess_info, save_folder=image_path_overview)

    self.add_reference_lines_to_display()


def make_overview_plot(folder, sess_info, save_folder=None):

    image_folder = folder
    image_info = sess_info
    if not save_folder:
        save_folder = image_folder

    def load_image(image_name, ax):
        with image_name as ifile:
            image = plt.imread(ifile)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_axis_off()
        ax.set_aspect('equal')
        ax.imshow(image)
        return image

    fig = plt.figure(constrained_layout=True, figsize=(18, 9))
    gs = fig.add_gridspec(3, 18)
    gs.update(wspace=0.025, hspace=0.05)

    ignore_img_plots = ['leftGabor', 'rightGabor', 'noiseOn', 'valveOn', 'toneOn']
    img_row_order = [0, 0, 0, 0, 0, 0, 1, 1, 1]
    img_column_order = [0, 3, 6, 9, 12, 15, 0, 3, 6]
    img_idx = [0, 5, 4, 6, 7, 8, 1, 2, 3]
    img_files = glob.glob(str(image_folder.joinpath(image_info + 'img_*.png')))
    img_files = [img for img in img_files if not any([ig in img for ig in ignore_img_plots])]
    img_files_sort = [img_files[idx] for idx in img_idx]

    for iF, file in enumerate(img_files_sort):
        ax = fig.add_subplot(gs[img_row_order[iF], img_column_order[iF]:img_column_order[iF] + 3])
        load_image(Path(file), ax)

    ignore_probe_plots = ['RF Map']
    probe_row_order = [1, 1, 1, 1, 1, 1, 2, 2, 2]
    probe_column_order = [9, 10, 11, 12, 13, 14, 12, 13, 14]
    probe_idx = [0, 3, 1, 2, 4, 5, 6]
    probe_files = glob.glob(str(image_folder.joinpath(image_info + 'probe_*.png')))
    probe_files = [probe for probe in probe_files if not any([pr in probe for pr in
                                                              ignore_probe_plots])]
    probe_files_sort = [probe_files[idx] for idx in probe_idx]
    line_files = glob.glob(str(image_folder.joinpath(image_info + 'line_*.png')))

    for iF, file in enumerate(probe_files_sort + line_files):
        ax = fig.add_subplot(gs[probe_row_order[iF], probe_column_order[iF]])
        load_image(Path(file), ax)

    slice_files = glob.glob(str(image_folder.joinpath(image_info + 'slice_*.png')))
    slice_row_order = [2, 2, 2, 2]
    slice_idx = [0, 1, 2, 3]
    slice_column_order = [0, 3, 6, 9]
    slice_files_sort = [slice_files[idx] for idx in slice_idx]

    for iF, file in enumerate(slice_files_sort):
        ax = fig.add_subplot(gs[slice_row_order[iF],
                                slice_column_order[iF]:slice_column_order[iF] + 3])
        load_image(Path(file), ax)

    slice_files = glob.glob(str(image_folder.joinpath(image_info + 'slice_zoom*.png')))
    slice_row_order = [2, 2, 2, 2]
    slice_idx = [0, 1, 2, 3]
    slice_column_order = [2, 5, 8, 11]
    slice_files_sort = [slice_files[idx] for idx in slice_idx]

    for iF, file in enumerate(slice_files_sort):
        ax = fig.add_subplot(gs[slice_row_order[iF], slice_column_order[iF]])
        load_image(Path(file), ax)

    hist_files = glob.glob(str(image_folder.joinpath(image_info + 'hist*.png')))
    for iF, file in enumerate(hist_files):
        ax = fig.add_subplot(gs[1:3, 15:18])
        load_image(Path(file), ax)

    ax.text(0.5, 0, image_info[:-1], va="center", ha="center", transform=ax.transAxes)
    plt.savefig(save_folder.joinpath(image_info + "overview.png"),
                bbox_inches='tight', pad_inches=0)
    # plt.close()
    # plt.show()
