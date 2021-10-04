import matplotlib.pyplot as plt
from pathlib import Path
import glob


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
