import matplotlib.pyplot as plt
from pathlib import Path
import glob


def make_overview_plot(folder, sess_info, save_folder=None):

    image_folder = folder
    image_info = sess_info
    if not save_folder:
        save_folder = image_folder

    def load_image(image_name, ax, equal_aspect=True):
        with image_name as ifile:
            image = plt.imread(ifile)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_axis_off()
        if equal_aspect: ax.set_aspect('equal')
        ax.imshow(image, aspect='equal' if equal_aspect else None)
        return image

    fig = plt.figure(figsize=(8, 6), dpi=500)
    gs = fig.add_gridspec(4, 21, wspace=-0.1, hspace=0.05, top=0.88, bottom=0.05, left=0.01, right=0.99)
    plt.figtext(0.02, 0.9, '/'.join(folder.parts[-3:]), fontsize=5)

    # --- Image view ---
    img_files = glob.glob(str(image_folder.joinpath(image_info + 'img_*.png')))
    img_row_order = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    img_column_order = [0, 3, 0, 3, 6, 9, 12, 0, 3, 6, 9, 12, 0, 3, 6, 9, 12]
    img_lfp_bands = [f'{epoch}_{band}' for epoch in ['spont', 'opto', 'diff'] for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']]
    img_keywords = [
        'Cluster Amp vs Depth vs FR',
        'Spike Correlation',
        *img_lfp_bands
    ]
    img_files_sort = [next((file for file in img_files if keyword in file), None) for keyword in img_keywords]

    for iF, file in enumerate(img_files_sort):
        if file is None:
            continue
        ax = fig.add_subplot(gs[img_row_order[iF], img_column_order[iF]:img_column_order[iF] + 3])
        load_image(Path(file), ax, equal_aspect=True)
        ax.set_title(img_keywords[iF], fontsize=2, pad=0.5)

    # --- Probe and line view ---
    probe_line_row_order = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    probe_line_column_order = [6, 7, 8, 9, 10, 11, 12, 13, 14]
    probe_line_keywords = [
        'line_Amplitude',
        'line_Firing Rate',
        'RMS AP',
        'RMS LFP',
        '0 - 4 Hz',
        '4 - 10 Hz',
        '10 - 30 Hz',
        '30 - 80 Hz',
        '80 - 200 Hz'
    ]
    probe_files = glob.glob(str(image_folder.joinpath(image_info + 'probe_*.png')))
    line_files = glob.glob(str(image_folder.joinpath(image_info + 'line_*.png')))
    probe_line_files = probe_files + line_files
    probe_line_files_sort = [next((file for file in probe_line_files if keyword in file), None) for keyword in probe_line_keywords]

    for iF, file in enumerate(probe_line_files_sort):
        if file is None:
            continue
        ax = fig.add_subplot(gs[probe_line_row_order[iF], probe_line_column_order[iF]])
        load_image(Path(file), ax, equal_aspect=False)
        ax.set_title(probe_line_keywords[iF], fontsize=2, pad=0.5)
        
    # --- Slice view ---
    slice_files = glob.glob(str(image_folder.joinpath(image_info + 'slice_*.png')))
    slice_row_order = [0, 1, 2]
    slice_column_order = [18, 18, 18]
    slice_keywords = [
        'slice_Annotation',  # To explicitly exclude "zoom"
        'slice_CCF',
        'slice_histology_registration',
    ]
    slice_files_sort = [next((file for file in slice_files if keyword in file), None) for keyword in slice_keywords]

    for iF, file in enumerate(slice_files_sort):
        if file is None:
            continue
        ax = fig.add_subplot(gs[slice_row_order[iF],
                                slice_column_order[iF]:slice_column_order[iF] + 3])
        load_image(Path(file), ax, equal_aspect=True)

    slice_files_zoom = glob.glob(str(image_folder.joinpath(image_info + 'slice_zoom*.png')))
    slice_row_order = [0, 1, 2]
    slice_column_order = [20, 20, 20]
    slice_keywords = [
        'slice_zoom_Annotation',
        'slice_zoom_CCF',
        'slice_zoom_histology_registration',
    ]
    slice_files_sort = [next((file for file in slice_files_zoom if keyword in file), None) for keyword in slice_keywords]

    for iF, file in enumerate(slice_files_sort):
        if file is None:
            continue
        ax = fig.add_subplot(gs[slice_row_order[iF], slice_column_order[iF]])
        load_image(Path(file), ax, equal_aspect=True)

    # --- Histology view ---
    hist_file = glob.glob(str(image_folder.joinpath(image_info + 'hist*.png')))[0]
    ax = fig.add_subplot(gs[0:3, 15:18])
    load_image(Path(hist_file), ax, equal_aspect=False)

    ax.text(0.5, 0, image_info[:-1], va="center", ha="center", transform=ax.transAxes)

    fig.savefig(save_folder.joinpath(image_info + "overview.png"),
                pad_inches=0.1, dpi=fig.dpi, bbox_inches='tight')
    plt.show()
