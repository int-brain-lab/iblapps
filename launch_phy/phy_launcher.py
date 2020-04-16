import glob
import os
import numpy as np

from phy.apps.template import TemplateController, template_gui
from phy.gui.qt import create_app, run_app
from oneibl.one import ONE


def launch_phy(eid, probe_name, one=None):
    """
    Launch phy given an eid and probe name.

    TODO calculate metrics and save as .tsvs to include in GUI when launching?
    """

    # This is a first draft, no error handling and a draft dataset list.

    # Load data from probe #
    # -------------------- #

    if one is None:
        one = ONE()

    dtypes = [
        'spikes.times',
        'spikes.clusters',
        'spikes.amps',
        'spikes.templates',
        'spikes.samples',
        'templates.waveforms',
        'templates.waveformsChannels',
        'clusters.uuids',
        'clusters.metrics',
        'clusters.waveforms',
        'clusters.waveformsChannels',
        'clusters.depths',
        'clusters.amps',
        'clusters.channels',
        'channels.probes',
        'channels.rawInd',
        'channels.localCoordinates',
        # 'ephysData.raw.ap'
        # '_phy_spikes_subset.waveforms'
        # '_phy_spikes_subset.spikes'
        # '_phy_spikes_subset.channels'
    ]
    _ = one.load(eid, dataset_types=dtypes, download_only=True)
    ses_path = one.path_from_eid(eid)
    alf_probe_dir = os.path.join(ses_path, 'alf', probe_name)
    ephys_file_dir = os.path.join(ses_path, 'raw_ephys_data', probe_name)
    raw_files = glob.glob(os.path.join(ephys_file_dir, '*ap.*bin'))
    raw_file = [raw_files[0]] if raw_files else None

    # TODO download ephys meta-data, and extract TemplateController input arg params

    # Launch phy #
    # -------------------- #
    create_app()
    controller = TemplateController(dat_path=raw_file, dir_path=alf_probe_dir, dtype=np.int16,
                                    n_channels_dat=384, sample_rate=3e4)
    gui = controller.create_gui()
    gui.show()
    run_app()
    gui.close()
    controller.model.close()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('eid', nargs=1, type=str)
    parser.add_argument('probe_name', nargs=1, type=str)
    args = parser.parse_args()
    launch_phy(args.eid[0], args.probe_name[0])

    # eid = '5cf2b2b7-1a88-40cd-adfc-f4a031ff7412'
    # probe_name = 'probe_right'
    # launch_phy(eid, probe_name)
