import glob
import logging
import os
import pandas as pd

from phy.apps.template import TemplateController, template_gui
from phy.gui.qt import create_app, run_app
from phylib import add_default_handler
from one.api import ONE
from brainbox.metrics.single_units import quick_unit_metrics
from pathlib import Path
from brainbox.io.one import SpikeSortingLoader


def launch_phy(probe_name=None, eid=None, pid=None, subj=None, date=None, sess_no=None, one=None):
    """
    Launch phy given an eid and probe name.
    """

    # This is a first draft, no error handling and a draft dataset list.

    # Load data from probe #
    # -------------------- #

    from one.api import ONE
    from ibllib.atlas import AllenAtlas
    from brainbox.io.one import SpikeSortingLoader
    from ibllib.io import spikeglx
    one = one or ONE(base_url='https://openalyx.internationalbrainlab.org')
    ba = AllenAtlas()

    datasets = [
        'spikes.times',
        'spikes.clusters',
        'spikes.amps',
        'spikes.templates',
        'spikes.samples',
        'spikes.depths',
        'clusters.uuids',
        'clusters.metrics',
        'clusters.waveforms',
        'clusters.waveformsChannels',
        'clusters.depths',
        'clusters.amps',
        'clusters.channels']

    if pid is None:
        ssl = SpikeSortingLoader(eid=eid, pname=probe_name, one=one, atlas=ba)
    else:
        ssl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    ssl.download_spike_sorting(dataset_types=datasets)
    ssl.download_spike_sorting_object('templates')
    ssl.download_spike_sorting_object('spikes_subset')

    alf_dir = ssl.session_path.joinpath(ssl.collection)

    if not alf_dir.joinpath('clusters.metrics.pqt').exists():
        spikes, clusters, channels = ssl.load_spike_sorting()
        ssl.merge_clusters(spikes, clusters, channels, cache_dir=alf_dir)

    raw_file = next(ssl.session_path.joinpath('raw_ephys_folder', ssl.pname).glob('*.ap.*bin'), None)

    if raw_file is not None:
        sr = spikeglx.Reader(raw_file)
        sample_rate = sr.fs
        n_channel_dat = sr.nc - sr.nsync
    else:
        sample_rate = 30000
        n_channel_dat = 384

    # Launch phy #
    # -------------------- #
    add_default_handler('DEBUG', logging.getLogger("phy"))
    add_default_handler('DEBUG', logging.getLogger("phylib"))
    create_app()
    controller = TemplateController(dat_path=raw_file, dir_path=alf_dir, dtype=np.int16,
                                    n_channels_dat=n_channel_dat, sample_rate=sample_rate,
                                    plugins=['IBLMetricsPlugin'],
                                    plugin_dirs=[Path(__file__).resolve().parent / 'plugins'])
    gui = controller.create_gui()
    gui.show()
    run_app()
    gui.close()
    controller.model.close()


if __name__ == '__main__':
    """
    `python int-brain-lab\iblapps\launch_phy\phy_launcher.py -e a3df91c8-52a6-4afa-957b-3479a7d0897c -p probe00`
    `python int-brain-lab\iblapps\launch_phy\phy_launcher.py -pid c07d13ed-e387-4457-8e33-1d16aed3fd92`
    """
    from argparse import ArgumentParser
    import numpy as np

    parser = ArgumentParser()
    parser.add_argument('-s', '--subject', default=False, required=False,
                        help='Subject Name')
    parser.add_argument('-d', '--date', default=False, required=False,
                        help='Date of session YYYY-MM-DD')
    parser.add_argument('-n', '--session_no', default=1, required=False,
                        help='Session Number', type=int)
    parser.add_argument('-e', '--eid', default=False, required=False,
                        help='Session eid')
    parser.add_argument('-p', '--probe_label', default=False, required=False,
                        help='Probe Label')
    parser.add_argument('-pid', '--pid', default=False, required=False,
                        help='Probe ID')

    args = parser.parse_args()
    if args.eid:
        launch_phy(probe_name=str(args.probe_label), eid=str(args.eid))
    elif args.pid:
        launch_phy(pid=str(args.pid))
    else:
        if not np.all(np.array([args.subject, args.date, args.session_no],
                               dtype=object)):
            print('Must give Subject, Date and Session number')
        else:
            launch_phy(probe_name=str(args.probe_label), subj=str(args.subject),
                       date=str(args.date), sess_no=args.session_no)
    # launch_phy('probe00', subj='KS022',
            # date='2019-12-10', sess_no=1)
