import glob
import logging
import os

from phy.apps.template import TemplateController, template_gui
from phy.gui.qt import create_app, run_app
from phylib import add_default_handler
from one.api import ONE
from pathlib import Path


def launch_phy(probe_name, eid=None, subj=None, date=None, sess_no=None, one=None):
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
        'spikes.times.npy',
        'spikes.clusters.npy',
        'spikes.amps.npy',
        'spikes.templates.npy',
        'spikes.samples.npy',
        'spikes.depths.npy',
        'templates.waveforms.npy',
        'templates.waveformsChannels.npy',
        'clusters.uuids.csv',
        'clusters.metrics.pqt',
        'clusters.waveforms.npy',
        'clusters.waveformsChannels.npy',
        'clusters.depths.npy',
        'clusters.amps.npy',
        'clusters.channels.npy',
        'channels.rawInd.npy',
        'channels.localCoordinates.npy',
        # 'ephysData.raw.ap'
        '_phy_spikes_subset.waveforms.npy',
        '_phy_spikes_subset.spikes.npy',
        '_phy_spikes_subset.channels.npy'
    ]

    collection = [f'alf/{probe_name}'] * len(dtypes)

    if eid is None:
        eid = one.search(subject=subj, date=date, number=sess_no)[0]

    _ = one.load_datasets(eid, datasets=dtypes, collections=collection, download_only=True,
                          assert_present=False)
    ses_path = one.eid2path(eid)
    alf_probe_dir = os.path.join(ses_path, 'alf', probe_name)
    ephys_file_dir = os.path.join(ses_path, 'raw_ephys_data', probe_name)
    raw_files = glob.glob(os.path.join(ephys_file_dir, '*ap.*bin'))
    raw_file = [raw_files[0]] if raw_files else None



    # Launch phy #
    # -------------------- #
    add_default_handler('DEBUG', logging.getLogger("phy"))
    add_default_handler('DEBUG', logging.getLogger("phylib"))
    create_app()
    controller = TemplateController(dat_path=raw_file, dir_path=alf_probe_dir, dtype=np.int16,
                                    n_channels_dat=384, sample_rate=3e4,
                                    plugins=['IBLMetricsPlugin'],
                                    plugin_dirs=[Path(__file__).resolve().parent / 'plugins'])
    gui = controller.create_gui()
    gui.show()
    run_app()
    gui.close()
    controller.model.close()


if __name__ == '__main__':
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
    parser.add_argument('-p', '--probe_label', default=False, required=True,
                        help='Probe Label')

    args = parser.parse_args()

    if args.eid:
        launch_phy(str(args.probe_label), eid=str(args.eid))
    else:
        if not np.all(np.array([args.subject, args.date, args.session_no],
                               dtype=object)):
            print('Must give Subject, Date and Session number')
        else:
            launch_phy(str(args.probe_label), subj=str(args.subject),
                       date=str(args.date), sess_no=args.session_no)
    # launch_phy('probe00', subj='KS022',
            # date='2019-12-10', sess_no=1)
