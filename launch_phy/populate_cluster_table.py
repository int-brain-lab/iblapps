from launch_phy import cluster_table
import alf.io
import pandas as pd
from datetime import datetime
from oneibl.one import ONE
from ibllib.misc import print_progress
import sys


def populate_dj_with_phy(probe_label, eid=None, subj=None, date=None,
                         sess_no=None, one=None):

    if one is None:
        one = ONE()

    if eid is None:
        eid = one.search(subject=subj, date=date, number=sess_no)[0]

    # Find the alf path associated with eid
    sess_path = one.path_from_eid(eid)
    alf_path = sess_path.joinpath('alf', probe_label)

    # Find user and current date to add to dj table
    user = one._par.ALYX_LOGIN
    current_date = datetime.now().replace(microsecond=0)

    # Load in output from phy
    uuid = alf.io.load_file_content(alf_path.joinpath('clusters.uuids.csv'))
    try:
        cluster_info = alf.io.load_file_content(alf_path.joinpath('cluster_group.tsv'))
        cluster_info['cluster_uuid'] = uuid['uuids'][cluster_info['cluster_id']].values
    except Exception as err:
        print(err)
        print('Could not find cluster group file output from phy')
        sys.exit(1)

    # dj table that holds data
    cluster = cluster_table.ClusterLabel()

    # Find clusters that have already been labelled by user
    old_clust = cluster & cluster_info & {'user_name': user}

    dj_clust = pd.DataFrame()
    dj_clust['cluster_uuid'] = (old_clust.fetch('cluster_uuid')).astype(str)
    dj_clust['cluster_label'] = old_clust.fetch('cluster_label')

    # First find the new clusters to insert into datajoint
    idx_new = np.where(np.isin(cluster_info['cluster_uuid'],
                               dj_clust['cluster_uuid'], invert=True))[0]
    cluster_uuid = cluster_info['cluster_uuid'][idx_new].values
    cluster_label = cluster_info['group'][idx_new].values

    if idx_new.size != 0:
        print('Populating dj with ' + str(idx_new.size) + ' new labels')
    else:
        print('No new labels to add')
    for iIter, (iClust, iLabel) in enumerate(zip(cluster_uuid, cluster_label)):
        cluster.insert1(dict(cluster_uuid=iClust, user_name=user,
                             label_time=current_date,
                             cluster_label=iLabel),
                        allow_direct_insert=True)
        print_progress(iIter, cluster_uuid.size, '', '')

    # Next look through clusters already on datajoint and check if any labels have
    # been changed
    comp_clust = pd.merge(cluster_info, dj_clust, on='cluster_uuid')
    idx_change = np.where(comp_clust['group'] != comp_clust['cluster_label'])[
        0]

    cluster_uuid = comp_clust['cluster_uuid'][idx_change].values
    cluster_label = comp_clust['group'][idx_change].values

    # Populate table
    if idx_change.size != 0:
        print('Replacing label of ' + str(idx_change.size) + ' clusters')
    else:
        print('No labels to change')
    for iIter, (iClust, iLabel) in enumerate(zip(cluster_uuid, cluster_label)):
        prev_clust = cluster & {'user_name': user} & {'cluster_uuid': iClust}
        cluster.insert1(dict(*prev_clust.proj(),
                             label_time=current_date,
                             cluster_label=iLabel),
                        allow_direct_insert=True, replace=True)
        print_progress(iIter, cluster_uuid.size, '', '')

    print('Upload to datajoint complete')


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
        populate_dj_with_phy(str(args.probe_label), eid=str(args.eid))
    else:
        if not np.all(np.array([args.subject, args.date, args.session_no],
                               dtype=object)):
            print('Must give Subject, Date and Session number')
        else:
            populate_dj_with_phy(str(args.probe_label), subj=str(args.subject),
                                 date=str(args.date), sess_no=args.session_no)

    # populate_dj_with_phy('probe00', subj='KS022', date='2019-12-10', sess_no=1)
