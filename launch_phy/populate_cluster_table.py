from launch_phy import cluster_table
import os
import csv
from datetime import datetime, timedelta
from oneibl.one import ONE
from ibl_pipeline import subject, ephys


def populate_dj_with_phy(probe_label, eid=None, subj=None, date=None,
                         sess_no=None, one=None):

    if one is None:
        one = ONE()

    if eid is None:
        eid = one.search(subject=subj, date=date, number=sess_no)[0]

    # Find the alf path associated with eid
    sess_path = one.path_from_eid(eid)
    alf_path = sess_path.joinpath('alf', probe_label)
    # From path find subject name, session date
    subj = os.path.split(sess_path.parent.parent)[-1]
    session_date = os.path.split(sess_path.parent)[-1]
    probe = int(probe_label[-1])
    user = one._par.ALYX_LOGIN
    # Dates to restrict dj query
    date = datetime.strptime(session_date, "%Y-%m-%d")
    next_date = date + timedelta(days=1)
    current_date = datetime.now().replace(microsecond=0)

    # Read in the manual labelling results output by phy
    cluster_id = np.empty((0, 1), dtype=int)
    cluster_label = np.empty((0, 1))

    with open(alf_path.joinpath('cluster_group.tsv'), newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        for iR, row in enumerate(reader):
            if iR == 0:
                assert(row[0] == 'cluster_id')
                assert(row[1] == 'group')
            else:
                cluster_id = np.append(cluster_id, int(row[0]))
                cluster_label = np.append(cluster_label, row[1])

    # dj table that we will write to
    cluster = cluster_table.ClusterLabel()

    # Find the clusters associated with subj, session and probe
    restrict_dict = dict(subject_nickname=subj, probe_idx=probe)
    sess_dates = np.unique(((ephys.Cluster * subject.Subject) & restrict_dict)
                           .fetch('session_start_time'))
    sess_start_time = sess_dates[np.where((sess_dates >= date) &
                                          (sess_dates < next_date))[0][0]]
    restrict_dict.update(session_start_time=sess_start_time)
    cluster_keys = ((ephys.Cluster * subject.Subject) & restrict_dict).proj()

    # Populate the dj table with the manual labelling results
    for iClust, iLabel in zip(cluster_id, cluster_label):
        prev_clust = (cluster * cluster_keys) & {'user_name': user} \
                     & {'cluster_id': iClust}
        # if the key has already previously been filled with manual labelling
        # result from user, either
        # 1. Skip if label is same as previous label
        # 2. Overwrite if label is different
        if prev_clust:
            prev_label = prev_clust.fetch1('cluster_label')
            if prev_label != iLabel:
                cluster.insert1(dict(*prev_clust.proj(),
                                     label_time=current_date,
                                     cluster_label=iLabel),
                                allow_direct_insert=True, replace=True)
                print('Overwriting label for cluster ' + str(iClust))
        else:
            key = cluster_keys & {'cluster_id': iClust}
            cluster.insert1(dict(*key, user_name=user, label_time=current_date,
                                 cluster_label=iLabel),
                            allow_direct_insert=True)


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


