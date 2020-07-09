from launch_phy import cluster_table
import alf.io
import pandas as pd
from datetime import datetime
from oneibl.one import ONE
from ibllib.misc import print_progress
import sys
from pathlib import Path
import uuid


def populate_dj_with_phy(probe_label, eid=None, subj=None, date=None,
                         sess_no=None, one=None):
    if one is None:
        one = ONE()

    if eid is None:
        eid = one.search(subject=subj, date=date, number=sess_no)[0]

    sess_path = one.path_from_eid(eid)
    alf_path = sess_path.joinpath('alf', probe_label)

    cluster_path = Path(alf_path, 'spikes.clusters.npy')
    template_path = Path(alf_path, 'spikes.templates.npy')

    # Compare spikes.clusters with spikes.templates to find which clusters have been merged
    phy_clusters = np.load(cluster_path)
    id_phy = np.unique(phy_clusters)
    orig_clusters = np.load(template_path)
    id_orig = np.unique(orig_clusters)

    uuid_list = alf.io.load_file_content(alf_path.joinpath('clusters.uuids.csv'))

    # First deal with merged clusters and make sure they have cluster uuids assigned
    # Find the original cluster ids that have been merged into a new cluster
    merged_idx = np.setdiff1d(id_orig, id_phy)

    # See if any clusters have been merged, if not skip to the next bit
    if np.any(merged_idx):
        # Make association between original cluster and new cluster id and save in dict
        merge_list = {}
        for m in merged_idx:
            idx = phy_clusters[np.where(orig_clusters == m)[0][0]]
            if idx in merge_list:
                merge_list[idx].append(m)
            else:
                merge_list[idx] = [m]

        # Create a dataframe from the dict
        merge_clust = pd.DataFrame(columns={'cluster_idx', 'merged_uuid', 'merged_id'})
        for key, value in merge_list.items():
            value_uuid = uuid_list['uuids'][value]
            merge_clust = merge_clust.append({'cluster_idx': key, 'merged_uuid': tuple(value_uuid),
                                              'merged_idx': tuple(value)},
                                             ignore_index=True)

        # Get the dj table that has previously stored merged clusters and store in frame
        merge = cluster_table.MergedClusters()
        merge_dj = pd.DataFrame(columns={'cluster_uuid', 'merged_uuid'})
        merge_dj['cluster_uuid'] = merge.fetch('cluster_uuid').astype(str)
        merge_dj['merged_uuid'] = tuple(map(tuple, merge.fetch('merged_uuid')))

        # Merge the two dataframe to see if any merge combinations already have a cluster_uuid
        merge_comb = pd.merge(merge_dj, merge_clust, on=['merged_uuid'], how='outer')

        # Find the merged clusters that do not have a uuid assigned
        no_uuid = np.where(pd.isnull(merge_comb['cluster_uuid']))[0]

        # Assign new uuid to new merge pairs and add to the merge table
        for nid in no_uuid:
            new_uuid = str(uuid.uuid4())
            merge_comb['cluster_uuid'].iloc[nid] = new_uuid
            merge.insert1(
                dict(cluster_uuid=new_uuid, merged_uuid=merge_comb['merged_uuid'].iloc[nid]),
                allow_direct_insert=True)

        # Add all the uuids to the cluster_uuid frame with index according to cluster id from phy
        for idx, c_uuid in zip(merge_comb['cluster_idx'].values,
                               merge_comb['cluster_uuid'].values):
            uuid_list.loc[idx] = c_uuid

        csv_path = Path(alf_path, 'merge_info.csv')
        merge_comb = merge_comb.reindex(columns=['cluster_idx', 'cluster_uuid', 'merged_idx',
                                                 'merged_uuid'])

        try:
            merge_comb.to_csv(csv_path, index=False)
        except Exception as err:
            print(err)
            print('Close merge_info.csv file and then relaunch script')
            sys.exit(1)
    else:
        print('No merges detected, continuing...')

    # Now populate datajoint with cluster labels
    user = one._par.ALYX_LOGIN
    current_date = datetime.now().replace(microsecond=0)

    try:
        cluster_group = alf.io.load_file_content(alf_path.joinpath('cluster_group.tsv'))
    except Exception as err:
        print(err)
        print('Could not find cluster group file output from phy')
        sys.exit(1)

    try:
        cluster_notes = alf.io.load_file_content(alf_path.joinpath('cluster_notes.tsv'))
        cluster_info = pd.merge(cluster_group, cluster_notes, on=['cluster_id'], how='outer')
    except Exception as err:
        cluster_info = cluster_group
        cluster_info['notes'] = None

    cluster_info = cluster_info.where(cluster_info.notnull(), None)
    cluster_info['cluster_uuid'] = uuid_list['uuids'][cluster_info['cluster_id']].values

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
    cluster_note = cluster_info['notes'][idx_new].values

    if idx_new.size != 0:
        print('Populating dj with ' + str(idx_new.size) + ' new labels')
    else:
        print('No new labels to add')
    for iIter, (iClust, iLabel, iNote) in enumerate(
            zip(cluster_uuid, cluster_label, cluster_note)):
        cluster.insert1(dict(cluster_uuid=iClust, user_name=user,
                             label_time=current_date,
                             cluster_label=iLabel,
                             cluster_note=iNote),
                        allow_direct_insert=True)
        print_progress(iIter, cluster_uuid.size, '', '')

    # Next look through clusters already on datajoint and check if any labels have
    # been changed
    comp_clust = pd.merge(cluster_info, dj_clust, on='cluster_uuid')
    idx_change = np.where(comp_clust['group'] != comp_clust['cluster_label'])[0]

    cluster_uuid = comp_clust['cluster_uuid'][idx_change].values
    cluster_label = comp_clust['group'][idx_change].values
    cluster_note = comp_clust['notes'][idx_change].values

    # Populate table
    if idx_change.size != 0:
        print('Replacing label of ' + str(idx_change.size) + ' clusters')
    else:
        print('No labels to change')
    for iIter, (iClust, iLabel, iNote) in enumerate(
            zip(cluster_uuid, cluster_label, cluster_note)):
        prev_clust = cluster & {'user_name': user} & {'cluster_uuid': iClust}
        cluster.insert1(dict(*prev_clust.proj(),
                             label_time=current_date,
                             cluster_label=iLabel,
                             cluster_note=iNote),
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

    #populate_dj_with_phy('probe00', subj='KS022', date='2019-12-10', sess_no=1)