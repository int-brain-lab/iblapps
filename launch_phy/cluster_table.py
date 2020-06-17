import datajoint as dj
from ibl_pipeline import reference

schema = dj.schema('group_shared_testing')
dj.config["enable_python_native_blobs"] = True


@schema
class ClusterLabel(dj.Imported):
    definition = """
    cluster_uuid: uuid              # uuid of cluster
    -> reference.LabMember          #user name
    ---
    label_time: datetime            # date on which labelling was done
    cluster_label=null: enum('good', 'mua', 'noise', 'unsorted') # user assigned label
    """


@schema
class MergedClusters(dj.Imported):
    definition = """
    cluster_uuid: uuid              # uuid of merged cluster
    ---
    merged_uuid: longblob            # array of uuids of original clusters that form merged cluster
    """