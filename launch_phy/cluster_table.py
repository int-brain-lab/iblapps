import datajoint as dj
from ibl_pipeline import reference

schema = dj.schema('group_shared_ephys')


@schema
class ClusterLabel(dj.Imported):
    definition = """
    cluster_uuid: uuid              # uuid of cluster
    -> reference.LabMember          #user name
    ---
    label_time: datetime            # date on which labelling was done
    cluster_label=null: enum('good', 'mua', 'noise') # user assigned label
    """
