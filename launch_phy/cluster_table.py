import datajoint as dj
from ibl_pipeline import ephys, reference

schema = dj.schema('group_shared_testing')

@schema
class ClusterLabel(dj.Imported):
    definition = """
    -> ephys.Cluster
    -> reference.LabMember          #user name
    ---
    label_time: datetime            # date on which labelling was done
    cluster_label=null: enum('good', 'mua', 'noise') # user assigned label
    """