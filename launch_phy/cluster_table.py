import datajoint as dj
from ibl_pipeline import reference

schema = dj.schema('group_shared_ephys')
dj.config["enable_python_native_blobs"] = True


@schema
class ClusterLabel(dj.Imported):
    definition = """ 
    cluster_uuid: uuid              # uuid of cluster 
    -> reference.LabMember          #user name 
    --- 
    label_time: datetime            # date on which labelling was done 
    cluster_label=null: varchar(255)     # user assigned label  
    cluster_note=null: varchar(255)      # user note about cluster
    """


@schema
class MergedClusters(dj.Imported):
    definition = """ 
    cluster_uuid: uuid              # uuid of merged cluster 
    --- 
    merged_uuid: longblob            # array of uuids of original clusters that form merged cluster 
    """
