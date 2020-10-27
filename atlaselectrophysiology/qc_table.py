import datajoint as dj

schema = dj.schema('group_shared_ephys')


@schema
class EphysQC(dj.Imported):
    definition = """
    probe_insertion_uuid: uuid         # probe insertion uuid
    -> reference.LabMember             # user name
    ---
    alignment_qc=null: enum('high', 'medium', 'low')   # confidence in alignment
    ephys_qc=null: enum('pass', 'critical', 'warning') # quality of ephys
    ephys_qc_description=null: varchar(255)  #Description for ephys_qc
    """
