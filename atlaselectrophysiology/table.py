@schema
class EphysQC(dj.Imported):
    definition = """ 
    session_uuid: uuid              # session uuid
    probe_id: uuid                  # probe_uuid
    --- 
    histology_qc=null: enum('critical', 'pass')        # whether histology track traced
    alignment_qc=null: enum('high', 'medium', 'low')   # confidence in alignment
    ephys_qc=null: enum('pass', 'critical', 'warning') # quality of ephys  
    ephys_qc_description=null: enum('Noise and artifact', 'Drift', 'Poor neural activity',
                                    'Brain damage')    #Description for ephys_qc

    """