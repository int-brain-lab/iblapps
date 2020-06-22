## Launch phy with an eid and probe_name

To launch phy, first follow [these instructions](https://github.com//int-brain-lab/iblenv) for setting up a unified IBL conda environment.

Then in your terminal, make sure you are on the 'develop' branch of this repository, activate your unified conda environment, and run either:

`python <path/to/phy_launcher.py> -s subject -d date -n session_no -p probe_name`
e.g `python int-brain-lab\iblapps\launch_phy\phy_launcher.py -s KS022 -d 2019-12-10 -n 1 -p probe00`

or:

`python <path/to/phy_launcher.py> -s subject -e eid -p probe_name`
e.g. `python int-brain-lab\iblapps\launch_phy\phy_launcher.py -e a3df91c8-52a6-4afa-957b-3479a7d0897c -p probe00`

## Compute quality metrics

To compute quality metrics, and then display them in phy, simply add -m True:

`python <path/to/phy_launcher.py> -s subject -e eid -p probe_name -m True`
e.g. `python int-brain-lab\iblapps\launch_phy\phy_launcher.py -e a3df91c8-52a6-4afa-957b-3479a7d0897c -p probe00 -m True`

If quality metrics have already been computed, then no need to add this extra argument. The metrics will be displayed when you launch phy normally (as above).
Description of metrics can be found here: https://docs.google.com/document/d/1ba_krsfm4epiAd0zbQ8hdvDN908P9VZOpTxkkH3P_ZY/edit#

## Upload manual labels to datajoint

Once you have manually labelled clusters in phy these can be uploaded and stored in a datajoint table by running either:

`python <path/to/populate_cluster_table.py> -s subject -d date -n session_no -p probe_name`
e.g `python int-brain-lab\iblapps\launch_phy\populate_cluster_table.py -s KS022 -d 2019-12-10 -n 1 -p probe00`

or:

`python <path/to/populate_cluster_table.py> -s subject -e eid -p probe_name`
e.g. `python int-brain-lab\iblapps\launch_phy\populate_cluster_table.py -e a3df91c8-52a6-4afa-957b-3479a7d0897c -p probe00`


The result of merging clusters will be saved on google drive (for the time being). The files that 
need to be uploaded are,
- spikes.clusters
- merge_info

These should be uploaded [here](https://drive.google.com/drive/u/1/folders/1_KDshAIblNiFNDQD37ZtoFks8b4VDWgg), 
within a folder using the following naming convention,

subject_date_session_probe_user

e.g KS022_2019-12-10_001_probe00_mayo