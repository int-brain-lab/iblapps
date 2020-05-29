## Launch phy with an eid and probe_name

To launch phy, first follow [these instructions](https://github.com//int-brain-lab/iblenv) for setting up a unified IBL conda environment.

Then in your terminal, make sure you are on the 'develop' branch of this repository, activate your unified conda environment, and run either:

`python <path/to/phy_launcher.py> -s subject -d date -n session_no -p probe_name`
e.g `python int-brain-lab\iblapps\launch_phy\phy_launcher.py -s KS022 -d 2019-12-10 -n 1 -p probe00`

or:

`python <path/to/phy_launcher.py> -s subject -e eid -p probe_name`
e.g. `python int-brain-lab\iblapps\launch_phy\phy_launcher.py -e a3df91c8-52a6-4afa-957b-3479a7d0897c -p probe00`


## Upload manual labels to datajoint

Once you have manually labelled clusters in phy these can be uploaded and stored in a datajoint table by running either:

`python <path/to/populate_cluster_table.py> -s subject -d date -n session_no -p probe_name`
e.g `python int-brain-lab\iblapps\launch_phy\populate_cluster_table.py -s KS022 -d 2019-12-10 -n 1 -p probe00`

or:

`python <path/to/populate_cluster_table.py> -s subject -e eid -p probe_name`
e.g. `python int-brain-lab\iblapps\launch_phy\populate_cluster_table.py -e a3df91c8-52a6-4afa-957b-3479a7d0897c -p probe00`
