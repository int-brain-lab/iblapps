## Launch Phy with an eid and probe_name

To launch Phy, first follow [these instructions](https://github.com//int-brain-lab/iblenv) for setting up a unified IBL conda environment.

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

## Manual Curation
Manual curation comprises three different steps

1) Labelling clusters (assigning each cluster with a label Good, Mua or Noise)
    * Select a cluster within Cluster View and click on `Edit -> Move best to` and assign your chosen label. 
    * Alternatively you can use the shortcuts alt+G, alt+M, alt+N to label the cluster as good, mua or noise respectively

2)  Additional notes associated with clusters (extra information about clusters, for example if it looks like artifact or drift)
    * Select a cluster within Cluster View
    * Ensure snippet mode is enabled by clicking `File-> Enable Snippet Mode`
    * Type `:l notes your_note` and then hit enter (when typing you should see the text appear in the lower left 
    hand corner of the main window)
    * A new column, with the heading **notes** should have been created in the Cluster View window
        
3)  Merging clusters
    * Clusters can be merged by selecting the two clusters in Cluster View and using `Edit-> Merge` or by pressing G

Make sure you hit the save button frequently during manual curation so your results are saved and can be
recovered in case Phy freezes or crashes!


## Upload manual labels to datajoint

Once you have completed manual curation, the labels that you assigned clusters and any additional notes that you 
added can be uploaded and stored in a datajoint table by running either:

`python <path/to/populate_cluster_table.py> -s subject -d date -n session_no -p probe_name`
e.g `python int-brain-lab\iblapps\launch_phy\populate_cluster_table.py -s KS022 -d 2019-12-10 -n 1 -p probe00`

or:

`python <path/to/populate_cluster_table.py> -s subject -e eid -p probe_name`
e.g. `python int-brain-lab\iblapps\launch_phy\populate_cluster_table.py -e a3df91c8-52a6-4afa-957b-3479a7d0897c -p probe00`

## Upload merge results to google drive
When clusters are merged in Phy, the merge results are saved by updating the indices of clusters in the 
`spikes.clusters.npy` data set. For the time being (until versioning is implemented in the data architecture) we will
save the results of merging on google drive.

The files that need to be uploaded are,
- `spikes.clusters.npy`
- `merge_info.csv`(this is created after running `populate_cluster_table.py`)

These should be uploaded [here](https://drive.google.com/drive/u/1/folders/1_KDshAIblNiFNDQD37ZtoFks8b4VDWgg), 
within a folder using the following naming convention,

subject_date_session_probe_user

e.g KS022_2019-12-10_001_probe00_mayo