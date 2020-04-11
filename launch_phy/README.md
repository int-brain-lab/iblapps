## Launch phy with an eid and probe_name

To launch phy, first follow [these instructions](github.com/int-brain-lab/iblenv) for setting up a unified IBL conda environment.

Then in your terminal, checkout the 'launch_phy' branch of 'iblapps' (this branch will soon be merged into 'develop' but will continue to exist), activate your unified conda environment, and run:

`python '<path/to/phy_launcher.py>' '<eid>' '<probe_name>'`

e.g. `python '~/int-brain-lab/iblapps/phy_launcher.py' 'a3df91c8-52a6-4afa-957b-3479a7d0897c' 'probe00'`
