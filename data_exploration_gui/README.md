# Data Exploration GUI

GUI to allow user to explore ephys data from IBL task

## Setup

Install ibl environment following [these instructions](https://github.com/int-brain-lab/iblenv#iblenv-installation-guide) 

Go to the ```data_exploration_gui``` folder

```
cd iblapps/data_exploration_gui
```

### Using GUI
To launch the gui you should run the following from the command line. You can specify either a probe insertion id
e.g
```
python data_explore_gui.py -pid 9657af01-50bd-4120-8303-416ad9e24a51
```

or an eid and probe name, e.g
```
python data_explore_gui.py -eid 7f6b86f9-879a-4ea2-8531-294a221af5d0 -name probe00
```
