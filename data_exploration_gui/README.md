# Data Exploration GUI

GUI to allow user to explore ephys data from IBL task

## Setup

Install ibl environment following [these instructions](https://github.com/int-brain-lab/iblenv#iblenv-installation-guide) 

Go to the ```data_exploration_gui``` folder

```
cd iblapps/data_exploration_gui
```

## Usage
### Getting Data
To download the data provide eid (or subject, date, number) and probe name, for example

```
python load_data.py -s SWC_014 -d 2019-12-10 -n 1 -p probe00
or
python load_data.py -e 614e1937-4b24-4ad3-9055-c8253d089919 -p probe00

```

### Using GUI
Main window can be launched by running

```
python gui_main.py
```

To read in data click on the ```...``` button in the top left corner and select a data folder.

The GUI expects the ```\alf\probe``` folder. An example selection would be 
```C:\Users\Mayo\Downloads\FlatIron\hoferlab\Subjects\SWC_014\2019-12-10\001\alf\probe00```

N.B. The first time you run the gui with a new dataset it will take a bit of time to load as it is performing some initial calculations of cluster depth and amplitude

