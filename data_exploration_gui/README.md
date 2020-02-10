# Data Exploration GUI

GUI to allow user to explore ephys data from IBL task

## Setup

GUI must be run from an environment where brainbox is installed.

To create a conda environment with brainbox follow the install instructions here:
http://brainbox.internationalbrainlab.org/usage/installation.html

You will also need to install PyQt5 and pyqtgraph.
These can be installed using the following commands (from within the brainbox environment)

```
pip install pyqt5
pip install pyqtgraph
```

Clone the iblapps repository onto your computer where you have IBL packages installed and checkout the develop branch

```
git clone https://github.com/int-brain-lab/iblapps.git
git checkout develop
```

Go to the ```data_exploration_gui``` folder

```
cd iblapps/data_exploration_gui
```

## Usage
### Getting Data
To download the data run (change the subject, date and session number according to the data you want to download)

```
python load_data.py
```

The GUI can run with or without the ephys.bin data. 

If you want to dowload the ephys data uncomment lines 10 and 11 in ```load_data.py```. **N.B. Downloading ephys.bin data can take a long time!**



### Using GUI
Main window can be launched by running

```
python gui_main.py
```

To read in data click on the ```...``` button in the top left corner and select a data folder.

The GUI expects the ```\alf\probe``` folder. An example selection would be 
```C:\Users\Mayo\Downloads\FlatIron\hoferlab\Subjects\SWC_014\2019-12-10\001\alf\probe00```

N.B. The first time you run the gui with a new dataset it will take a bit of time to load as it is performing some initial calculations of cluster depth and amplitude

