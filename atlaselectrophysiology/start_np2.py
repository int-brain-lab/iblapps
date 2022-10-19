# SH012/2020-01-31/001
from iblapps.atlaselectrophysiology.ephys_atlas_gui import MainWindow

mw1 = MainWindow._get_or_create('tutu')
mw2 = MainWindow._get_or_create('tata')
mw1.show()
mw2.show()
# TODO load data from python script

id(mw1.slice_data) == id(mw2.slice_data)
id(mw1.loaddata.brain_atlas) == id(mw2.loaddata.brain_atlas)
