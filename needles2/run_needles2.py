from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtGui, QtCore,  uic
from PyQt5.QtCore import Qt, QModelIndex
from PyQt5.QtGui import QTransform
import pyqtgraph as pg
import matplotlib

from ibllib.atlas import AllenAtlas
from ibllib.atlas.regions import BrainRegions
from iblutil.numerical import ismember
import qt

from one.api import ONE
from needles2.probe_model import ProbeModel
import copy

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class MainWindow(QtWidgets.QMainWindow):

    @staticmethod
    def _instances():
        app = QtWidgets.QApplication.instance()
        return [w for w in app.topLevelWidgets() if isinstance(w, MainWindow)]

    @staticmethod
    def _get_or_create(title=None, **kwargs):
        av = next(filter(lambda e: e.isVisible() and e.windowTitle() == title,
                         MainWindow._instances()), None)
        if av is None:
            av = MainWindow(**kwargs)
            av.setWindowTitle(title)
        return av

    def __init__(self, lazy=False, res=25):
        super(MainWindow, self).__init__()
        uic.loadUi(Path(__file__).parent.joinpath('mainUI.ui'), self)

        self.atlas = AllenAtlas(res_um=res)
        self.atlas.compute_surface()
        one = ONE()
        self.dist = 354

        # Configure the Menu bar
        menu_bar = QtWidgets.QMenuBar(self)
        menu_bar.setNativeMenuBar(False)
        self.setMenuBar(menu_bar)

        # Add menu bar for mappings
        self.map_menu = menu_bar.addMenu('Mappings')
        self.map_group = QtWidgets.QActionGroup(self.map_menu)
        # Only allow one to plot to be selected at any one time
        self.map_group.setExclusive(True)
        self.add_menu_bar(self.map_menu, self.map_group, list(self.atlas.regions.mappings.keys()),
                          callback=self.change_mapping, default='Allen')

        # Add menu bar for base image
        self.img_menu = menu_bar.addMenu('Images')
        self.img_group = QtWidgets.QActionGroup(self.img_menu)
        self.img_group.setExclusive(True)
        images = ['Image', 'Annotation']
        self.add_menu_bar(self.img_menu, self.img_group, images, callback=self.change_image,
                          default='Image')

        # Add menu bar for coverage
        self.coverage_menu = menu_bar.addMenu('Coverage')
        self.coverage_group = QtWidgets.QActionGroup(self.coverage_menu)
        self.coverage_group.setExclusive(True)
        # coverages = ['Coverage cosine', 'Coverage grid 500', 'Coverage grid 250',
        #              'Coverage grid 100', 'Coverage 354', 'Coverage 250', 'Coverage 120']
        coverages = ['Coverage 354', 'Coverage 250', 'Coverage 120']
        self.add_menu_bar(self.coverage_menu, self.coverage_group, coverages,
                          callback=self.add_coverage, default='Coverage 354')

        # Add menubar for insertions
        self.insertion_menu = menu_bar.addMenu('Insertions')
        self.insertion_group = QtWidgets.QActionGroup(self.insertion_menu)
        self.insertion_group.setExclusive(True)
        insertions = ['Resolved', 'Ephys aligned histology track', 'Histology track',
                      'Histology track (best)', 'Micro-manipulator', 'Micro-manipulator (best)',
                      'Planned', 'Planned (best)']
        self.add_menu_bar(self.insertion_menu, self.insertion_group, insertions,
                          callback=self.add_insertions, default='Histology track (best)')

        # Get all the views
        self.coronal = SliceView(self, self.fig_coronal, self.atlas, slice='coronal',
                                 waxis=0, haxis=2, daxis=1)
        self.sagittal = SliceView(self, self.fig_sagittal, self.atlas, slice='sagittal',
                                  waxis=1, haxis=2, daxis=0)
        self.horizontal = SliceView(self, self.fig_horizontal, self.atlas, slice='horizontal',
                                    waxis=1, haxis=0, daxis=2)
        self.horizontal.fig_slice.getViewBox().invertY(True)

        self.top = TopView(self, self.fig_top, self.atlas)
        self.probe = ProbeView(self, self.fig_probe)
        self.coverage = CoverageView(self)
        self.probe_model = ProbeModel(one=one, ba=self.atlas, lazy=lazy)
        self.region = RegionView(self, self.atlas)
        self.table = InsertionTableView(self)
        self.layers = LayersView(self)
        self.change_mapping()

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.coverage)
        self.coverage_placeholder.setLayout(layout)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.region)
        self.region_placeholder.setLayout(layout)

        if not lazy:
            self.initialise()

    def initialise(self, second_pass_map=None):
        self.add_insertions()
        self.add_coverage()
        self.missing_coverage(second_pass_map=second_pass_map)
        self.coverage_increase()
        self.layers.initialise_table()

    def missing_coverage(self, second_pass_map=None):
        # get the coverage volume layer from the coverage
        cov = self.coronal.ctrl.get_image_layer('coverage').slice_kwargs['region_values']
        self.second_pass = self.load_second_pass_volume(second_pass_map=second_pass_map)
        self.second_pass_missing = np.copy(self.second_pass)
        self.second_pass_missing[cov >= 1] = np.nan
        self.ixyz_second = np.where(~np.isnan(self.second_pass.flatten()))[0]
        self.ixyz_second_missing = np.where(~np.isnan(self.second_pass_missing.flatten()))[0]
        self.add_volume_layer(self.second_pass_missing, name='missing', cmap='Reds', levels=(0, 1))


    def change_mapping(self):
        data = np.unique(self.atlas.regions.name[self.atlas._get_mapping(
            mapping=self.get_mapping())])
        self.region.populate_combobox(data)
        self._refresh()

    def get_mapping(self):
        return self.map_group.checkedAction().text()

    def change_image(self):
        self.coronal.ctrl.change_base_layer(self.img_group.checkedAction().text())
        self.sagittal.ctrl.change_base_layer(self.img_group.checkedAction().text())
        self.horizontal.ctrl.change_base_layer(self.img_group.checkedAction().text())
        self._refresh()

    def add_coverage(self):
        provenance = self.insertion_group.checkedAction().text()
        coverage_choice = self.coverage_group.checkedAction().text()
        if not self.probe_model.initialised:
            self.probe_model.initialise()

        if '(best)' in provenance:
            provenance = provenance[:-7]
            self.probe_model.compute_best_for_provenance(provenance)
            self.provenance = 'Best'
        else:
            self.provenance = provenance

        if 'cosine' in coverage_choice:
            all_channels = self.probe_model.get_all_channels(self.provenance)
            cov = self.probe_model.compute_coverage(all_channels)
            bc = None
        elif 'grid' in coverage_choice:
            all_channels = self.probe_model.get_all_channels(self.provenance)
            bin_size = int(coverage_choice[-3:])
            cov, bc = self.probe_model.grid_coverage(all_channels, bin_size)
        else:
            self.dist = int(coverage_choice[-3:])
            # self.dist = 354
            cov = self.probe_model.report_coverage(self.provenance, self.dist)
            bc = None

        self.add_volume_layer(cov, 'coverage', bc=bc)
        self._refresh()

    def add_extra_coverage(self, traj, ins=None):
        # This had an ins option, need to look into that ra
        # when double clicked
        self.top.ctrl.set_scatter_layer('active_insertion', x=[traj['x'] / 1e6],
                                        y=[traj['y'] / 1e6])
        self.do_this_thing(traj, ins=ins)

    def coverage_added(self):
        # when add button pressed
        # get out the pandas df
        #df = self.coverage.ctrl.model.to_df()
        df = self.coverage.ctrl.model.get_traj()
        # Add the insertions twice
        print('i got this far')
        self.table.on_insert_row(df)
        # self.table.on_insert_row(df) #TODO instead multiply by factor of two

        self.get_coverage_from_table()

        #self.table.tableview.selectionModel().blockSignals(True)
        #self.table.tableview.clearSelection()
        #self.table.tableview.selectionModel().blockSignals(False)

    def get_coverage_from_table(self):

        if self.table.ctrl.model.rowCount() > 0:
            print(0)
            print(self.table.ctrl.model.to_dict())
            print(1)
            cov, _ = self.probe_model.compute_coverage(self.table.ctrl.model.to_dict(),
                                                       dist_fcn=[self.dist, self.dist + 1])
            print(2)
            self.add_volume_layer(cov, name='planned_insertions', cmap='Purples', levels=(0, 1))
            print(3)
            self.top.ctrl.set_scatter_layer('planned_insertions',
                                            x=self.table.ctrl.model._data.x.values / 1e6,
                                            y=self.table.ctrl.model._data.y.values / 1e6)

        else:
            self.remove_volume_layer('planned_insertions')
            self.top.ctrl.set_scatter_layer('planned_insertions')
        print(4)
        self.remove_volume_layer('selected_insertion')
        print(5)
        self.top.ctrl.set_scatter_layer('active_insertion')
        print(6)
        self.top.ctrl.set_scatter_layer('selected_insertion')
        print(7)
        self._refresh()
        print(8)
        self.layers.update_table('planned_insertions')
        print(9)
        #self.coverage_increase()


    def add_trajectory(self, x, y):
        self.coverage.ctrl.model.initialise_data()
        self.coverage.ctrl.set_value(x, 'x')
        self.coverage.ctrl.set_value(y, 'y')
        self.coverage.update_view()

    def add_volume_layer(self, volume, name, cmap='viridis', opacity=0.8, levels=None,
                         bc=None):
        self.coronal.ctrl.add_volume_layer(volume, name=name, cmap=cmap, opacity=opacity,
                                           levels=levels, bc=bc)
        self.sagittal.ctrl.add_volume_layer(volume, name=name, cmap=cmap, opacity=opacity,
                                            levels=levels, bc=bc)
        self.horizontal.ctrl.add_volume_layer(volume, name=name, cmap=cmap, opacity=opacity,
                                              levels=levels, bc=bc)

    def remove_volume_layer(self, name):
        self.coronal.ctrl.remove_image_layer(name)
        self.sagittal.ctrl.remove_image_layer(name)
        self.horizontal.ctrl.remove_image_layer(name)

    def toggle_volume_layer(self, name, checked):
        self.coronal.ctrl.toggle_image_layer(name, checked)
        self.sagittal.ctrl.toggle_image_layer(name, checked)
        self.horizontal.ctrl.toggle_image_layer(name, checked)
        self._refresh()


    def add_insertion_by_id(self, ins_id):
        traj, ins = self.probe_model.insertion_by_id(ins_id)
        self.add_extra_coverage(traj, ins)

    def add_insertions(self):
        provenance = self.insertion_group.checkedAction().text()
        if not self.probe_model.initialised:
            self.probe_model.initialise()

        if '(best)' in provenance:
            provenance = provenance[:-7]
            self.probe_model.compute_best_for_provenance(provenance)
            self.provenance = 'Best'
        else:
            if not 'traj' in self.probe_model.traj[provenance].keys():

                self.probe_model.get_traj_for_provenance(provenance)
            self.provenance = provenance

        self.top.ctrl.set_scatter_layer('coverage',
                                        x=self.probe_model.traj[self.provenance]['x']/1e6,
                                        y=self.probe_model.traj[self.provenance]['y']/1e6)

    def on_insertion_clicked(self, scatter, point):
        idx = np.argwhere(self.probe_model.traj[self.provenance]['x']/1e6 ==
                          point[0].pos().x())[0][0]
        traj = self.probe_model.traj[self.provenance]['traj'][idx]

        self.do_this_thing(traj)

    def on_first_pass_insertion_clicked(self, scatter, point):
        idx = np.argwhere(self.first_pass_map.x.values/1e6 ==
                          point[0].pos().x())[0][0]
        traj = self.first_pass_map.iloc[idx].to_dict()

        self.do_this_thing(traj)


# TODO need to catch for dodgey coordinates that return rubbish

    def do_this_thing(self, traj, ins=None):
        self.top.ctrl.set_scatter_layer('selected_insertion', x=[traj['x'] / 1e6],
                                        y=[traj['y'] / 1e6])


        (region, region_lab, region_col) = self.probe_model.get_brain_regions(
            traj, ins=ins, mapping=self.get_mapping())
        self.probe.plot_region_along_probe(region, region_lab, region_col)

        cov, xyz_cnt = self.probe_model.compute_coverage([traj], limit=False,
                                                         dist_fcn=[self.dist, self.dist + 1])
        self.add_volume_layer(cov, name='selected_insertion', cmap='Wistia')

        self.refresh_slices('horizontal', 'x', xyz_cnt[1])
        self.refresh_slices('horizontal', 'y', xyz_cnt[0])
        self.refresh_slices('sagittal', 'y', xyz_cnt[2])
        self._refresh()
# TODO figure this out
        # self.table.tableview.selectionModel().blockSignals(True)
        # self.table.tableview.clearSelection()
        # self.table.tableview.selectionModel().blockSignals(False)

    def on_planned_insertion_clicked(self, scatter, point):
        idx = np.argwhere(self.table.ctrl.model._data.x.values/1e6 ==
                          point[0].pos().x())[0][0]
        traj = self.table.ctrl.model._data.iloc[idx].to_dict()

        self.do_this_thing(traj)

        # also highlight the table
        # self.table.tableview.selectionModel().blockSignals(True)
        # self.table.tableview.selectRow(idx)
        # self.table.tableview.selectionModel().blockSignals(False)

    def on_active_insertion_clicked(self, scatter, point):
        traj = self.coverage.ctrl.model.get_traj()
        self.do_this_thing(traj)


    def add_menu_bar(self, menu, group, items, callback=None, default=None):
        for item in items:
            if item == default:
                _item = QtWidgets.QAction(item, self, checkable=True, checked=True)
            else:
                _item = QtWidgets.QAction(item, self, checkable=True, checked=False)
            if callback:
                _item.triggered.connect(callback)
            menu.addAction(_item)
            group.addAction(_item)

        return menu, group

    def _refresh(self):
        self._refresh_sagittal()
        self._refresh_coronal()
        self._refresh_horizontal()

    def _refresh_coronal(self):
        self.coronal.ctrl.set_slice()

    def _refresh_sagittal(self):
        self.sagittal.ctrl.set_slice()

    def _refresh_horizontal(self):
        self.horizontal.ctrl.set_slice()

    def refresh_highlighted_region(self, region, mlapdv):
        if (region is None) or (region['acronym'][0] == 'void'):
            self._reset_region('non_locked')
            self.region.update_labels(mlapdv[0], mlapdv[1], mlapdv[2], None)
        else:
            self._refresh_non_locked_region(region['id'][0])
            self.region.update_labels(mlapdv[0], mlapdv[1], mlapdv[2], region)

    def refresh_locked_region(self, region):
        region_values = np.zeros_like(self.atlas.regions.id)
        self._change_region_value('non_locked', region_values)

        region_idx = np.where(self.atlas.regions.id == region)[0]
        region_values[region_idx] = 100
        self._change_region_value('locked', region_values)
        self._refresh()
        self.region.update_selected_region(region)

    def _reset_region(self, name):
        region_values = np.zeros_like(self.atlas.regions.id)
        self._change_region_value(name, region_values)
        self._refresh()

    def _refresh_non_locked_region(self, region):
        region_values = np.zeros_like(self.atlas.regions.id)
        region_idx = np.where(self.atlas.regions.id == region)[0]
        region_values[region_idx] = 100
        self._change_region_value('non_locked', region_values)
        self._refresh()

    def _change_region_value(self, name, values):
        image = self.coronal.ctrl.get_image_layer(name=name)
        image.slice_kwargs['region_values'] = values
        image = self.sagittal.ctrl.get_image_layer(name=name)
        image.slice_kwargs['region_values'] = values
        image = self.horizontal.ctrl.get_image_layer(name=name)
        image.slice_kwargs['region_values'] = values

    def refresh_slices(self, slice, orientation, value, move_top=True):
        if slice == 'coronal' and orientation == 'x':
            self.sagittal.slice_coord = value
            self.horizontal.line_y.setValue(value)
            self.coronal.line_x.setValue(value)
            if move_top:
                self.top.line_x.setValue(value)
            self._refresh_sagittal()

        if slice == 'coronal' and orientation == 'y':
            self.horizontal.slice_coord = value
            self.sagittal.line_y.setValue(value)
            self.coronal.line_y.setValue(value)
            self._refresh_horizontal()

        if slice == 'sagittal' and orientation == 'x':
            self.coronal.slice_coord = value
            self.horizontal.line_x.setValue(value)
            self.sagittal.line_x.setValue(value)
            if move_top:
                self.top.line_y.setValue(value)
            self._refresh_coronal()

        if slice == 'sagittal' and orientation == 'y':
            self.horizontal.slice_coord = value
            self.coronal.line_y.setValue(value)
            self.sagittal.line_y.setValue(value)
            self._refresh_horizontal()

        if slice == 'horizontal' and orientation == 'x':
            self.coronal.slice_coord = value
            self.sagittal.line_x.setValue(value)
            self.horizontal.line_x.setValue(value)
            self.top.line_y.setValue(value)
            self._refresh_coronal()

        if slice == 'horizontal' and orientation == 'y':
            self.sagittal.slice_coord = value
            self.coronal.line_x.setValue(value)
            self.horizontal.line_y.setValue(value)
            self.top.line_x.setValue(value)
            self._refresh_sagittal()

    def load_first_pass_map(self):
        first_pass_map_path = r'C:\Users\Mayo\iblenv\ibllib-matlab\needles\maps\first_pass_map.csv'
        self.first_pass_map = pd.read_csv(first_pass_map_path)
        self.first_pass_map = self.first_pass_map.rename(columns={'ml_um': 'x', 'ap_um': 'y',
                                                                  'dv_um': 'z',
                                                                  'depth_um': 'depth'})
        self.first_pass_map = self.first_pass_map.drop(columns=['coronal_index', 'sagittal_index',
                                                                'index'])
        self.first_pass_map['provenance'] = 'Planned'

        cov, _ = self.probe_model.compute_coverage(self.first_pass_map.to_dict(orient='records'),
                                                   dist_fcn=[self.dist, self.dist + 1])
        self.add_volume_layer(cov, name='first_pass', cmap='YlGnBu', levels=(0, 1))
        self.top.ctrl.set_scatter_layer('first_pass', x=self.first_pass_map.x.values / 1e6,
                                        y=self.first_pass_map.y.values / 1e6)

    def load_second_pass_volume(self, second_pass_map=None):

        if second_pass_map:
            reg_volume2 = np.load(second_pass_map)
            reg_volume2[reg_volume2 == 0] = np.nan
        else:
            flat_ind = np.arange(self.probe_model.ba.image.shape[0] *
                                 self.probe_model.ba.image.shape[1] *
                                 self.probe_model.ba.image.shape[2])
            ixyz = np.unravel_index(flat_ind,
                                    shape=(self.probe_model.ba.image.shape[0],
                                           self.probe_model.ba.image.shape[1],
                                           self.probe_model.ba.image.shape[2]))
            ixyz = (np.c_[ixyz[1], ixyz[0], ixyz[2]])
            xyz = self.probe_model.ba.bc.i2xyz(ixyz.astype(np.float))

            ra = self.probe_model.ba.label.flatten().astype(np.float64)
            flat = np.zeros_like(ra)

            # because
            rh = int((np.max(ra) + 1) / 2)
            t = np.bitwise_and(ra >= rh, xyz[:, 1] < 3800 / 1e6)
            # t = np.where(ra >= rh)
            flat[t] = 1
            flat[ra == 0] = 0

            reg_volume2 = flat.reshape(self.probe_model.ba.image.shape[0],
                                       self.probe_model.ba.image.shape[1],
                                       self.probe_model.ba.image.shape[2]).astype(np.float32)
            reg_volume2[reg_volume2 == 0] = np.nan

        return reg_volume2

    def coverage_increase(self):

        layer = None
        layer_planned = self.coronal.ctrl.get_image_layer('planned_insertions')
        if layer_planned is not None:
            layer = (copy.deepcopy(layer_planned.slice_kwargs['region_values']))

        layer_active = self.coronal.ctrl.get_image_layer('selected_insertion')
        if layer_active is not None:
            if layer is not None:
                layer += (copy.deepcopy(layer_active.slice_kwargs['region_values']) * 2)
            else:
                layer = (copy.deepcopy(layer_active.slice_kwargs['region_values']) * 2)

        layer_covered = self.coronal.ctrl.get_image_layer('coverage')
        if layer_covered is not None:
            if layer is not None:
                layer += copy.deepcopy(layer_covered.slice_kwargs['region_values'])
            else:
                layer = copy.deepcopy(layer_covered.slice_kwargs['region_values'])

        self.layer_val = layer
        if layer is not None:
            la = layer.flatten()
            # percentage of the second volume that is covered,
            # so we need to add the coverage to the main coverage

            bla = la[self.ixyz_second]
            per0 = (np.where(bla == 0)[0].shape[0] / self.ixyz_second.shape[0]) * 100
            per1 = (np.where(bla == 1)[0].shape[0] / self.ixyz_second.shape[0]) * 100
            per2 = (np.where(bla > 1)[0].shape[0] / self.ixyz_second.shape[0]) * 100

            self.coverage.coverage_label.setText(f'0%: {np.round(per0, 2)} %, '
                                                 f'1%: {np.round(per1, 2)} %, '
                                                 f'2%: {np.round(per2, 2)} % ')


class LayersView(QtWidgets.QWidget):
    def __init__(self, qmain: MainWindow):
        self.qmain = qmain
        super(LayersView, self).__init__()
        self.layer_list = self.qmain.layer_list
        self.ignore_layers = ['base', 'locked', 'non_locked']

        self.layer_list.itemClicked.connect(self.layer_clicked)

    def initialise_table(self):
        for layer in self.qmain.coronal.ctrl.image_layers:
            if not layer.name in self.ignore_layers:
                item = QtWidgets.QListWidgetItem()
                item.setText(layer.name)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked)
                self.layer_list.addItem(item)

    def layer_clicked(self, item):
        if item.checkState() == QtCore.Qt.Checked:
            self.qmain.toggle_volume_layer(item.text(), True)
            self.qmain.top.ctrl.toggle_scatter_layer(item.text(), True)
        else:
            self.qmain.toggle_volume_layer(item.text(), False)
            self.qmain.top.ctrl.toggle_scatter_layer(item.text(), False)
    # need to add layers
    # if they are not ticked then you keep them unticked

    def update_table(self, name):
        # Check if it already on the list
        for x in range(self.layer_list.count()):
            _item = self.layer_list.item(x)
            if _item.text() == name:

                if _item.checkState() != QtCore.Qt.Checked:
                    _item.setCheckState(QtCore.Qt.Checked)
                    self.qmain.top.ctrl.toggle_scatter_layer(_item.text(), True)
                return

        # Otherwise add to the list
        item = QtWidgets.QListWidgetItem()
        item.setText(name)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Checked)
        self.layer_list.addItem(item)


class RegionView(QtWidgets.QWidget):
    def __init__(self, qmain: MainWindow, atlas: AllenAtlas):
        self.atlas = atlas
        self.qmain = qmain
        super(RegionView, self).__init__()
        uic.loadUi(Path(__file__).parent.joinpath('regionUI.ui'), self)

        self.region_list = QtGui.QStandardItemModel()
        self.region_comboBox.setLineEdit(QtWidgets.QLineEdit())
        region_completer = QtWidgets.QCompleter()
        region_completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.region_comboBox.setCompleter(region_completer)
        self.region_comboBox.setModel(self.region_list)
        self.region_comboBox.completer().setModel(self.region_list)
        self.region_comboBox.activated.connect(self.on_region_selected)

    def populate_combobox(self, data):
        self.region_list.clear()
        for dat in data:
            item = QtGui.QStandardItem(dat)
            item.setEditable(False)
            self.region_list.appendRow(item)

        # Set the default to be the first option
        self.region_comboBox.setCurrentIndex(0)

    def update_labels(self, ml, ap, dv, region):
        self.ml_label.setText(f"{ml * 1e6:.0f}")
        self.ap_label.setText(f"{ap * 1e6:.0f}")
        self.dv_label.setText(f"{dv * 1e6:.0f}")
        if region is None:
            self.current_region_label.setText("")
        else:
            self.current_region_label.setText(region['acronym'][0])

    def update_selected_region(self, region):
        region_idx = np.where(self.atlas.regions.id == region)[0][0]
        region_name = self.atlas.regions.name[region_idx]
        item = self.region_list.findItems(region_name)
        self.region_comboBox.setCurrentIndex(self.region_list.indexFromItem(item[0]).row())

    def on_region_selected(self, idx):
        region_name = self.region_list.item(idx).text()
        region_idx = np.where(self.atlas.regions.name == region_name)[0][0]
        self.qmain.refresh_locked_region(self.atlas.regions.id[region_idx])


class CoverageView(QtWidgets.QWidget):
    def __init__(self, qmain: MainWindow):
        self.qmain = qmain
        super(CoverageView, self).__init__()
        uic.loadUi(Path(__file__).parent.joinpath('coverageUI.ui'), self)

        self.ctrl = CoverageController(self)

        self.populate_combobox(self.x_comboBox, self.ctrl.model.x_steps)
        self.populate_combobox(self.y_comboBox, self.ctrl.model.y_steps)
        self.populate_combobox(self.d_comboBox, self.ctrl.model.d_steps)
        self.populate_combobox(self.t_comboBox, self.ctrl.model.t_steps)
        self.populate_combobox(self.p_comboBox, self.ctrl.model.p_steps)
        self.x_comboBox.currentIndexChanged.connect(lambda: self.update_step(self.x_comboBox, 'x'))
        self.y_comboBox.currentIndexChanged.connect(lambda: self.update_step(self.y_comboBox, 'y'))
        self.d_comboBox.currentIndexChanged.connect(lambda: self.update_step(self.d_comboBox, 'd'))
        self.t_comboBox.currentIndexChanged.connect(lambda: self.update_step(self.t_comboBox, 't'))
        self.p_comboBox.currentIndexChanged.connect(lambda: self.update_step(self.p_comboBox, 'p'))

        self.xplus_pushButton.clicked.connect(lambda: self.ctrl.increase_value('x'))
        self.yplus_pushButton.clicked.connect(lambda: self.ctrl.increase_value('y'))
        self.dplus_pushButton.clicked.connect(lambda: self.ctrl.increase_value('d'))
        self.tplus_pushButton.clicked.connect(lambda: self.ctrl.increase_value('t'))
        self.pplus_pushButton.clicked.connect(lambda: self.ctrl.increase_value('p'))

        self.xminus_pushButton.clicked.connect(lambda: self.ctrl.decrease_value('x'))
        self.yminus_pushButton.clicked.connect(lambda: self.ctrl.decrease_value('y'))
        self.dminus_pushButton.clicked.connect(lambda: self.ctrl.decrease_value('d'))
        self.tminus_pushButton.clicked.connect(lambda: self.ctrl.decrease_value('t'))
        self.pminus_pushButton.clicked.connect(lambda: self.ctrl.decrease_value('p'))

        self.x_label.returnPressed.connect(self.set_values)
        self.y_label.returnPressed.connect(self.set_values)
        self.d_label.returnPressed.connect(self.set_values)
        self.t_label.returnPressed.connect(self.set_values)
        self.p_label.returnPressed.connect(self.set_values)

        self.add_pushButton.clicked.connect(self.qmain.coverage_added)

        self.update_labels()

    def populate_combobox(self, combobox, values):
        for val in values:
            combobox.addItem(str(val))

    def update_step(self, combobox, target):
        index = combobox.currentIndex()
        step = int(combobox.itemText(index))
        self.ctrl.set_step(step, target)

    def update_labels(self):
        self.x_label.setText(f"{self.ctrl.model.data['x']['value']:.0f}")
        self.y_label.setText(f"{self.ctrl.model.data['y']['value']:.0f}")
        self.d_label.setText(f"{self.ctrl.model.data['d']['value']:.0f}")
        self.t_label.setText(f"{self.ctrl.model.data['t']['value']:.0f}")
        self.p_label.setText(f"{self.ctrl.model.data['p']['value']:.0f}")

    def update_view(self):
        self.update_labels()
        self.qmain.add_extra_coverage(self.ctrl.model.get_traj())
        self.qmain.coverage_increase()

    def set_values(self):
        self.ctrl.set_value(int(self.x_label.text()), 'x')
        self.ctrl.set_value(int(self.y_label.text()), 'y')
        self.ctrl.set_value(int(self.d_label.text()), 'd')
        self.ctrl.set_value(int(self.t_label.text()), 't')
        self.ctrl.set_value(int(self.p_label.text()), 'p')
        self.update_view()


class InsertionTableView(QtWidgets.QWidget):
    def __init__(self, qmain: MainWindow):
        super(InsertionTableView, self).__init__()
        self.qmain = qmain
        data = pd.DataFrame(columns=['x', 'y', 'z', 'phi', 'theta', 'depth', 'provenance'])
        self.ctrl = InsertionTableController(self, data)

        self.tableview = self.qmain.tableview
        self.tableview.setModel(self.ctrl.model)

        # self.delete_button = QtWidgets.QPushButton('-')
        self.delete_button = self.qmain.delete_button
        self.delete_button.clicked.connect(self.on_delete_pressed)
        self.tableview.selectionModel().selectionChanged.connect(self.row_selected)
        self.ctrl.model.dataChanged.connect(self.data_changed)

    def on_delete_pressed(self):
        if len(self.tableview.selectedIndexes()) == self.ctrl.model.columnCount():
            row = self.tableview.selectedIndexes()[0].row()
            self.ctrl.model.removeRow(row)
            self.qmain.get_coverage_from_table()

    def on_insert_row(self, df):
        self.ctrl.model.insertRows(self.ctrl.model.rowCount(), 1, QModelIndex(), df)

    def row_selected(self):
        if len(self.tableview.selectedIndexes()) == self.ctrl.model.columnCount():
            row = self.tableview.selectedIndexes()[0].row()
            traj = self.ctrl.model._data.iloc[row].to_dict()
            self.qmain.do_this_thing(traj)

    def initialise_data(self, data):
        self.ctrl.model._data = data
        self.qmain.get_coverage_from_table()

    def data_changed(self):
        print('IN DATA CHANGED')
        self.qmain.get_coverage_from_table()


class InsertionTableController:
    def __init__(self, view: InsertionTableView, data):
        self.view = view
        self.model = InsertionTableModel(data)


class InsertionTableModel(QtCore.QAbstractTableModel):
    def __init__(self, data=None):
        super(InsertionTableModel, self).__init__()
        if data is None:
            data = pd.DataFrame()
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole or role == Qt.EditRole:
                value = self._data.iloc[index.row(), index.column()]
                return str(value)

    def setData(self, index, value, role):
        print('SET DATA')
        if role == Qt.EditRole:
            self._data.iloc[index.row(), index.column()] = int(value)
            self.dataChanged.emit(index, index)
            return True
        return False

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return self._data.index[col]

    def insertRows(self, position, rows, QModelIndex, data=None):
        print('now i am heref')
        self.beginInsertRows(QModelIndex, position, position+rows-1)
        for i in range(rows):
            self._data = pd.concat([self._data, pd.DataFrame.from_dict([data])])
        self._data.reset_index(drop=True, inplace=True)
        self.endInsertRows()
        self.layoutChanged.emit()
        return True

    def removeRows(self, position, rows, QModelIndex):
        self.beginRemoveRows(QModelIndex, position, position+rows-1)
        for i in range(rows):
            self._data.drop(position + i, inplace=True)
        self._data.reset_index(drop=True, inplace=True)
        self.endRemoveRows()
        self.layoutChanged.emit()
        return True

    def flags(self, index):
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def to_dict(self):
        return self._data.to_dict(orient='records')


class CoverageController:
    def __init__(self, view: CoverageView):
        self.view = view
        self.model = CoverageModel()

    def increase_value(self, target):
        self.model.data[target]['value'] += self.model.data[target]['step']
        if self.model.data[target]['max'] is not None:
            if self.model.data[target]['value'] > self.model.data[target]['max']:
                self.model.data[target]['value'] = self.model.data[target]['max']

        self.view.update_view()

    def decrease_value(self, target):
        self.model.data[target]['value'] -= self.model.data[target]['step']
        if self.model.data[target]['min'] is not None:
            if self.model.data[target]['value'] < self.model.data[target]['min']:
                self.model.data[target]['value'] = self.model.data[target]['min']

        self.view.update_view()

    def set_step(self, value, target):
        self.model.data[target]['step'] = value

    def set_value(self, value, target):
        self.model.data[target]['value'] = value


class CoverageModel:
    def __init__(self):
        self.data = {'x': {}, 'y': {}, 'd': {}, 't': {}, 'p': {}}
        self.x_steps = [100, 500, 50, 10]
        self.y_steps = [100, 500, 50, 10]
        self.d_steps = [100, 500, 50, 10]
        self.t_steps = [5, 1, 10]
        self.p_steps = [5, 1, 10]
        self.x_minmax = [None, None]
        self.y_minmax = [None, None]
        self.d_minmax = [None, None]
        self.t_minmax = [-30, 30]
        self.p_minmax = [-30, 30]

        self.initialise_data()

    def initialise_data(self):
        self._initialise_data('x', self.x_steps[0], self.x_minmax, start_value=0)
        self._initialise_data('y', self.y_steps[0], self.y_minmax, start_value=0)
        self._initialise_data('d', self.d_steps[0], self.d_minmax, start_value=4000)
        self._initialise_data('t', self.t_steps[0], self.t_minmax, start_value=15)
        self._initialise_data('p', self.p_steps[0], self.p_minmax, start_value=0)

    def _initialise_data(self, target, step, minmax=(None, None), start_value=None):
        self.data[target]['value'] = start_value
        self.data[target]['step'] = step
        self.data[target]['min'] = minmax[0]
        self.data[target]['max'] = minmax[1]

    def get_traj(self):
        traj = {'x': float(self.data['x']['value']),
                'y': float(self.data['y']['value']),
                'z': 0.0,
                'phi': float(self.data['p']['value']),
                'theta': float(self.data['t']['value']),
                'depth': float(self.data['d']['value']),
                'provenance': 'Planned'}

        return traj

    def to_df(self):
        return pd.DataFrame([self.get_traj()])


class ProbeView:
    def __init__(self, qmain: MainWindow, fig: pg.PlotWidget):
        self.qmain = qmain
        self.fig_probe = fig
        self.fig_probe.setYRange(min=0, max=4000)

    def plot_region_along_probe(self, region, region_label, region_color):
        self.fig_probe.clear()
        axis = self.fig_probe.getAxis('left')
        axis.setTicks([region_label])
        axis.setZValue(10)

        # Plot each histology region
        for reg, reg_col in zip(region, region_color):
            colour = QtGui.QColor(*reg_col)
            region = pg.LinearRegionItem(values=(reg[0], reg[1]),
                                         orientation=pg.LinearRegionItem.Horizontal,
                                         brush=colour, movable=False)
            self.fig_probe.addItem(region)

        self.fig_probe.setYRange(min=0, max=4000)
        axis = self.fig_probe.getAxis('bottom')
        axis.setTicks([[(0, ''), (0.5, ''), (1, '')]])


class TopView:
    def __init__(self, qmain: MainWindow, fig: pg.PlotWidget, atlas: AllenAtlas, **kwargs):
        self.qmain = qmain
        self.atlas = atlas
        self.fig_top = fig
        self.ctrl = TopController(self, qmain, atlas, **kwargs)

        self.fig_top.setAspectLocked(True)
        self.ctrl.add_image_layer(name='top')

        line_kwargs = {'movable': True, 'pen': pg.mkPen((0, 255, 0), width=1)}
        self.line_x = pg.InfiniteLine(angle=90, pos=0, **line_kwargs)
        self.line_y = pg.InfiniteLine(angle=0, pos=0, **line_kwargs)
        self.line_x.setValue(0)
        self.line_y.setValue(0)

        self.line_x.sigDragged.connect(
            lambda: self.qmain.refresh_slices('coronal', 'x', self.line_x.value(), False))
        self.line_y.sigDragged.connect(
            lambda: self.qmain.refresh_slices('sagittal', 'x', self.line_y.value(), False))
        self.fig_top.addItem(self.line_x)
        self.fig_top.addItem(self.line_y)

        # selected insertion
        self.ctrl.add_scatter_layer('coverage', pen='y', brush='y',
                                    callback=self.qmain.on_insertion_clicked)

        # planned
        self.ctrl.add_scatter_layer('planned_insertions', pen='m', brush='m',
                                    callback=self.qmain.on_planned_insertion_clicked)

        # coverage
        self.ctrl.add_scatter_layer('active_insertion', pen='r', brush='r',
                                    callback=self.qmain.on_active_insertion_clicked)

        self.ctrl.add_scatter_layer('selected_insertion', pen='k', brush=pg.mkBrush(None))

        # First pass
        self.ctrl.add_scatter_layer('first_pass', pen='b', brush='b',
                                    callback=self.qmain.on_first_pass_insertion_clicked)

        s = self.fig_top.scene()
        s.sigMouseClicked.connect(self.mouseClick)

        self.ctrl.set_top()

    def mouseClick(self, event):
        if event.double():
            qpoint = self.ctrl.image_layers[0].image_item.mapFromScene(event.scenePos())
            iw, ih, w, h, v = self.ctrl.cursor2xyamp(qpoint)
            #TODO check within bounds
            self.ctrl.set_scatter_layer('selected_insertion', x=[w], y=[h])
            self.qmain.add_trajectory(w * 1e6, h * 1e6)


class SliceView:
    """
    Window containing a volume slice
    """
    def __init__(self, qmain: MainWindow, fig: pg.PlotWidget, atlas: AllenAtlas,
                 slice, waxis, haxis, daxis):
        self.qmain = qmain
        self.atlas = atlas
        self.slice = slice
        self.slice_coord = 0
        self.fig_slice = fig

        self.ctrl = SliceController(self, qmain, atlas, waxis, haxis, daxis)
        self.fig_slice.setAspectLocked(True)

        self.ctrl.add_image_layer(slice_kwargs={'volume': 'image', 'mode': 'clip'},
                                  pg_kwargs={'opacity': 0.8})
        self.ctrl.add_mask_layer(name='locked', cmap='Blues')
        self.ctrl.add_mask_layer(name='non_locked', cmap='Greens')

        # connect signals and slots
        s = self.fig_slice.scene()
        self.proxy = pg.SignalProxy(s.sigMouseMoved, rateLimit=60, slot=self.mouseMoveEvent)
        s.sigMouseClicked.connect(self.mouseClick)

        line_kwargs = {'movable': True, 'pen': pg.mkPen((0, 125, 0), width=1)}
        self.line_x = pg.InfiniteLine(angle=90, pos=0, **line_kwargs)
        self.line_y = pg.InfiniteLine(angle=0, pos=0, **line_kwargs)
        self.line_x.sigDragged.connect(lambda: self.line_dragged(self.line_x, 'x'))
        self.line_y.sigDragged.connect(lambda: self.line_dragged(self.line_y, 'y'))
        self.fig_slice.addItem(self.line_x)
        self.fig_slice.addItem(self.line_y)

    def line_dragged(self, line, orientation):
        value = line.value()
        self.qmain.refresh_slices(self.slice, orientation, value)

    def mouseClick(self, event):
        if event.double():
            self.qmain._reset_region('locked')
        else:
            qpoint = self.ctrl.image_layers[0].image_item.mapFromScene(event.scenePos())
            iw, ih, w, h, v, region, _ = self.ctrl.cursor2xyamp(qpoint)
            if region and region['acronym'][0] != 'void':
                self.qmain.refresh_locked_region(region['id'][0])

    def mouseMoveEvent(self, scenepos):
        if isinstance(scenepos, tuple):
            scenepos = scenepos[0]
        else:
            return
        qpoint = self.ctrl.image_layers[0].image_item.mapFromScene(scenepos)
        iw, ih, w, h, v, region, xyz = self.ctrl.cursor2xyamp(qpoint)

        self.qmain.refresh_highlighted_region(region, xyz)


class BaseController:
    """
    Abstract class that implements mapping from axes to voxels for any window.
    Not instantiated directly.
    """
    def __init__(self, atlas: AllenAtlas, fig):
        self.transform = None  # affine transform image indices 2 data domain
        self.image_layers = []
        self.atlas = atlas
        self.fig = fig

    def cursor2xyamp(self, qpoint):
        """Used for the mouse hover function over image display"""
        iw, ih = self.cursor2ind(qpoint)
        v = self.im[iw, ih]
        w, h, _ = np.matmul(self.transform, np.array([iw, ih, 1]))
        return iw, ih, w, h, v

    def cursor2ind(self, qpoint):
        """ image coordinates over the image display"""
        iw = np.max((0, np.min((int(np.floor(qpoint.x())), self.nw - 1))))
        ih = np.max((0, np.min((int(np.round(qpoint.y())), self.nh - 1))))
        return iw, ih

    def set_image(self, image_item, im, dw, dh, w0, h0, **pg_kwargs):
        """
        :param im:
        :param dw:
        :param dh:
        :param w0:
        :param h0:
        :param pgkwargs: pg.ImageItem.setImage() parameters: level=None, lut=None, opacity=1
        :return:
        """
        self.im = im
        self.nw, self.nh = self.im.shape[0:2]
        image_item.setImage(self.im, **pg_kwargs)
        transform = [dw, 0., 0., 0., dh, 0., w0, h0, 1.]
        self.transform = np.array(transform).reshape((3, 3)).T
        image_item.setTransform(QTransform(*transform))

    def add_image_layer(self, idx=None, **kwargs):
        """
        :param name: name of the image item to keep track of layers
        :param pg_kwargs: pyqtgraph setImage arguments: {'levels': None, 'lut': None,
        'opacity': 1.0}
        :param slice_kwargs: ibllib.atlas.slice arguments: {'volume': 'image', 'mode': 'clip'}
        :return:
        """
        il = ImageLayer(**kwargs)
        if idx is not None:
            self.image_layers.insert(idx, il)
        else:
            self.image_layers.append(il)
        self.fig.addItem(il.image_item)

    def add_mask_layer(self, name='locked', cmap='Blues'):
        colormap = matplotlib.cm.get_cmap(cmap)
        colormap._init()
        # The last one is [0, 0, 0, 0] so remove this
        lut = (colormap._lut * 255).view(np.ndarray)[:-1]
        lut = np.insert(lut, 0, [0, 0, 0, 0], axis=0)
        self.add_image_layer(name=name, pg_kwargs={'lut': lut, 'opacity': 1}, slice_kwargs={
            'volume': 'value', 'region_values': np.zeros_like(self.atlas.regions.id),
            'mode': 'clip', 'bc': None})

    def add_volume_layer(self, volume, name='coverage', cmap='viridis', opacity=0.8,
                         bc=None, levels=None):
        # If there is a layer with the same name remove it
        self.remove_image_layer(name)
        colormap = matplotlib.cm.get_cmap(cmap)
        colormap._init()
        # The last one is [0, 0, 0, 0] so remove this
        lut = (colormap._lut * 255).view(np.ndarray)[:-1]
        lut = np.insert(lut, 0, [0, 0, 0, 0], axis=0)
        if levels is None:
            levels = (0, np.nanmax(volume))
        self.add_image_layer(name=name, pg_kwargs={'lut': lut, 'opacity': opacity, 'levels': levels},
                             slice_kwargs={'volume': 'volume', 'region_values': volume,
                                           'mode': 'clip', 'bc': bc})

    def change_base_layer(self, image='Image'):
        base_layer = self.get_image_layer(name='base')
        if image == 'Image':
            base_layer.slice_kwargs = {'volume': 'image', 'mode': 'clip', 'bc': None}
        elif image == 'Annotation':
            base_layer.slice_kwargs = {'volume': 'annotation', 'mode': 'clip', 'bc': None}

    def remove_image_layer(self, name):
        im_idx = np.where([im.name == name for im in self.image_layers])[0]
        if len(im_idx) != 0:
            il = self.image_layers.pop(im_idx[0])
            self.fig.removeItem(il.image_item)

    def get_image_layer(self, name=None):
        """Returns the either the first image item or the image item with specified name"""
        if not name:
            return self.image_layers[0]
        else:
            im_idx = np.where([im.name == name for im in self.image_layers])[0]
            if len(im_idx) == 1:
                return self.image_layers[im_idx[0]]
            else:
                return None

    def toggle_image_layer(self, name, checked=True):
        layer = self.get_image_layer(name=name)
        if checked:
            self.fig.addItem(layer.image_item)
        else:
            self.fig.removeItem(layer.image_item)


class TopController(BaseController):
    """
    TopView ControllerTopView
    """
    def __init__(self, topview: TopView, qmain: MainWindow, atlas: AllenAtlas):
        super(TopController, self).__init__(atlas, topview.fig_top)
        self.fig = topview.fig_top
        self.atlas = atlas
        self.scatter_layers = []

    def set_top(self):
        self.atlas.compute_surface()
        img = self.atlas.top.transpose()
        img[np.isnan(img)] = np.nanmin(img)  # img has dims ml, ap
        dw, dh = (self.atlas.bc.dxyz[0], self.atlas.bc.dxyz[1])
        wl, hl = (self.atlas.bc.xlim, self.atlas.bc.ylim)
        self.set_image(self.image_layers[0].image_item, img, dw, dh, wl[0], hl[0])

    def set_scatter_layer(self, name, x=None, y=None):
        layer = self.get_scatter_layer(name)
        if x is not None:
            layer.scatter_item.setData(x=x, y=y, **layer.pg_kwargs)
        else:
            layer.scatter_item.setData()

    def add_scatter_layer(self, name, pen='g', brush='g', callback=None):
        self.remove_scatter_layer(name)

        sc = self.add_scatter(name=name, pg_kwargs={'pen': pen, 'brush': brush})
        if callback:
            sc.scatter_item.sigClicked.connect(callback)

    def get_scatter_layer(self, name=None):
        """Returns the either the first image item or the image item with specified name"""
        if not name:
            return self.scatter_layers[0]
        else:
            sc_idx = np.where([sc.name == name for sc in self.scatter_layers])[0]
            if len(sc_idx) == 1:
                return self.scatter_layers[sc_idx[0]]
            else:
                return None


    def add_scatter(self, idx=None, **kwargs):
        """
        :param name: name of the image item to keep track of layers
        :param pg_kwargs: pyqtgraph setImage arguments: {'levels': None, 'lut': None,
        'opacity': 1.0}
        :param slice_kwargs: ibllib.atlas.slice arguments: {'volume': 'image', 'mode': 'clip'}
        :return:
        """
        sc = ScatterLayer(**kwargs)
        if idx is not None:
            self.scatter_layers.insert(idx, sc)
        else:
            self.scatter_layers.append(sc)
        self.fig.addItem(sc.scatter_item)
        return sc

    def remove_scatter_layer(self, name):
        sc_idx = np.where([sc.name == name for sc in self.scatter_layers])[0]
        if len(sc_idx) != 0:
            sc = self.image_layers.pop(sc_idx[0])
            self.fig.removeItem(sc.scatter_item)

    def toggle_scatter_layer(self, name, checked=True):
        layer = self.get_scatter_layer(name=name)
        if layer:
            if checked:
                self.fig.addItem(layer.scatter_item)
            else:
                self.fig.removeItem(layer.scatter_item)



class SliceController(BaseController):

    def __init__(self, sliceview: SliceView, qmain: MainWindow, atlas: AllenAtlas,
                 waxis=None, haxis=None, daxis=None):
        """
        :param waxis: brain atlas axis corresponding to display abscissa (coronal: 0, sagittal: 1)
        :param haxis: brain atlas axis corresponding to display ordinate (coronal: 2, sagittal: 2)
        :param daxis: brain atlas axis corresponding to display abscissa (coronal: 1, sagittal: 0)
        """
        self.view = sliceview
        self.fig = self.view.fig_slice
        super(SliceController, self).__init__(atlas, self.fig)
        self.waxis = waxis
        self.haxis = haxis
        self.daxis = daxis
        self.atlas = atlas
        self.qmain = qmain

    def cursor2xyamp(self, qpoint):
        """
        Extends the superclass method to also get the brain region from the model
        :param qpoint:
        :return:
        """
        iw, ih, w, h, v = super(SliceController, self).cursor2xyamp(qpoint)
        xyz = np.zeros(3)
        xyz[np.array([self.waxis, self.haxis, self.daxis])] = [w, h, self.view.slice_coord]
        try:
            region = self.atlas.regions.get(
                self.atlas.get_labels(xyz, mapping=self.qmain.get_mapping()))
        except ValueError:
            region = None
        return iw, ih, w, h, v, region, xyz

    def set_slice(self):
        # construct the transform matrix image 2 ibl coordinates
        # the ImageLayer object carries slice kwargs and pyqtgraph ImageSet kwargs
        # reversed order so the self.im is set with the base layer
        #
        for layer in reversed(self.image_layers):
            bc = layer.slice_kwargs.get('bc', None) or self.atlas.bc
            dw = bc.dxyz[self.waxis]
            dh = bc.dxyz[self.haxis]
            wl = bc.lim(self.waxis) - dw / 2
            hl = bc.lim(self.haxis) - dh / 2

            _slice = self.atlas.slice(self.view.slice_coord, axis=self.daxis,
                                      mapping=self.qmain.get_mapping(),
                                      **layer.slice_kwargs)
            self.set_image(layer.image_item, _slice, dw, dh, wl[0], hl[0], **layer.pg_kwargs)
            # needs to have it's own bc
            # bc is on the image levell


@dataclass
class ImageLayer:
    """
    Class for keeping track of image layers.
    :param image_item
    :param pg_kwargs: pyqtgraph setImage arguments: {'levels': None, 'lut': None, 'opacity': 1.0}
    :param slice_kwargs: ibllib.atlas.slice arguments: {'volume': 'image', 'mode': 'clip'}
    :param
    """
    name: str = field(default='base')
    image_item: pg.ImageItem = field(default_factory=pg.ImageItem)
    pg_kwargs: dict = field(default_factory=lambda: {})
    slice_kwargs: dict = field(default_factory=lambda: {'volume': 'image', 'mode': 'clip',
                                                        'bc': None})

@dataclass
class ScatterLayer:
    """
    Class for keeping track of scatter layers.
    :param scatter_item
    :param pg_kwargs: pyqtgraph scatter arguments: {'pen': None, 'brush': None, 'size': None}
    :param callback: callback function
    :param
    """
    name: str = field(default='base')
    scatter_item: pg.ImageItem = field(default_factory=pg.ScatterPlotItem)
    pg_kwargs: dict = field(default_factory=lambda: {})
    scatter_kwargs: dict = field(default_factory=lambda: {'x': None, 'y': None})


def view(title=None, lazy=False, res=25):
    """
    """
    qt.create_app()
    av = MainWindow._get_or_create(title=title, lazy=lazy, res=res)
    av.show()
    return av


if __name__ == '__main__':

    app = QtWidgets.QApplication([])
    mainapp = MainWindow()
    # mainapp = MainWindow(offline=True)
    mainapp.show()
    app.exec_()

