import oursin as urchin
from one.api import ONE
from iblatlas.atlas import AllenAtlas, Insertion
from ibllib.plots import color_cycle

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QModelIndex, QObject, QVariant
from PyQt5.QtGui import QPalette, QColor
from iblqt.core import DataFrameTableModel, DataFrame

import time
import pandas as pd
import numpy as np

from ibllib.misc import qt
import matplotlib.pyplot as mpl  # noqa  # This is needed to make qt show properly :/

one = ONE()
ba = AllenAtlas()


class TrajectoryViewer(QtWidgets.QMainWindow):

    @staticmethod
    def _instances():
        app = QtWidgets.QApplication.instance()
        return [w for w in app.topLevelWidgets() if isinstance(w, TrajectoryViewer)]

    @staticmethod
    def _get_or_create(title=None, **kwargs):
        av = next(filter(lambda e: e.isVisible() and e.windowTitle() == title,
                         TrajectoryViewer._instances()), None)
        if av is None:
            av = TrajectoryViewer(**kwargs)
            av.setWindowTitle(title)
        return av


    _columnOn = dict()
    _rowOn = dict()
    probes = None
    clusters = None
    data = dict()
    regions = list()

    def __init__(self):
        super(TrajectoryViewer, self).__init__()

        self.bool_columns = ['hist', 'micro', 'picks']

        # Define table model & view
        self.tableModel = BoolDataFrameTableModel(self, bool_columns=self.bool_columns, alpha=0.3)
        self.tableView = QtWidgets.QTableView(self)
        self.tableView.setModel(self.tableModel)
        self.tableView.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.tableView.horizontalHeader().setSectionsMovable(False)
        self.tableView.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)

        # Define colors for highlighted cells
        p = self.tableView.palette()
        p.setColor(QPalette.Highlight, Qt.gray)
        p.setColor(QPalette.HighlightedText, Qt.white)
        self.tableView.setPalette(p)

        self.setCentralWidget(self.tableView)

        # Sizing
        self.setMinimumSize(800, 400)

        # Define slots and connections
        self.tableView.clicked.connect(self.on_item_clicked)
        self.tableView.horizontalHeader().sectionClicked.connect(self.on_column_header_clicked)
        self.tableView.verticalHeader().sectionClicked.connect(self.on_row_header_clicked)
        self.tableModel.dataChanged.connect(self.on_data_changed)

    def setup(self):

        urchin.setup()
        urchin.ccf25.load()
        time.sleep(10)
        urchin.ccf25.grey.set_visibility(True)
        urchin.ccf25.grey.set_material('transparent-unlit')
        urchin.ccf25.grey.set_color('#000000')
        urchin.ccf25.grey.set_alpha(0.1)
        time.sleep(4)

    def add_regions(self, regions):

        if len(self.regions) > 0:
            urchin.ccf25.set_visibilities(self.regions, False, urchin.utils.Side.LEFT)
            self.regions = []

        if len(regions) > 0:
            regions = [r for r in regions if r not in ['void', 'root']]
            self.regions = urchin.ccf25.get_areas(regions)
            urchin.ccf25.set_visibilities(self.regions, True, urchin.utils.Side.LEFT)
            urchin.ccf25.set_materials(self.regions, 'transparent-unlit', 'left')
            urchin.ccf25.set_alphas(self.regions, 0.25, 'left')

    def load_subject(self, subject):

        self.data = self.getData(subject)
        dataFrame = self.prepareDataFrame()
        self.tableModel.setDataFrame(dataFrame)

        for col in self.bool_columns:
            self._columnOn[col] = True

        for row in dataFrame.key.values:
            self._rowOn[row] = True

        self.tableView.setColumnHidden(self.tableModel.dataFrame.columns.get_loc('color'), True)
        self.tableView.setColumnHidden(self.tableModel.dataFrame.columns.get_loc('key'), True)

        self.on_data_changed(0, 0)

    def prepareDataFrame(self):

        dataFrame = pd.DataFrame()
        dataFrame['key'] = list(self.data.keys())
        dataFrame['eid'] = [v['eid'] for _, v in self.data.items()]
        dataFrame['insertion'] = [k[:-1] for k in list(self.data.keys())]
        dataFrame['shank'] = [v['name'] for _, v in self.data.items()]
        dataFrame['micro'] = [True] * len(self.data)
        dataFrame['hist'] = [True] * len(self.data)
        dataFrame['picks'] = [True] * len(self.data)
        dataFrame['pid'] = [v['pid'] for _, v in self.data.items()]
        dataFrame['color'] = [v['hist']['color'] for _, v in self.data.items()]
        dataFrame = dataFrame.sort_values(['insertion', 'shank']).reset_index(drop=True)

        return dataFrame

    def getData(self, subject):

        def _get_key(ins):
            key = f"{ins['session_info']['subject']}_{ins['session_info']['start_time'][0:10]}_00" \
                  f"{ins['session_info']['number']}_{ins['name']}"
            return key

        insertions = one.alyx.rest('insertions', 'list', subject=subject)
        traj_micro = one.alyx.rest('trajectories', 'list', subject=subject, provenance='Micro-manipulator')
        traj_hist = one.alyx.rest('trajectories', 'list', subject=subject, provenance='Histology track')

        # Group shanks according to their parents and get a colour for each probe

        shank2probe = {}
        for ins in insertions:
            key = _get_key(ins)[:-1] if len(ins['name']) == 8 else _get_key(ins)
            shank2probe[ins['id']] = key
        probe2shank = {}
        for i, probe in enumerate(sorted(set(shank2probe.values()))):
            probe2shank[probe] = color_cycle(i)

        # Want to group by session and probe name
        insertion_info = dict()
        for i, ins in enumerate(insertions):
            micro = next((tr for tr in traj_micro if tr['probe_insertion'] == ins['id']), None)
            hist = next((tr for tr in traj_hist if tr['probe_insertion'] == ins['id']), None)
            # TODO better error handling
            if hist is None:
                continue

            picks = np.array(ins['json']['xyz_picks']) / 1e6

            micro_info = dict()
            if micro is not None:
                ins_micro = Insertion.from_dict(micro, brain_atlas=ba)
                mlapdv_micro = ba.xyz2ccf(ins_micro.tip)
                micro_info['position'] = [mlapdv_micro[1], mlapdv_micro[0], mlapdv_micro[2]]
                micro_info['angle'] = [90 - ins_micro.phi, 90 + ins_micro.theta, ins_micro.beta]
                micro_info['color'] = probe2shank[shank2probe[ins['id']]]

            # Update the hist info
            #ins_hist = Insertion.from_dict(hist, brain_atlas=ba)
            ins_hist = Insertion.from_track(picks, brain_atlas=ba)
            mlapdv_hist = ba.xyz2ccf(ins_hist.tip)
            hist_info = dict()
            hist_info['position'] = [mlapdv_hist[1], mlapdv_hist[0], mlapdv_hist[2]]
            hist_info['angle'] = [90 - ins_hist.phi, 90 + ins_hist.theta, ins_hist.beta]
            hist_info['color'] = probe2shank[shank2probe[ins['id']]]

            mlapdv_picks = ba.xyz2ccf(picks)
            picks_info = dict()
            picks_info['position'] = list(np.c_[mlapdv_picks[:, 1], mlapdv_picks[:, 0], mlapdv_picks[:, 2]])
            picks_info['color'] = [probe2shank[shank2probe[ins['id']]]] * len(mlapdv_picks)

            insertion_info[_get_key(ins)] = {'micro': micro_info, 'hist': hist_info, 'picks': picks_info,
                                             'eid': ins['session_info']['id'], 'pid': ins['id'], 'name': ins['name']}

        return insertion_info

    def on_column_header_clicked(self, section):
        key = self.tableModel.dataFrame.columns[section]
        if key not in self.bool_columns:
            return

        self._columnOn[key] = not self._columnOn[key]
        for row, _ in self._rowOn.items():
            self._rowOn[row] = self._columnOn[key]

        self.tableModel.setColumn(section=section, value=self._columnOn[key])

    def on_row_header_clicked(self, section):

        key = self.tableModel.dataFrame.iloc[section].key
        self._rowOn[key] = not self._rowOn[key]
        self.tableModel.setRow(section=section, value=self._rowOn[key])

    def on_item_clicked(self, index):
        row = index.row()
        column = index.column()
        column_name = self.tableModel.dataFrame.columns[column]
        if column_name in self.bool_columns:
            return

        if column_name in ['shank', 'pid']:
            self.on_row_header_clicked(row)
        else:
            value = self.tableModel.dataFrame.iloc[row, column]
            rows = self.tableModel.dataFrame.index[self.tableModel.dataFrame.iloc[:, column] == value].to_list()
            row_keys = [self.tableModel.dataFrame.iloc[r].key for r in rows]
            key = self.tableModel.dataFrame.iloc[row].key
            self._rowOn[key] = not self._rowOn[key]
            for rk in row_keys:
                self._rowOn[rk] = self._rowOn[key]
            self.tableModel.setRows(sections=rows, value=self._rowOn[key])

    def on_data_changed(self, _, __):

        if self.probes:
            for p in self.probes:
                p.delete()
            #urchin.probes.delete(self.probes)
        if self.clusters:
            urchin.particles.clear()

        positions = []
        angles = []
        colors = []

        for _, info in self.tableModel.dataFrame.iterrows():
            key = info.key
            for prov in ['micro', 'hist']:
                if info[prov]:
                    if self.data[key][prov]:
                        positions.append(self.data[key][prov]['position'])
                        angles.append(self.data[key][prov]['angle'])
                        colors.append(self.data[key][prov]['color'])

        self.probes = urchin.probes.create(len(positions))
        urchin.probes.set_positions(self.probes, positions)
        urchin.probes.set_angles(self.probes, angles)
        urchin.probes.set_colors(self.probes, colors)

        positions = []
        colors = []
        for _, info in self.tableModel.dataFrame.iterrows():
            key = info.key
            if info['picks']:
                positions += self.data[key]['picks']['position']
                colors += self.data[key]['picks']['color']

        self.clusters = urchin.particles.ParticleSystem(n=len(positions))
        self.clusters.set_positions(positions)
        self.clusters.set_colors(colors)


class BoolDataFrameTableModel(DataFrameTableModel):

    def __init__(
        self,
        parent: QObject | None = None,
        dataFrame: DataFrame | None = None,
        alpha: int = 255,
        bool_columns: list | None = None
    ):
        """
        Initialize the ColoredDataFrameTableModel.

        Parameters
        ----------
        parent : QObject, optional
            The parent object.
        dataFrame : DataFrame, optional
            The Pandas DataFrame to be represented by the model.
        colormap : str
            The colormap to be used. Can be the name of a valid colormap from matplotlib or colorcet.
        alpha : int
            The alpha value of the colormap. Must be between 0 and 255.
        *args : tuple
            Positional arguments passed to the parent class.
        **kwargs : dict
            Keyword arguments passed to the parent class.

        """
        super().__init__(parent=parent, dataFrame=dataFrame)
        self.bool_columns = bool_columns
        self._alpha = alpha

    def flags(self, index: QModelIndex):
        # Enable checkbox interaction for column 1
        column = index.column()
        if self._dataFrame.columns[column] in self.bool_columns:  # Checkboxes in column
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable
        else:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return QVariant()

        row = index.row()
        column = index.column()

        if (
            role in (Qt.ItemDataRole.BackgroundRole, Qt.ItemDataRole.ForegroundRole)
            and index.isValid()
        ):
            if role == Qt.ItemDataRole.BackgroundRole:
                if not self._dataFrame.iloc[row, self._dataFrame.columns.get_indexer(self.bool_columns)].values.sum():
                    r, g, b = (0.827, 0.827, 0.827)
                else:
                    r, g, b = self._dataFrame.iloc[row, self._dataFrame.columns.get_loc('color')]
                return QColor.fromRgb(int(r * 255), int(g * 255), int(b * 255), int(self._alpha * 255))

        # Display checkboxes for column 1 (or any column you choose)
        if role == Qt.CheckStateRole and self._dataFrame.columns[column] in self.bool_columns:  # Assuming 2nd column for checkboxes
            # Return Qt.Checked or Qt.Unchecked based on the boolean value
            return Qt.Checked if self._dataFrame.iloc[row, column] else Qt.Unchecked

        if role == Qt.DisplayRole:
            # Return the data for display in the rest of the columns
            return self._dataFrame.iloc[row, column]

        return QVariant()

    def setData(self, index: QModelIndex, value, role=Qt.EditRole):

        if not index.isValid():
            return False

        row = index.row()
        column = index.column()

        # Handle the checkbox change in column 1
        if role == Qt.CheckStateRole and self._dataFrame.columns[column] in self.bool_columns:
            print('in the place where I should change')
            # Toggle the boolean value
            self._dataFrame.iloc[row, column] = value == Qt.Checked
            # Emit dataChanged signal to update the view
            self.dataChanged.emit(index, index, [Qt.CheckStateRole])
            return True

        return False

    def setColumn(self, section: int, value: bool):
        self._dataFrame.iloc[:, section] = value
        self.dataChanged.emit(self.index(0, section), self.index(self.rowCount(), section), [Qt.CheckStateRole])

    def setRow(self, section: int, value: bool):
        self._dataFrame.iloc[section, self._dataFrame.columns.get_indexer(self.bool_columns)] = value
        self.dataChanged.emit(self.index(section, 0), self.index(section, self.columnCount()), [Qt.CheckStateRole])

    def setRows(self, sections: list, value: bool):
        self._dataFrame.iloc[sections, self._dataFrame.columns.get_indexer(self.bool_columns)] = value
        self.dataChanged.emit(self.index(sections[0], 0), self.index(sections[0], self.columnCount()), [Qt.CheckStateRole])


def view(title=None):
    qt.create_app()
    av = TrajectoryViewer._get_or_create(title=title)
    av.show()
    return av


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    mainapp = TrajectoryViewer()
    mainapp.show()
    app.exec_()


