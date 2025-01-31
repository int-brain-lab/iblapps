import sys
from collections import deque
from PyQt5.QtWidgets import QTreeView, QWidget, QVBoxLayout, QLineEdit
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QBrush, QColor
from PyQt5.QtCore import Qt, QSortFilterProxyModel, QRegExp, pyqtSignal
from PyQt5.Qt import pyqtSlot

import numpy as np
from iblutil.numerical import ismember
from iblatlas.atlas import BrainRegions
from qt_helpers import qt


class BrainTree(QWidget):
    signal_region_selected = pyqtSignal(int)

    def __init__(self, regions=None):
        super(BrainTree, self).__init__()
        self.regions = regions or BrainRegions()
        self.tree = QTreeView(self)
        layout = QVBoxLayout(self)

        self.lineEditFilterAcronym = QLineEdit(self)
        self.lineEditFilterAcronym.setPlaceholderText('Filter by acronym')
        layout.addWidget(self.lineEditFilterAcronym)

        self.lineEditFilterLevel = QLineEdit(self)
        self.lineEditFilterLevel.setPlaceholderText('Filter by level')
        layout.addWidget(self.lineEditFilterLevel)

        self.lineEditFilterName = QLineEdit(self)
        layout.addWidget(self.lineEditFilterName)
        self.lineEditFilterName.setPlaceholderText('Filter by description')

        layout.addWidget(self.tree)
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(['acronym', 'index', 'allen_id', 'level', 'description'])
        self.tree.header().setDefaultSectionSize(20)

        self.treeProxyAcronym = QSortFilterProxyModel()
        self.treeProxyAcronym.setRecursiveFilteringEnabled(True)
        self.treeProxyAcronym.setSourceModel(self.model)
        self.treeProxyAcronym.setFilterKeyColumn(0)

        self.treeProxyLevel = QSortFilterProxyModel()
        self.treeProxyLevel.setRecursiveFilteringEnabled(True)
        self.treeProxyLevel.setSourceModel(self.treeProxyAcronym)
        self.treeProxyLevel.setFilterKeyColumn(3)

        self.treeProxyName = QSortFilterProxyModel()
        self.treeProxyName.setRecursiveFilteringEnabled(True)
        self.treeProxyName.setSourceModel(self.treeProxyLevel)
        self.treeProxyName.setFilterKeyColumn(4)

        self.tree.setModel(self.treeProxyName)
        self.import_brain_regions()
        self.tree.expandAll()
        self.setWindowTitle('Allen anatomy parcellation')
        self.setGeometry(200, 100, 1200, 800)
        for i in range(4):
            self.tree.resizeColumnToContents(i)
        self.tree.setEditTriggers(QTreeView.NoEditTriggers)
        self.lineEditFilterAcronym.textChanged.connect(self._filter)
        self.lineEditFilterName.textChanged.connect(self._filter)
        self.lineEditFilterLevel.textChanged.connect(self._filter)
        self.tree.selectionModel().selectionChanged.connect(self.handle_selection_change)
        self.show()

    def handle_selection_change(self, selected, deselected):
        indexes = selected.indexes()
        if indexes:
            rid = int(self.tree.model().itemData(indexes[1])[0])
            self.signal_region_selected.emit(rid)

    @pyqtSlot(str)
    def _filter(self, text: str):
        if self.sender() is self.lineEditFilterAcronym:
            self.treeProxyAcronym.setFilterRegExp(QRegExp(text, Qt.CaseInsensitive))
        elif self.sender() is self.lineEditFilterLevel:
            self.treeProxyLevel.setFilterRegExp(QRegExp(text, Qt.CaseInsensitive))
        else:
            self.treeProxyName.setFilterRegExp(QRegExp(text, Qt.CaseInsensitive))
        self.tree.expandAll()

    def import_brain_regions(self):
        rind = np.unique(self.regions.mappings['Allen'])
        iparents = np.zeros_like(rind)
        a, b = ismember(self.regions.parent[rind], self.regions.id[rind])
        iparents[a] = b
        data = [{
            'index': i,
            'parent_index': iparents[i],
            'acronym': self.regions.acronym[i],
            'name': self.regions.name[i],
            'level': self.regions.level[i],
            'rgb': self.regions.rgb[i],
            'allen_id': self.regions.id[i],
        } for i in rind]
        self.model.setRowCount(0)
        root = self.model.invisibleRootItem()
        seen = {}   # List of  QStandardItem
        values = deque(data)
        r = 0
        while values:
            value = values.popleft()
            if value['index'] == 0:
                parent = root
            else:
                pid = value['parent_index']
                if pid not in seen:
                    values.append(value)
                    continue
                parent = seen[pid]
            unique_id = value['index']
            qis = [QStandardItem(str(value[cname])) for cname in ['acronym', 'index', 'allen_id', 'level', 'name']]
            parent.appendRow(qis)
            color = QBrush(QColor(*value['rgb'], 120))
            for q in qis:
                q.setData(color, Qt.BackgroundRole)
            seen[unique_id] = parent.child(parent.rowCount() - 1)
            r += 1


if __name__ == '__main__':
    app = qt.create_app()
    av = BrainTree()
    av.show()
    sys.exit(app.exec_())


# %%

# def iterate_nested_model(model, parent=None):
#     if parent is None:
#         parent = model.index(0, 0)
#     for row in range(model.rowCount(parent)):
#         index = model.index(row, 0, parent)
#         item = model.itemFromIndex(index)
#         print(parent, row, item.text())
#         iterate_nested_model(model, index)
#
# iterate_nested_model(self.model)
