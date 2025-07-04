from qtpy import QtCore, QtGui, QtWidgets
from atlaselectrophysiology.qt_utils.widgets import PopupWindow
import numpy as np
from pathlib import Path
import iblatlas.atlas as atlas
from one.alf.io import load_file_content

PLUGIN_NAME = 'Region Tree'

def setup(parent):

    parent.plugins[PLUGIN_NAME] = dict()

    action = QtWidgets.QAction(PLUGIN_NAME, parent)
    action.setShortcut('Shift+I')
    action.triggered.connect(lambda: callback(parent))
    parent.plugin_options.addAction(action)


def load_allen_csv():
    """
    Load in allen csv file
    :return allen: dataframe containing all information in csv file
    :type: pd.Dataframe
    """
    allen_path = Path(Path(atlas.__file__).parent, 'allen_structure_tree.csv')
    allen = load_file_content(allen_path)

    allen = allen.drop([0]).reset_index(drop=True)

    # Find the parent path of each structure by removing the structure id from path
    def parent_path(struct_path):
        return struct_path.rsplit('/', 2)[0] + '/'

    allen['parent_path'] = allen['structure_id_path'].apply(parent_path)

    return allen


def get_region_description(allen_tree, region_idx):
    struct_idx = np.where(allen_tree['id'] == region_idx)[0][0]
    description = None
    region_lookup = (allen_tree['acronym'][struct_idx] + ': ' + allen_tree['name'][struct_idx])

    if region_lookup == 'void: void':
        region_lookup = 'root: root'

    if not description:
        description = region_lookup + '\nNo information available for this region'
    else:
        description = region_lookup + '\n' + description

    return description, region_lookup



def callback(parent):

    if parent.hover_region:
        if parent.hover_config:
            items = parent.shank_items[parent.hover_shank][parent.hover_config]
        else:
            items = parent.shank_items[parent.hover_shank]

        idx = np.where(items.hist_regions['left'] == parent.hover_region)[0]
        if not np.any(idx):
            idx = np.where(items.hist_regions['right'] == parent.hover_region)[0]
        if not np.any(idx):
            idx = np.array([0])

    parent.label_win = RegionLookup._get_or_create('Structure Info', parent=parent)

    if idx:
        if parent.hover_config:
            region = parent.loaddata.shanks[parent.hover_shank][parent.hover_config].loaders['align'].align.ephysalign.region_id[idx[0]][0]
        else:
            region = parent.loaddata.shanks[parent.hover_shank].loaders['align'].align.ephysalign.region_id[idx[0]][0]

        parent.label_win.label_selected(region)



class RegionLookup(PopupWindow):

    def __init__(self, title, parent=None):
        self.allen = load_allen_csv()
        super().__init__(title, parent=parent, size=(500, 700), graphics=False)

    def setup(self):

        self.struct_list = QtGui.QStandardItemModel()
        self.struct_view = QtWidgets.QTreeView()
        self.struct_view.setModel(self.struct_list)
        self.struct_view.setHeaderHidden(True)
        self.struct_view.clicked.connect(self.label_pressed)

        icon = QtGui.QPixmap(20, 20)

        def _create_item(idx):
            item = QtGui.QStandardItem(self.allen['acronym'][idx] + ': ' + self.allen['name'][idx])
            icon.fill(QtGui.QColor('#' + self.allen['color_hex_triplet'][idx]))
            item.setIcon(QtGui.QIcon(icon))
            item.setAccessibleText(str(self.allen['id'][idx]))
            item.setEditable(False)

            return item

        unique_levels = np.unique(self.allen['depth']).astype(int)
        parent_info = {}
        # Create root
        idx = np.where(self.allen['depth'] == unique_levels[0])[0]
        item = _create_item(idx[0])
        self.struct_list.appendRow(item)
        parent_info.update({self.allen['structure_id_path'][idx[0]]: item})
        # Create rest of tree
        for level in unique_levels[1:]:
            idx_levels = np.where(self.allen['depth'] == level)[0]
            for idx in idx_levels:
                parent_item = parent_info[self.allen['parent_path'][idx]]
                item = _create_item(idx)
                parent_item.appendRow(item)
                parent_info.update({self.allen['structure_id_path'][idx]: item})

        self.struct_description = QtWidgets.QTextEdit()

        self.layout.addWidget(self.struct_view)
        self.layout.addWidget(self.struct_description)
        self.layout.setRowStretch(0, 7)
        self.layout.setRowStretch(1, 3)


    def label_pressed(self, item):
        idx = int(item.model().itemFromIndex(item).accessibleText())
        description, lookup = get_region_description(self.allen, idx)
        item = self.struct_list.findItems(lookup, flags=QtCore.Qt.MatchRecursive)
        model_item = self.struct_list.indexFromItem(item[0])
        self.struct_view.setCurrentIndex(model_item)
        self.struct_description.setText(description)

    def label_selected(self, region):

        description, lookup = get_region_description(self.allen, region)
        item = self.struct_list.findItems(lookup, flags=QtCore.Qt.MatchRecursive)
        model_item = self.struct_list.indexFromItem(item[0])
        self.struct_view.collapseAll()
        self.struct_view.scrollTo(model_item)
        self.struct_view.setCurrentIndex(model_item)
        self.struct_description.setText(description)


