from PyQt5 import QtGui, QtWidgets
from data_exploration_gui import utils


class ClusterGroup:

    def __init__(self):

        self.cluster_buttons = QtWidgets.QButtonGroup()
        self.cluster_group = QtWidgets.QGroupBox('Sort Clusters By:')
        self.cluster_layout = QtWidgets.QHBoxLayout()
        for i, val in enumerate(utils.SORT_CLUSTER_OPTIONS):
            button = QtWidgets.QRadioButton(val)
            if i == 0:
                button.setChecked(True)
            else:
                button.setChecked(False)
            self.cluster_buttons.addButton(button)
            self.cluster_layout.addWidget(button)

        self.cluster_group.setLayout(self.cluster_layout)

        self.cluster_colours = QtWidgets.QGroupBox()
        h_layout_col = QtWidgets.QHBoxLayout()
        h_layout_lab = QtWidgets.QHBoxLayout()
        for val in utils.UNIT_OPTIONS:
            img = QtWidgets.QLabel()
            label = QtWidgets.QLabel(val)
            pix = QtGui.QPixmap(40, 5)
            pix.fill(utils.colours[val])
            img.setPixmap(pix)
            h_layout_lab.addWidget(img)
            h_layout_col.addWidget(label)

        v_layout = QtWidgets.QVBoxLayout()
        v_layout.addLayout(h_layout_lab)
        v_layout.addLayout(h_layout_col)

        self.cluster_colours.setLayout(v_layout)

        self.cluster_list = QtWidgets.QListWidget()

        self.cluster_next_button = QtWidgets.QPushButton('Next')
        self.cluster_next_button.setFixedSize(90, 30)
        self.cluster_previous_button = QtWidgets.QPushButton('Previous')
        self.cluster_previous_button.setFixedSize(90, 30)

        self.cluster_list_group = QtWidgets.QGroupBox()
        #self.cluster_list_group.setFixedSize(400, 300)

        group_layout = QtWidgets.QGridLayout()
        group_layout.addWidget(self.cluster_group, 0, 0, 1, 3)
        group_layout.addWidget(self.cluster_colours, 1, 0, 1, 3)
        group_layout.addWidget(self.cluster_list, 2, 0, 3, 3)
        group_layout.addWidget(self.cluster_previous_button, 5, 0)
        group_layout.addWidget(self.cluster_next_button, 5, 2)
        self.cluster_list_group.setLayout(group_layout)

        self.reset()

    def reset(self):
        self.cluster_list.clear()
        self.clust_colour_ks = None
        self.clust_colour_ibl = None
        self.clust_ids = None
        self.clust = None
        self.clust_prev = None

    def populate(self, clust_ids, clust_colour_ks, clust_colour_ibl):
        self.clust_ids = clust_ids
        self.clust_colour_ks = clust_colour_ks
        self.clust_colour_ibl = clust_colour_ibl

        for idx, val in enumerate(clust_ids):
            item = QtWidgets.QListWidgetItem('Cluster Number ' + str(val))
            icon = utils.get_icon(self.clust_colour_ibl[idx], self.clust_colour_ks[idx], 20)
            item.setIcon(QtGui.QIcon(icon))
            self.cluster_list.addItem(item)
        self.cluster_list.setCurrentRow(0)

    def update_list_icon(self):
        item = self.cluster_list.item(self.clust_prev)
        icon = utils.get_icon(self.clust_colour_ibl[self.clust_prev],
                              self.clust_colour_ks[self.clust_prev], 12)
        item.setIcon(QtGui.QIcon(icon))
        item.setText('  Cluster Number ' + str(self.clust_ids[self.clust_prev]))

    def update_current_row(self, clust):
        self.clust = clust
        self.cluster_list.setCurrentRow(self.clust)

    def on_next_cluster_clicked(self):
        self.clust += 1
        self.cluster_list.setCurrentRow(self.clust)
        return self.clust

    def on_previous_cluster_clicked(self):
        self.clust -= 1
        self.cluster_list.setCurrentRow(self.clust)
        return self.clust

    def on_cluster_list_clicked(self):
        self.clust = self.cluster_list.currentRow()
        return self.clust

    def update_cluster_index(self):
        self.clust_prev = self.clust
        return self.clust_prev

    def initialise_cluster_index(self):
        self.clust = 0
        self.clust_prev = 0

        return self.clust, self.clust_prev

    def get_selected_sort(self):
        return self.cluster_buttons.checkedButton().text()
