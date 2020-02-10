from PyQt5 import QtCore, QtGui, QtWidgets


class ClusterGroup:

    def __init__(self):

        self.cluster_options = QtWidgets.QGroupBox('Sort Clusters By:')
        self.cluster_option1 = QtWidgets.QRadioButton('cluster no.')
        self.cluster_option2 = QtWidgets.QRadioButton('no. of spikes')
        self.cluster_option3 = QtWidgets.QRadioButton('good units')
        self.cluster_option1.setChecked(True)
        self.group_buttons()

        self.cluster_list = QtWidgets.QListWidget()
        self.cluster_list.SingleSelection

        self.cluster_next_button = QtWidgets.QPushButton('Next')
        self.cluster_next_button.setFixedSize(90, 30)
        self.cluster_previous_button = QtWidgets.QPushButton('Previous')
        self.cluster_previous_button.setFixedSize(90, 30)

        self.cluster_list_group = QtWidgets.QGroupBox()
        self.cluster_list_group.setFixedSize(400, 200)
        self.group_widget()

        self.clust_colour = []
        self.clust_ids = []
        self.clust = []
        self.clust_prev = []

    def group_buttons(self):
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.cluster_option1)
        button_layout.addWidget(self.cluster_option2)
        button_layout.addWidget(self.cluster_option3)
        self.cluster_options.setLayout(button_layout)

    def group_widget(self):
        group_layout = QtWidgets.QGridLayout()
        group_layout.addWidget(self.cluster_options, 0, 0, 1, 3)
        group_layout.addWidget(self.cluster_list, 1, 0, 3, 3)
        group_layout.addWidget(self.cluster_previous_button, 4, 0)
        group_layout.addWidget(self.cluster_next_button, 4, 2)
        self.cluster_list_group.setLayout(group_layout)

    def reset(self):
        self.cluster_list.clear()
        self.clust_colour = []
        self.clust_ids = []
        self.clust = []
        self.clust_prev = []

    def populate(self, clust_ids, clust_colour):
        self.clust_ids = clust_ids
        self.clust_colour = clust_colour
        icon = QtGui.QPixmap(20, 20)
        for idx, val in enumerate(clust_ids):
            item = QtWidgets.QListWidgetItem('Cluster Number ' + str(val))
            icon.fill(self.clust_colour[idx])
            item.setIcon(QtGui.QIcon(icon))
            self.cluster_list.addItem(item)
        self.cluster_list.setCurrentRow(0)

    def update_list_icon(self):
        item = self.cluster_list.item(self.clust_prev)
        icon = QtGui.QPixmap(10, 10)
        icon.fill(self.clust_colour[self.clust_prev])
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
