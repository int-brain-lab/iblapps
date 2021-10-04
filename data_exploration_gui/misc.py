from PyQt5 import QtWidgets
from data_exploration_gui import utils

class MiscGroup:
    def __init__(self):

        self.qc_group = QtWidgets.QGroupBox()
        self.dlc_warning_label = QtWidgets.QLabel()
        self.sess_qc_group = QtWidgets.QHBoxLayout()
        self.clust_qc_group = QtWidgets.QHBoxLayout()

        self.sess_qc_labels = []
        self.clust_qc_labels = []

        for qc in utils.SESS_QC:
            qc_label = QtWidgets.QLabel(f'{qc}:')
            self.sess_qc_group.addWidget(qc_label)
            self.sess_qc_labels.append(qc_label)

        for qc in utils.CLUSTER_QC:
            qc_label = QtWidgets.QLabel(f'{qc}: ')
            self.clust_qc_group.addWidget(qc_label)
            self.clust_qc_labels.append(qc_label)

        vlayout = QtWidgets.QVBoxLayout()
        vlayout.addWidget(self.dlc_warning_label)
        vlayout.addLayout(self.sess_qc_group)
        vlayout.addLayout(self.clust_qc_group)
        self.qc_group.setLayout(vlayout)

    def set_sess_qc_text(self, data):
        for label in self.sess_qc_labels:
            title = label.text().split(':')[0]
            text = title + ': ' + str(data[title])
            label.setText(text)

    def set_clust_qc_text(self, data):
        for label in self.clust_qc_labels:
            title = label.text().split(':')[0]
            text = title + ': ' + str(data[title])
            label.setText(text)

    def set_dlc_label(self, aligned):
        if not aligned:
            self.dlc_warning_label.setText(utils.dlc_warning)