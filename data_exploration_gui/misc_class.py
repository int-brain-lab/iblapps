from PyQt5 import QtCore, QtGui, QtWidgets

import pandas as pd


class MiscGroup:
    def __init__(self):

        folder_prompt = QtWidgets.QLabel('Select Folder')
        self.folder_line = QtWidgets.QLineEdit()
        self.folder_button = QtWidgets.QToolButton()
        self.folder_button.setText('...')

        folder_layout = QtWidgets.QHBoxLayout()
        folder_layout.addWidget(folder_prompt)
        folder_layout.addWidget(self.folder_line)
        folder_layout.addWidget(self.folder_button)

        self.folder_group = QtWidgets.QGroupBox()
        self.folder_group.setLayout(folder_layout)

        clust_list1_title = QtWidgets.QLabel('Go Cue Aligned Clusters')
        self.clust_list1 = QtWidgets.QListWidget()
        self.remove_clust_button1 = QtWidgets.QPushButton('Remove')
        
        clust_list2_title = QtWidgets.QLabel('Feedback Aligned Clusters')
        self.clust_list2 = QtWidgets.QListWidget()
        self.remove_clust_button2 = QtWidgets.QPushButton('Remove')
        self.save_clust_button = QtWidgets.QPushButton('Save Clusters')
        
        clust_interest_layout = QtWidgets.QGridLayout()
        clust_interest_layout.addWidget(clust_list1_title, 0 , 0)
        clust_interest_layout.addWidget(self.remove_clust_button1, 0, 1)
        clust_interest_layout.addWidget(self.clust_list1, 1, 0, 1, 2)
        clust_interest_layout.addWidget(clust_list2_title, 2, 0)
        clust_interest_layout.addWidget(self.remove_clust_button2, 2, 1)
        clust_interest_layout.addWidget(self.clust_list2, 3, 0, 1, 2)
        clust_interest_layout.addWidget(self.save_clust_button, 4, 0)

        self.clust_interest = QtWidgets.QGroupBox()
        self.clust_interest.setLayout(clust_interest_layout)
        self.clust_interest.setFixedSize(250, 350)

        self.terminal = QtWidgets.QTextBrowser() 
        self.terminal_button = QtWidgets.QPushButton('Clear')
        self.terminal_button.setFixedSize(70, 30)
        self.terminal_button.setParent(self.terminal)
        self.terminal_button.move(580, 0)
        
    
    def on_save_button_clicked(self, gui_path):
        save_file = QtWidgets.QFileDialog()
        save_file.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        file = save_file.getSaveFileName(None, 'Save File', gui_path, 'CSV (*.csv)')
        print(file[0])
        if file[0]:
            save_clusters = pd.DataFrame(columns=['go_Cue_aligned', 'feedback_aligned'])

            go_cue_clusters = []
            for idx in range(self.clust_list1.count()):
                cluster_no = self.clust_list1.item(idx).text()
                go_cue_clusters.append(int(cluster_no[8:]))

            go_cue = pd.Series(go_cue_clusters)
            save_clusters['go_Cue_aligned'] = go_cue

            feedback_clusters = []
            for idx in range(self.clust_list2.count()):
                cluster_no = self.clust_list2.item(idx).text()
                feedback_clusters.append(int(cluster_no[8:]))

            feedback = pd.Series(feedback_clusters)
            save_clusters['feedback_aligned'] = feedback

            save_clusters.to_csv(file[0], index=None, header=True) 

            return file[0]

        

