from PyQt5 import QtCore, QtWidgets, QtGui
import numpy as np
from iblutil.util import Bunch
from data_exploration_gui import utils


class FilterGroup:
    def __init__(self):

        # button to reset filters to default
        self.reset_button = QtWidgets.QPushButton('Reset Filters')

        # checkboxes for contrasts
        self.contrasts = utils.CONTRAST_OPTIONS
        self.contrast_buttons = QtWidgets.QButtonGroup()
        self.contrast_buttons.setExclusive(False)
        self.contrast_group = QtWidgets.QGroupBox('Stimulus Contrast')
        self.contrast_layout = QtWidgets.QVBoxLayout()
        for val in self.contrasts:
            button = QtWidgets.QCheckBox(str(val * 100))
            button.setCheckState(QtCore.Qt.Checked)
            self.contrast_buttons.addButton(button)
            self.contrast_layout.addWidget(button)

        self.contrast_group.setLayout(self.contrast_layout)

        # button for whether to overlay trial plots
        self.hold_button = QtWidgets.QCheckBox('Hold')
        self.hold_button.setCheckState(QtCore.Qt.Checked)

        # checkboxes for trial options
        self.trial_buttons = QtWidgets.QButtonGroup()
        # Just for now
        self.trial_group = QtWidgets.QGroupBox('Trial Options')
        self.trial_layout = QtWidgets.QVBoxLayout()
        for val in utils.TRIAL_OPTIONS:
            button = QtWidgets.QCheckBox(val)
            if val == utils.TRIAL_OPTIONS[0]:
                button.setCheckState(QtCore.Qt.Checked)
            else:
                button.setCheckState(QtCore.Qt.Unchecked)
            self.trial_buttons.addButton(button)
            self.trial_layout.addWidget(button)

        self.trial_colours = QtWidgets.QVBoxLayout()
        for val in utils.TRIAL_OPTIONS:
            img = QtWidgets.QLabel()
            pix = QtGui.QPixmap(40, 5)
            pix.fill(utils.colours[val])
            img.setPixmap(pix)
            self.trial_colours.addWidget(img)

        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addLayout(self.trial_layout)
        hlayout.addLayout(self.trial_colours)

        self.trial_group.setLayout(hlayout)

        # radio buttons for order options
        self.order_buttons = QtWidgets.QButtonGroup()
        self.order_group = QtWidgets.QGroupBox('Order Trials By:')
        self.order_layout = QtWidgets.QVBoxLayout()
        for val in utils.ORDER_OPTIONS:
            button = QtWidgets.QRadioButton(val)
            if val == utils.ORDER_OPTIONS[0]:
                button.setChecked(True)
            else:
                button.setChecked(False)
            self.order_buttons.addButton(button)
            self.order_layout.addWidget(button)

        self.order_group.setLayout(self.order_layout)

        # radio buttons for sort options
        self.sort_buttons = QtWidgets.QButtonGroup()
        self.sort_group = QtWidgets.QGroupBox('Sort Trials By:')
        self.sort_layout = QtWidgets.QVBoxLayout()
        for val in utils.SORT_OPTIONS:
            button = QtWidgets.QRadioButton(val)
            if val == utils.SORT_OPTIONS[0]:
                button.setChecked(True)
            else:
                button.setChecked(False)
            self.sort_buttons.addButton(button)
            self.sort_layout.addWidget(button)
        self.sort_group.setLayout(self.sort_layout)

        self.event_group = QtWidgets.QGroupBox('Align to:')
        self.event_list = QtGui.QStandardItemModel()
        self.event_combobox = QtWidgets.QComboBox()
        self.event_combobox.setModel(self.event_list)
        self.event_layout = QtWidgets.QVBoxLayout()
        self.event_layout.addWidget(self.event_combobox)
        self.event_group.setLayout(self.event_layout)

        self.behav_group = QtWidgets.QGroupBox('Show behaviour:')
        self.behav_list = QtGui.QStandardItemModel()
        self.behav_combobox = QtWidgets.QComboBox()
        self.behav_combobox.setModel(self.behav_list)
        self.behav_layout = QtWidgets.QVBoxLayout()
        self.behav_layout.addWidget(self.behav_combobox)
        self.behav_group.setLayout(self.behav_layout)


        # Now group everything into one big box
        self.filter_options_group = QtWidgets.QGroupBox()
        group_layout = QtWidgets.QVBoxLayout()
        group_layout.addWidget(self.reset_button)
        group_layout.addWidget(self.contrast_group)
        group_layout.addWidget(self.hold_button)
        group_layout.addWidget(self.trial_group)
        group_layout.addWidget(self.order_group)
        group_layout.addWidget(self.sort_group)
        group_layout.addWidget(self.event_group)
        group_layout.addWidget(self.behav_group)
        self.filter_options_group.setLayout(group_layout)

    def populate(self, event_options, behav_options):
        self.populate_lists(event_options, self.event_list, self.event_combobox)
        self.populate_lists(behav_options, self.behav_list, self.behav_combobox)


    def populate_lists(self, data, list_name, combobox):

        list_name.clear()
        for dat in data:
            item = QtGui.QStandardItem(dat)
            item.setEditable(False)
            list_name.appendRow(item)

        # This makes sure the drop down menu is wide enough to show full length of string
        min_width = combobox.fontMetrics().width(max(data, key=len))
        min_width += combobox.view().autoScrollMargin()
        min_width += combobox.style().pixelMetric(QtGui.QStyle.PM_ScrollBarExtent)
        combobox.view().setMinimumWidth(min_width)

        # Set the default to be the first option
        combobox.setCurrentIndex(0)

    def set_selected_event(self, event):
        for it in range(self.event_combobox.count()):
            if self.event_combobox.itemText(it) == event:
                self.event_combobox.setCurrentIndex(it)

    def get_selected_event(self):
        return self.event_combobox.currentText()

    def get_selected_behaviour(self):
        return self.behav_combobox.currentText()

    def get_selected_contrasts(self):
        contrasts = []
        for button in self.contrast_buttons.buttons():
            if button.isChecked():
                contrasts.append(np.float(button.text()) / 100)

        return contrasts

    def get_selected_trials(self):
        trials = []
        for button in self.trial_buttons.buttons():
            if button.isChecked():
                trials.append(button.text())
        return trials

    def get_selected_order(self):
        return self.order_buttons.checkedButton().text()

    def get_selected_sort(self):
        return self.sort_buttons.checkedButton().text()

    def get_hold_status(self):
        return self.hold_button.isChecked()

    def get_selected_filters(self):
        contrasts = self.get_selected_contrasts()
        order = self.get_selected_order()
        sort = self.get_selected_sort()
        hold = self.get_hold_status()

        return contrasts, order, sort, hold

    def set_sorted_button(self, sort):
        for button in self.sort_buttons.buttons():
            if button.text() == sort:
                button.setChecked(True)
            else:
                button.setChecked(False)

    def reset_filters(self, contrasts=True):

        if contrasts:
            for button in self.contrast_buttons.buttons():
                button.setChecked(True)

        for button in self.trial_buttons.buttons():
            if button.text() == utils.TRIAL_OPTIONS[0]:
                if not button.isChecked():
                    button.setChecked(True)
            else:
                button.setChecked(False)

        for button in self.order_buttons.buttons():
            if button.text() == utils.ORDER_OPTIONS[0]:
                button.setChecked(True)
            else:
                button.setChecked(False)

        for button in self.sort_buttons.buttons():
            if button.text() == utils.SORT_OPTIONS[0]:
                button.setChecked(True)
            else:
                button.setChecked(False)















