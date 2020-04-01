from PyQt5 import QtCore, QtWidgets, QtGui
import numpy as np
import brainbox as bb


class FilterGroup:

    def __init__(self):

        self.reset_filter_button = QtWidgets.QPushButton('Reset Filters')

        self.contrast_options_text = QtWidgets.QLabel('Stimulus Contrast')
        self.contrasts = [1, 0.25, 0.125, 0.0625, 0]
        self.contrast_options = QtWidgets.QListWidget()
        for val in self.contrasts:
            item = QtWidgets.QListWidgetItem(str(val * 100) + ' %')
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.contrast_options.addItem(item)

        self.hold_button = QtWidgets.QCheckBox('Hold')
        self.hold_button.setCheckState(QtCore.Qt.Checked)

        self.filter_buttons = QtWidgets.QButtonGroup()
        self.filter_group = QtWidgets.QGroupBox('Filter Options')
        self.filter_layout = QtWidgets.QVBoxLayout()
        self.filter_layout.setSpacing(5)
        #self.filter_buttons.setExclusive(False)
        filter_options = ['all', 'correct', 'incorrect', 'left', 'right', 'left correct', 'left incorrect', 'right correct', 'right incorrect']
        for i, val in enumerate(filter_options):
            button = QtWidgets.QCheckBox(val)
            if val == 'all':
                button.setCheckState(QtCore.Qt.Checked)
            else:
                button.setCheckState(QtCore.Qt.Unchecked)
            self.filter_buttons.addButton(button, id=i)
            self.filter_layout.addWidget(button)
        
        self.filter_group.setLayout(self.filter_layout)

        self.trial_buttons = QtWidgets.QButtonGroup()
        self.trial_group = QtWidgets.QGroupBox('Sort Trials By:')
        self.trial_layout = QtWidgets.QHBoxLayout()
        trial_options = ['trial no.', 'correct vs incorrect', 'left vs right', 'correct vs incorrect and left vs right']
        for i, val in enumerate(trial_options):
            button = QtWidgets.QRadioButton(val)
            if i == 0:
                button.setChecked(True)
            else:
                button.setChecked(False)
            self.trial_buttons.addButton(button, id = i)
            self.trial_layout.addWidget(button)
        
        self.trial_group.setLayout(self.trial_layout)

       # Print out no. of trials for each filter condition
        self.ntrials_text = QtWidgets.QLabel('No. of trials  = ')
        
        self.filter_options_group = QtWidgets.QGroupBox()
        self.group_filter_widget()
        self.filter_options_group.setFixedSize(250, 380)

    def group_filter_widget(self):
        group_layout = QtWidgets.QVBoxLayout()
        group_layout.addWidget(self.reset_filter_button)
        group_layout.addWidget(self.contrast_options_text)
        group_layout.addWidget(self.contrast_options)
        group_layout.addWidget(self.hold_button)
        group_layout.addWidget(self.filter_group)
        group_layout.addWidget(self.ntrials_text)
        self.filter_options_group.setLayout(group_layout)

    def get_checked_contrasts(self):

        '''
        Finds the contrast options that are selected. Called by on_contrast_list_changed in gui_main.

        Returns
        ----------
        
        stim_contrast: list
            A list of the contrast options that are selected

        '''

        stim_contrast = []
        for idx in range(self.contrast_options.count()):
            if self.contrast_options.item(idx).checkState() == QtCore.Qt.Checked:
                stim_contrast.append(self.contrasts[idx])
        
        return stim_contrast

    
    def compute_and_sort_trials(self, stim_contrast):
        #Precompute trials for a given contrast set        
        #All

        all_trials = bb.core.Bunch()
        all_trials['colour'] = QtGui.QColor('#808080')
        all_trials['fill'] = QtGui.QColor('#808080')
        all_trials['linestyle'] = QtGui.QPen(QtCore.Qt.SolidLine)
        trials_id = np.empty(0,int)
        idx= np.empty(0, int)
        for c in stim_contrast:
                idx = np.where((self.trials['contrastLeft'] == c) | (self.trials['contrastRight'] == c))[0]
                trials_id = np.append(trials_id, idx)
        trials_id = np.setdiff1d(trials_id, self.nan_trials)
        trials_no = bb.core.Bunch()
        trials_no['trials'] = trials_id
        trials_no['lines'] = []
        trials_no['linecolours'] = []
        trials_no['text'] = []
        all_trials['trial no.'] = trials_no


        trials_ic = bb.core.Bunch()
        correct = np.intersect1d(trials_id, self.correct_idx)
        incorrect = np.intersect1d(trials_id, self.incorrect_idx)
        trials_ic['trials'] = np.append(correct, incorrect)
        trials_ic['lines'] = [[0, len(correct)], [len(correct), len(trials_ic['trials'])]]
        trials_ic['linecolours'] = [QtGui.QColor('#1f77b4'), QtGui.QColor('#d62728')]
        trials_ic['text'] = ['correct', 'incorrect']
        all_trials['correct vs incorrect'] = trials_ic

        trials_lf = bb.core.Bunch()
        left = np.intersect1d(trials_id, self.left_idx)
        right = np.intersect1d(trials_id, self.right_idx)
        trials_lf['trials'] = np.append(left, right)
        trials_lf['lines'] = [[0, len(left)], [len(left), len(trials_lf['trials'])]]
        trials_lf['linecolours'] = [QtGui.QColor('#2ca02c'), QtGui.QColor('#bcbd22')]
        trials_lf['text'] = ['left', 'right']
        all_trials['left vs right'] = trials_lf

        trials_iclf = bb.core.Bunch()
        correct_right = np.intersect1d(trials_id, self.correct_right_idx)
        correct_left = np.intersect1d(trials_id, self.correct_left_idx)
        incorrect_right = np.intersect1d(trials_id, self.incorrect_right_idx)
        incorrect_left = np.intersect1d(trials_id, self.incorrect_left_idx)
        trials_iclf['trials'] = np.concatenate((correct_left, correct_right, incorrect_left, incorrect_right))
        trials_iclf['lines'] = [[0, len(correct_left)], [len(correct_left), len(correct_left) 
            + len(correct_right)], [len(correct_left) + len(correct_right), len(correct_left) 
                + len(correct_right) + len(incorrect_left)],[len(correct_left) + len(correct_right)
                     + len(incorrect_left), len(trials_iclf['trials'])]]
        trials_iclf['linecolours'] = [QtGui.QColor('#17becf'), QtGui.QColor('#9467bd'), QtGui.QColor('#8c564b'), QtGui.QColor('#ff7f0e')]
        trials_iclf['text'] = ['left correct', 'right correct', 'left incorrect', 'right incorrect']
        all_trials['correct vs incorrect and left vs right'] = trials_iclf


        #Correct
        correct_trials = bb.core.Bunch()
        correct_trials['colour'] = QtGui.QColor('#1f77b4')
        correct_trials['fill'] = QtGui.QColor('#1f77b4')
        correct_trials['linestyle'] = QtGui.QPen(QtCore.Qt.SolidLine)
        trials_id = np.empty(0,int)
        idx= np.empty(0, int)
        for c in stim_contrast:
            idx = np.where(((self.trials['contrastLeft'] == c) | (self.trials['contrastRight'] == c)) & (self.trials['feedbackType'] == 1))[0]
            trials_id = np.append(trials_id, idx)
        trials_id = np.setdiff1d(trials_id, self.nan_trials)
    

        trials_no = bb.core.Bunch()
        trials_no['trials'] = trials_id
        trials_no['lines'] = []
        trials_no['linecolours'] = []
        trials_no['text'] = []
        correct_trials['trial no.'] = trials_no


        trials_ic = bb.core.Bunch()
        trials_ic['trials'] = trials_id
        trials_ic['lines'] = [[0, len(trials_ic['trials'])]]
        trials_ic['linecolours'] = [QtGui.QColor('#1f77b4')]
        trials_ic['text'] = ['correct']
        correct_trials['correct vs incorrect'] = trials_ic

        trials_lf = bb.core.Bunch()
        left = np.intersect1d(trials_id, self.correct_left_idx)
        right = np.intersect1d(trials_id, self.correct_right_idx)
        trials_lf['trials'] = np.append(left, right)
        trials_lf['lines'] = [[0, len(left)], [len(left), len(trials_lf['trials'])]]
        trials_lf['linecolours'] = [QtGui.QColor('#17becf'), QtGui.QColor('#9467bd')]
        trials_lf['text'] = ['left correct', 'right correct']
        correct_trials['left vs right'] = trials_lf
        correct_trials['correct vs incorrect and left vs right'] = trials_lf

        #Incorrect
        incorrect_trials = bb.core.Bunch()
        incorrect_trials['colour'] = QtGui.QColor('#d62728')
        incorrect_trials['fill'] = QtGui.QColor('#d62728')
        incorrect_trials['linestyle'] = QtGui.QPen(QtCore.Qt.SolidLine)
        trials_id = np.empty(0,int)
        idx= np.empty(0, int)
        for c in stim_contrast:
            idx = np.where(((self.trials['contrastLeft'] == c) | (self.trials['contrastRight'] == c)) & (self.trials['feedbackType'] == -1))[0]
            trials_id = np.append(trials_id, idx)
        trials_id = np.setdiff1d(trials_id, self.nan_trials)

        trials_no = bb.core.Bunch()
        trials_no['trials'] = trials_id
        trials_no['lines'] = []
        trials_no['linecolours'] = []
        trials_no['text'] = []
        incorrect_trials['trial no.'] = trials_no

        trials_ic = bb.core.Bunch()
        trials_ic['trials'] = trials_id
        trials_ic['lines'] = [[0, len(trials_ic['trials'])]]
        trials_ic['linecolours'] = [QtGui.QColor('#d62728')]
        trials_ic['text'] = ['incorrect']
        incorrect_trials['correct vs incorrect'] = trials_ic

        trials_lf = bb.core.Bunch()
        trials_iclf = bb.core.Bunch()
        left = np.intersect1d(trials_id, self.incorrect_left_idx)
        right = np.intersect1d(trials_id, self.incorrect_right_idx)
        trials_lf['trials'] = np.append(left, right)
        trials_lf['lines'] = [[0, len(left)], [len(left), len(trials_lf['trials'])]]
        trials_lf['linecolours'] = [QtGui.QColor('#8c564b'), QtGui.QColor('#ff7f0e')]
        trials_lf['text'] = ['left incorrect', 'right incorrect']
        incorrect_trials['left vs right'] = trials_lf
        incorrect_trials['correct vs incorrect and left vs right'] = trials_lf

        #Left
        left_trials = bb.core.Bunch()
        left_trials['colour'] = QtGui.QColor('#2ca02c')
        left_trials['fill'] = QtGui.QColor('#2ca02c')
        left_trials['linestyle'] = QtGui.QPen(QtCore.Qt.SolidLine)
        trials_id = np.empty(0,int)
        idx= np.empty(0, int)
        for c in stim_contrast:
            idx = np.where(self.trials['contrastLeft'] == c)[0]
            trials_id = np.append(trials_id, idx)
        trials_id = np.setdiff1d(trials_id, self.nan_trials)
    
        trials_no = bb.core.Bunch()
        trials_no['trials'] = trials_id
        trials_no['lines'] = []
        trials_no['linecolours'] = []
        trials_no['text'] = []
        left_trials['trial no.'] = trials_no

        trials_lf = bb.core.Bunch()
        trials_lf['trials'] = trials_id
        trials_lf['lines'] = [[0, len(trials_lf['trials'])]]
        trials_lf['linecolours'] = [QtGui.QColor('#2ca02c')]
        trials_lf['text'] = ['left']
        left_trials['left vs right'] = trials_lf

        trials_ic = bb.core.Bunch()
        correct = np.intersect1d(trials_id, self.correct_left_idx)
        incorrect = np.intersect1d(trials_id, self.incorrect_left_idx)
        trials_ic['trials'] = np.append(correct, incorrect)
        trials_ic['lines'] = [[0, len(correct)], [len(correct), len(trials_ic['trials'])]]
        trials_ic['linecolours'] = [QtGui.QColor('#17becf'), QtGui.QColor('#8c564b')]
        trials_ic['text'] = ['left correct', 'left incorrect']
        left_trials['correct vs incorrect'] = trials_ic
        left_trials['correct vs incorrect and left vs right'] = trials_ic

        #Right
        right_trials = bb.core.Bunch()
        right_trials['colour'] = QtGui.QColor('#bcbd22')
        right_trials['fill'] = QtGui.QColor('#bcbd22')
        right_trials['linestyle'] = QtGui.QPen(QtCore.Qt.SolidLine)
        trials_id = np.empty(0,int)
        idx= np.empty(0, int)
        for c in stim_contrast:
            idx = np.where(self.trials['contrastRight'] == c)[0]
            trials_id = np.append(trials_id, idx)
        trials_id = np.setdiff1d(trials_id, self.nan_trials)

        trials_no = bb.core.Bunch()
        trials_no['trials'] = trials_id
        trials_no['lines'] = []
        trials_no['linecolours'] = []
        trials_no['text'] = []
        right_trials['trial no.'] = trials_no

        trials_lf = bb.core.Bunch()
        trials_lf['trials'] = trials_id
        trials_lf['lines'] = [[0, len(trials_lf['trials'])]]
        trials_lf['linecolours'] = [QtGui.QColor('#bcbd22')]
        trials_lf['text'] = ['right']
        right_trials['left vs right'] = trials_lf

        trials_ic = bb.core.Bunch()
        correct = np.intersect1d(trials_id, self.correct_right_idx)
        incorrect = np.intersect1d(trials_id, self.incorrect_right_idx)
        trials_ic['trials'] = np.append(correct, incorrect)
        trials_ic['lines'] = [[0, len(correct)], [len(correct), len(trials_ic['trials'])]]
        trials_ic['linecolours'] = [QtGui.QColor('#9467bd'), QtGui.QColor('#ff7f0e')]
        trials_ic['text'] = ['right correct', 'right incorrect']
        right_trials['correct vs incorrect'] = trials_ic
        right_trials['correct vs incorrect and left vs right'] = trials_ic
        

        #Left Correct
        left_correct_trials = bb.core.Bunch()
        left_correct_trials['colour'] = QtGui.QColor('#17becf')
        left_correct_trials['fill'] = QtGui.QColor('#17becf')
        left_correct_trials['linestyle'] = QtGui.QPen(QtCore.Qt.DashLine)
        trials_id = np.empty(0,int)
        idx= np.empty(0, int)

        for c in stim_contrast:
            idx = np.where((self.trials['contrastLeft'] == c) & (self.trials['feedbackType'] == 1))[0]
            trials_id = np.append(trials_id, idx)
        trials_id = np.setdiff1d(trials_id, self.nan_trials)

        trials_no = bb.core.Bunch()
        trials_no['trials'] = trials_id
        trials_no['lines'] = []
        trials_no['linecolours'] = []
        trials_no['text'] = []
        left_correct_trials['trial no.'] = trials_no

        trials_lf = bb.core.Bunch()
        trials_lf['trials'] = trials_id
        trials_lf['lines'] = [[0, len(trials_lf['trials'])]]
        trials_lf['linecolours'] = [QtGui.QColor('#17becf')]
        trials_lf['text'] = ['left correct']

        left_correct_trials['left vs right'] = trials_lf
        left_correct_trials['correct vs incorrect'] = trials_lf
        left_correct_trials['correct vs incorrect and left vs right'] = trials_lf


        #Left Incorrect
        left_incorrect_trials = bb.core.Bunch()
        left_incorrect_trials['colour'] = QtGui.QColor('#8c564b')
        left_incorrect_trials['fill'] = QtGui.QColor('#8c564b')
        left_incorrect_trials['linestyle'] = QtGui.QPen(QtCore.Qt.DashLine)
        trials_id = np.empty(0,int)
        idx= np.empty(0, int)

        for c in stim_contrast:
            idx = np.where((self.trials['contrastLeft'] == c) & (self.trials['feedbackType'] == -1))[0]
            trials_id = np.append(trials_id, idx)
        trials_id = np.setdiff1d(trials_id, self.nan_trials)

        trials_no = bb.core.Bunch()
        trials_no['trials'] = trials_id
        trials_no['lines'] = []
        trials_no['linecolours'] = []
        trials_no['text'] = []
        left_incorrect_trials['trial no.'] = trials_no

        trials_lf = bb.core.Bunch()
        trials_lf['trials'] = trials_id
        trials_lf['lines'] = [[0, len(trials_lf['trials'])]]
        trials_lf['linecolours'] = [QtGui.QColor('#8c564b')]
        trials_lf['text'] = ['left incorrect']
        left_incorrect_trials['left vs right'] = trials_lf
        left_incorrect_trials['correct vs incorrect'] = trials_lf
        left_incorrect_trials['correct vs incorrect and left vs right'] = trials_lf

        #Right Correct
        right_correct_trials = bb.core.Bunch()
        right_correct_trials['colour'] = QtGui.QColor('#9467bd')
        right_correct_trials['fill'] = QtGui.QColor('#9467bd')
        right_correct_trials['linestyle'] = QtGui.QPen(QtCore.Qt.DashLine)
        trials_id = np.empty(0,int)
        idx= np.empty(0, int)

        for c in stim_contrast:
            idx = np.where((self.trials['contrastRight'] == c) & (self.trials['feedbackType'] == 1))[0]
            trials_id = np.append(trials_id, idx)
        trials_id = np.setdiff1d(trials_id, self.nan_trials)

        trials_no = bb.core.Bunch()
        trials_no['trials'] = trials_id
        trials_no['lines'] = []
        trials_no['linecolours'] = []
        trials_no['text'] = []
        right_correct_trials['trial no.'] = trials_no

        trials_lf = bb.core.Bunch()
        trials_lf['trials'] = trials_id
        trials_lf['lines'] = [[0, len(trials_lf['trials'])]]
        trials_lf['linecolours'] = [QtGui.QColor('#9467bd')]
        trials_lf['text'] = ['right correct']
        right_correct_trials['left vs right'] = trials_lf
        right_correct_trials['correct vs incorrect'] = trials_lf
        right_correct_trials['correct vs incorrect and left vs right'] = trials_lf

        #Right Incorrect
        right_incorrect_trials = bb.core.Bunch()
        right_incorrect_trials['colour'] = QtGui.QColor('#ff7f0e')
        right_incorrect_trials['fill'] = QtGui.QColor('#ff7f0e')
        right_incorrect_trials['linestyle'] = QtGui.QPen(QtCore.Qt.DashLine)
        trials_id = np.empty(0,int)
        idx= np.empty(0, int)

        for c in stim_contrast:
            idx = np.where((self.trials['contrastRight'] == c) & (self.trials['feedbackType'] == -1))[0]
            trials_id = np.append(trials_id, idx)
        trials_id = np.setdiff1d(trials_id, self.nan_trials)

        trials_no = bb.core.Bunch()
        trials_no['trials'] = trials_id
        trials_no['lines'] = []
        trials_no['linecolours'] = []
        trials_no['text'] = []
        right_incorrect_trials['trial no.'] = trials_no
        
        trials_lf = bb.core.Bunch()
        trials_lf['trials'] = trials_id
        trials_lf['lines'] = [[0, len(trials_lf['trials'])]]
        trials_lf['linecolours'] = [QtGui.QColor('#ff7f0e')]
        trials_lf['text'] = ['right incorrect']
        right_incorrect_trials['left vs right'] = trials_lf
        right_incorrect_trials['correct vs incorrect'] = trials_lf
        right_incorrect_trials['correct vs incorrect and left vs right'] = trials_lf


        trials = bb.core.Bunch()
        trials['all'] = all_trials
        trials['correct'] = correct_trials
        trials['incorrect'] = incorrect_trials
        trials['left'] = left_trials
        trials['right'] = right_trials
        trials['left correct'] = left_correct_trials
        trials['left incorrect'] = left_incorrect_trials
        trials['right correct'] = right_correct_trials
        trials['right incorrect'] = right_incorrect_trials

        return trials

    def get_sort_method(self, case):
        if case == 'all':
            sort_method = 'trial no.'
            id = 0
        elif (case == 'correct') | (case == 'incorrect'):
            sort_method = 'correct vs incorrect'
            id = 1
        elif (case == 'left') | (case == 'right'):
            sort_method = 'left vs right'
            id = 2
        else:
            sort_method = 'correct vs incorrect and left vs right'
            id = 3
        
        return sort_method, id


   
    def compute_trial_options(self, trials):
        self.trials = trials
        nan_feedback = np.where(np.isnan(self.trials['feedback_times']))[0]
        nan_goCue = np.where(np.isnan(self.trials['goCue_times']))[0]
        self.nan_trials = np.unique(np.append(nan_feedback, nan_goCue))
        self.n_trials = len(np.setdiff1d(np.arange(len(self.trials['feedbackType'])), self.nan_trials))
        self.correct_idx = np.setdiff1d(np.where(self.trials['feedbackType'] == 1)[0], self.nan_trials)
        self.incorrect_idx = np.setdiff1d(np.where(self.trials['feedbackType'] == -1)[0], self.nan_trials)
        self.right_idx = np.setdiff1d(np.where(np.isfinite(self.trials['contrastRight']))[0], self.nan_trials)
        self.left_idx = np.setdiff1d(np.where(np.isfinite(self.trials['contrastLeft']))[0], self.nan_trials)
        self.correct_right_idx = np.setdiff1d(np.intersect1d(self.correct_idx, self.right_idx), self.nan_trials)
        self.correct_left_idx = np.setdiff1d(np.intersect1d(self.correct_idx, self.left_idx), self.nan_trials)
        self.incorrect_right_idx = np.setdiff1d(np.intersect1d(self.incorrect_idx, self.right_idx), self.nan_trials)
        self.incorrect_left_idx = np.setdiff1d(np.intersect1d(self.incorrect_idx, self.left_idx), self.nan_trials)

        return self.nan_trials

    def reset_filters(self, stim = True):
        stim_contrast = [1, 0.25, 0.125, 0.0625, 0]
        case = 'all'
        sort_method = 'trial no.'
        if stim is True:
            for idx in range(self.contrast_options.count()):
                item = self.contrast_options.item(idx)
                item.setCheckState(QtCore.Qt.Checked)
        
        for idx, but in enumerate(self.filter_buttons.buttons()):
            if idx == 0:
                but.setCheckState(QtCore.Qt.Checked)
            else:
                but.setCheckState(QtCore.Qt.Unchecked)
        
        for idx, but in enumerate(self.trial_buttons.buttons()):
            if idx == 0:
                but.setChecked(True)
            else:
                but.setChecked(False)

        

        return stim_contrast, case, sort_method

