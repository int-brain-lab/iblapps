from PyQt5 import QtCore, QtWidgets
import numpy as np


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
  
        self.choice_options = QtWidgets.QGroupBox('Stimulus Choice')
        self.choice_option1 = QtWidgets.QRadioButton('all')
        self.choice_option2 = QtWidgets.QRadioButton('correct')
        self.choice_option3 = QtWidgets.QRadioButton('incorrect')
        self.choice_option1.setChecked(True)
        self.group_buttons_V(self.choice_option1, self.choice_option2, self.choice_option3, self.choice_options)

        # Option for filtering by choice
        self.stim_options = QtWidgets.QGroupBox('Stimulus Side')
        self.stim_option1 = QtWidgets.QRadioButton('all')
        self.stim_option2 = QtWidgets.QRadioButton('left')
        self.stim_option3 = QtWidgets.QRadioButton('right')
        self.stim_option1.setChecked(True)
        self.group_buttons_V(self.stim_option1, self.stim_option2, self.stim_option3, self.stim_options)

        # Print out no. of trials for each filter condition
        self.ntrials_text = QtWidgets.QLabel('No. of trials  = ')
        
        self.filter_options_group = QtWidgets.QGroupBox()
        self.group_filter_widget()
        self.filter_options_group.setFixedSize(250, 350)

        self.trial_options = QtWidgets.QGroupBox('Sort Trials By:')
        self.trial_option1 = QtWidgets.QRadioButton('trial no.')
        self.trial_option2 = QtWidgets.QRadioButton('correct vs incorrect')
        self.trial_option3 = QtWidgets.QRadioButton('left vs right')
        self.trial_option4 = QtWidgets.QRadioButton('correct vs incorrect and left vs right')
        self.trial_option1.setChecked(True)
        self.group_buttons_H(self.trial_option1, self.trial_option2, self.trial_option3, self.trial_option4, self.trial_options)

    def group_buttons_V(self, button1, button2, button3, group):
        button_layout = QtWidgets.QVBoxLayout()
        button_layout.setSpacing(8)
        button_layout.addWidget(button1)
        button_layout.addWidget(button2)
        button_layout.addWidget(button3)
        group.setLayout(button_layout)
      
    def group_buttons_H(self, button1, button2, button3, button4, group):
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(button1)
        button_layout.addWidget(button2)
        button_layout.addWidget(button3)
        button_layout.addWidget(button4)
        group.setLayout(button_layout)

    def group_filter_widget(self):
        group_layout = QtWidgets.QVBoxLayout()
        group_layout.addWidget(self.reset_filter_button)
        group_layout.addWidget(self.contrast_options_text)
        group_layout.addWidget(self.contrast_options)
        group_layout.addWidget(self.stim_options)
        group_layout.addWidget(self.choice_options)
        group_layout.addWidget(self.ntrials_text)
        self.filter_options_group.setLayout(group_layout)

    def get_checked_contrasts(self):
        stim_contrast = []
        for idx in range(self.contrast_options.count()):
            if self.contrast_options.item(idx).checkState() == QtCore.Qt.Checked:
                stim_contrast.append(self.contrasts[idx])
        
        return stim_contrast

    def filter_and_sort_trials(self, stim_contrast, stim_side, stim_choice, sort_method):
        trials_id = np.empty(0,int)
        idx= np.empty(0, int)
        lines = []
        line_colour = []
 
        if (stim_choice == 'all') & (stim_side == 'all'):
            for c in stim_contrast:
                idx = np.where((self.trials['contrastLeft'] == c) | (self.trials['contrastRight'] == c))[0]
                trials_id = np.append(trials_id, idx)
            trials_id = np.setdiff1d(trials_id, self.nan_trials)

            if sort_method == 'correct vs incorrect':
                correct = np.intersect1d(trials_id, self.correct_idx)
                incorrect = np.intersect1d(trials_id, self.incorrect_idx)
                sort = np.append(correct, incorrect)
                lines = [len(correct)]
                line_colour = ['r']
                trials_id = sort
            elif sort_method == 'left vs right':
                left = np.intersect1d(trials_id, self.left_idx)
                right = np.intersect1d(trials_id, self.right_idx)
                sort = np.append(left, right)
                lines = [len(left)]
                line_colour = ['b']
                trials_id = sort
            elif sort_method == 'correct vs incorrect and left vs right':
                correct_right = np.intersect1d(trials_id, self.correct_right_idx)
                correct_left = np.intersect1d(trials_id, self.correct_left_idx)
                incorrect_right = np.intersect1d(trials_id, self.incorrect_right_idx)
                incorrect_left = np.intersect1d(trials_id, self.incorrect_left_idx)

                sort = np.concatenate((correct_left, correct_right, incorrect_left, incorrect_right))
                lines = [len(correct_left), len(correct_left) + len(correct_right), len(correct_left) + len(correct_right) + len(incorrect_left)]
                line_colour = ['b', 'r', 'b']
                trials_id = sort

        if (stim_choice == 'all') & (stim_side != 'all'):
            contrast = 'contrastRight' if stim_side == 'right' else 'contrastLeft'
            for c in stim_contrast:
                idx = np.where(self.trials[contrast] == c)[0]
                trials_id = np.append(trials_id, idx)
            trials_id = np.setdiff1d(trials_id, self.nan_trials)
  
            if (sort_method == 'correct vs incorrect') | (sort_method == 'correct vs incorrect and left vs right'):
                if stim_side == 'right':
                    correct = np.intersect1d(trials_id, self.correct_right_idx)
                    incorrect = np.intersect1d(trials_id, self.incorrect_right_idx)
                    sort = np.append(correct, incorrect)
                    lines = [len(correct)]
                    line_colour = ['r']
                    trials_id = sort
                else:
                    correct = np.intersect1d(trials_id, self.correct_left_idx)
                    incorrect = np.intersect1d(trials_id, self.incorrect_left_idx)
                    sort = np.append(correct, incorrect)
                    lines = [len(correct)]
                    line_colour = ['r']
                    trials_id = sort

        if (stim_choice != 'all') & (stim_side == 'all'):
            outcome = 1 if stim_choice == 'correct' else -1
            for c in stim_contrast:
                idx = np.where(((self.trials['contrastLeft'] == c) | (self.trials['contrastRight'] == c)) & (self.trials['feedbackType'] == outcome))[0]
                trials_id = np.append(trials_id, idx)
            trials_id = np.setdiff1d(trials_id, self.nan_trials)

            if (sort_method == 'left vs right') | (sort_method == 'correct vs incorrect and left vs right'):
                if stim_choice == 'correct':
                    left = np.intersect1d(trials_id, self.correct_left_idx)
                    right = np.intersect1d(trials_id, self.correct_right_idx)
                    sort = np.append(left, right)
                    lines = [len(left)]
                    line_colour = ['b']
                    trials_id = sort
                else:
                    left = np.intersect1d(trials_id, self.incorrect_left_idx)
                    right = np.intersect1d(trials_id, self.incorrect_right_idx)
                    sort = np.append(left, right)
                    lines = [len(left)]
                    line_colour = ['b']
                    trials_id = sort
                
        if (stim_choice != 'all') & (stim_side != 'all'):
            outcome = 1 if stim_choice == 'correct' else -1
            contrast = 'contrastRight' if stim_side == 'right' else 'contrastLeft'
            for c in stim_contrast:
                idx = np.where((self.trials[contrast] == c) & (self.trials['feedbackType'] == outcome))[0]
                trials_id = np.append(trials_id, idx)
            trials_id = np.setdiff1d(trials_id, self.nan_trials)
        
        return trials_id, lines, line_colour
   
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

    def reset_filters(self):
        stim_contrast = [1, 0.25, 0.125, 0.0625, 0]
        stim_side = 'all'
        stim_choice = 'all'
        sort_method = 'trial no.'
        for idx in range(self.contrast_options.count()):
            item = self.contrast_options.item(idx)
            item.setCheckState(QtCore.Qt.Checked)
        self.choice_option1.setChecked(True)
        self.stim_option1.setChecked(True)
        self.trial_option1.setChecked(True)

        return stim_contrast, stim_choice, stim_side, sort_method

