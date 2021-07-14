from PyQt5 import QtGui, QtCore
from iblutil.util import Bunch
import copy

colours = {'all': QtGui.QColor('#808080'),
           'correct': QtGui.QColor('#1f77b4'),
           'incorrect': QtGui.QColor('#d62728'),
           'left': QtGui.QColor('#2ca02c'),
           'right': QtGui.QColor('#bcbd22'),
           'left correct': QtGui.QColor('#17becf'),
           'right correct': QtGui.QColor('#9467bd'),
           'left incorrect': QtGui.QColor('#8c564b'),
           'right incorrect': QtGui.QColor('#ff7f0e'),
           'KS good': QtGui.QColor('#1f77b4'),
           'KS mua': QtGui.QColor('#fdc086'),
           'IBL good': QtGui.QColor('#7fc97f'),
           'IBL bad': QtGui.QColor('#d62728'),
           'line': QtGui.QColor('#7732a8'),
           'no metric': QtGui.QColor('#989898')}


all = Bunch()
idx = Bunch()
idx['colours'] = []
idx['text'] = []
side = Bunch()
side['colours'] = [colours['left'], colours['right']]
side['text'] = ['left', 'right']
choice = Bunch()
choice['colours'] = [colours['correct'], colours['incorrect']]
choice['text'] = ['correct', 'incorrect']
choice_and_side = Bunch()
choice_and_side['colours'] = [colours['left correct'], colours['right correct'],
                              colours['left incorrect'], colours['right incorrect']]
choice_and_side['text'] = ['left correct', 'right correct', 'left incorrect', 'right incorrect']
all['idx'] = idx
all['side'] = side
all['choice'] = choice
all['choice and side'] = choice_and_side


correct = Bunch()
side = Bunch()
side['colours'] = [colours['left correct'], colours['right correct']]
side['text'] = ['left correct', 'right correct']
choice = Bunch()
choice['colours'] = [colours['correct']]
choice['text'] = ['correct']
correct['idx'] = idx
correct['side'] = side
correct['choice'] = choice
correct['choice and side'] = side

incorrect = Bunch()
side = Bunch()
side['colours'] = [colours['left incorrect'], colours['right incorrect']]
side['text'] = ['left incorrect', 'right incorrect']
choice = Bunch()
choice['colours'] = [colours['incorrect']]
choice['text'] = ['incorrect']
incorrect['idx'] = idx
incorrect['side'] = side
incorrect['choice'] = choice
incorrect['choice and side'] = side


left = Bunch()
side = Bunch()
side['colours'] = [colours['left']]
side['text'] = ['left']
choice = Bunch()
choice['colours'] = [colours['left correct'], colours['left incorrect']]
choice['text'] = ['left correct', 'left incorrect']
left['idx'] = idx
left['side'] = side
left['choice'] = choice
left['choice and side'] = choice


right = Bunch()
side = Bunch()
side['colours'] = [colours['right']]
side['text'] = ['right']
choice = Bunch()
choice['colours'] = [colours['right correct'], colours['right incorrect']]
choice['text'] = ['right correct', 'right incorrect']
right['idx'] = idx
right['side'] = side
right['choice'] = choice
right['choice and side'] = choice

left_correct = Bunch()
side = Bunch()
side['colours'] = [colours['left correct']]
side['text'] = ['left correct']
left_correct['idx'] = idx
left_correct['side'] = side
left_correct['choice'] = side
left_correct['choice and side'] = side

right_correct = Bunch()
side = Bunch()
side['colours'] = [colours['right correct']]
side['text'] = ['right correct']
right_correct['idx'] = idx
right_correct['side'] = side
right_correct['choice'] = side
right_correct['choice and side'] = side

left_incorrect = Bunch()
side = Bunch()
side['colours'] = [colours['left incorrect']]
side['text'] = ['left incorrect']
left_incorrect['idx'] = idx
left_incorrect['side'] = side
left_incorrect['choice'] = side
left_incorrect['choice and side'] = side

right_incorrect = Bunch()
side = Bunch()
side['colours'] = [colours['right incorrect']]
side['text'] = ['right incorrect']
right_incorrect['idx'] = idx
right_incorrect['side'] = side
right_incorrect['choice'] = side
right_incorrect['choice and side'] = side

RASTER_OPTIONS = Bunch()
RASTER_OPTIONS['all'] = all
RASTER_OPTIONS['left'] = left
RASTER_OPTIONS['right'] = right
RASTER_OPTIONS['correct'] = correct
RASTER_OPTIONS['incorrect'] = incorrect
RASTER_OPTIONS['left correct'] = left_correct
RASTER_OPTIONS['right correct'] = right_correct
RASTER_OPTIONS['left incorrect'] = left_incorrect
RASTER_OPTIONS['right incorrect'] = right_incorrect

all = Bunch()
all['colour'] = copy.copy(colours['all'])
all['fill'] = colours['all']
all['text'] = 'all'

left = Bunch()
left['colour'] = copy.copy(colours['left'])
left['fill'] = colours['left']
left['text'] = 'left'

right = Bunch()
right['colour'] = copy.copy(colours['right'])
right['fill'] = colours['right']
right['text'] = 'right'

correct = Bunch()
correct['colour'] = copy.copy(colours['correct'])
correct['fill'] = colours['correct']
correct['text'] = 'correct'

incorrect = Bunch()
incorrect['colour'] = copy.copy(colours['incorrect'])
incorrect['fill'] = colours['incorrect']
incorrect['text'] = 'incorrect'

left_correct = Bunch()
left_correct['colour'] = copy.copy(colours['left correct'])
left_correct['fill'] = colours['left correct']
left_correct['text'] = 'left correct'

right_correct = Bunch()
right_correct['colour'] = copy.copy(colours['right correct'])
right_correct['fill'] = colours['right correct']
right_correct['text'] = 'right correct'

left_incorrect = Bunch()
left_incorrect['colour'] = copy.copy(colours['left incorrect'])
left_incorrect['fill'] = colours['left incorrect']
left_incorrect['text'] = 'left incorrect'

right_incorrect = Bunch()
right_incorrect['colour'] = copy.copy(colours['right incorrect'])
right_incorrect['fill'] = colours['right incorrect']
right_incorrect['text'] = 'right incorrect'

PSTH_OPTIONS = Bunch()
PSTH_OPTIONS['all'] = all
PSTH_OPTIONS['left'] = left
PSTH_OPTIONS['right'] = right
PSTH_OPTIONS['correct'] = correct
PSTH_OPTIONS['incorrect'] = incorrect
PSTH_OPTIONS['left correct'] = left_correct
PSTH_OPTIONS['right correct'] = right_correct
PSTH_OPTIONS['left incorrect'] = left_incorrect
PSTH_OPTIONS['right incorrect'] = right_incorrect

MAP_SIDE_OPTIONS = Bunch()
MAP_SIDE_OPTIONS['all'] = 'all'
MAP_SIDE_OPTIONS['left'] = 'left'
MAP_SIDE_OPTIONS['right'] = 'right'
MAP_SIDE_OPTIONS['correct'] = 'all'
MAP_SIDE_OPTIONS['incorrect'] = 'all'
MAP_SIDE_OPTIONS['left correct'] = 'left'
MAP_SIDE_OPTIONS['right correct'] = 'right'
MAP_SIDE_OPTIONS['left incorrect'] = 'left'
MAP_SIDE_OPTIONS['right incorrect'] = 'right'

MAP_CHOICE_OPTIONS = Bunch()
MAP_CHOICE_OPTIONS['all'] = 'all'
MAP_CHOICE_OPTIONS['left'] = 'all'
MAP_CHOICE_OPTIONS['right'] = 'all'
MAP_CHOICE_OPTIONS['correct'] = 'correct'
MAP_CHOICE_OPTIONS['incorrect'] = 'incorrect'
MAP_CHOICE_OPTIONS['left correct'] = 'correct'
MAP_CHOICE_OPTIONS['right correct'] = 'correct'
MAP_CHOICE_OPTIONS['left incorrect'] = 'incorrect'
MAP_CHOICE_OPTIONS['right incorrect'] = 'incorrect'

MAP_SORT_OPTIONS = Bunch()
MAP_SORT_OPTIONS['all'] = 'idx'
MAP_SORT_OPTIONS['left'] = 'side'
MAP_SORT_OPTIONS['right'] = 'side'
MAP_SORT_OPTIONS['correct'] = 'choice'
MAP_SORT_OPTIONS['incorrect'] = 'choice'
MAP_SORT_OPTIONS['left correct'] = 'choice and side'
MAP_SORT_OPTIONS['right correct'] = 'choice and side'
MAP_SORT_OPTIONS['left incorrect'] = 'choice and side'
MAP_SORT_OPTIONS['right incorrect'] = 'choice and side'


TRIAL_OPTIONS = ['all', 'correct', 'incorrect', 'left', 'right', 'left correct', 'left incorrect',
                 'right correct', 'right incorrect']
CONTRAST_OPTIONS = [1, 0.25, 0.125, 0.0625, 0]
ORDER_OPTIONS = ['trial num', 'reaction time']
SORT_OPTIONS = ['idx', 'choice', 'side', 'choice and side']
UNIT_OPTIONS = ['IBL good', 'IBL bad', 'KS good', 'KS mua']
SORT_CLUSTER_OPTIONS = ['ids', 'n spikes', 'IBL good', 'KS good']

SESS_QC = ['task', 'behavior', 'dlcLeft', 'dlcRight', 'videoLeft', 'videoRight']
CLUSTER_QC = ['noise_cutoff', 'amp_median', 'slidingRP_viol']

dlc_warning = 'WARNING: dlc points and timestamps differ in length, dlc points are ' \
              'not aligned correctly'


def get_icon(col_outer, col_inner, pix_size):

    p1 = QtGui.QPixmap(pix_size, pix_size)
    p1.fill(col_outer)
    p2 = QtGui.QPixmap(pix_size, pix_size)
    p2.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(p2)
    p.fillRect(int(pix_size / 4), int(pix_size / 4), int(pix_size / 2),
               int(pix_size / 2), col_inner)
    p.end()

    result = QtGui.QPixmap(p1)
    painter = QtGui.QPainter(result)
    painter.drawPixmap(QtCore.QPoint(), p1)
    painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
    painter.drawPixmap(result.rect(), p2, p2.rect())
    painter.end()

    return result
