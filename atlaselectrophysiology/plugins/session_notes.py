from atlaselectrophysiology.qt_utils.utils import PopupWindow
from qtpy import QtWidgets

PLUGIN_NAME = "Session Notes"

def setup(parent):
    parent.plugins[PLUGIN_NAME] = dict()
    parent.plugins[PLUGIN_NAME]['loader'] = LoadSessionNotes(parent.shank.one, parent.shank.eid)

    action = QtWidgets.QAction(PLUGIN_NAME, parent)
    action.triggered.connect(lambda: callback(parent))
    parent.plugin_options.addAction(action)


def callback(parent):
    loader = parent.plugins[PLUGIN_NAME]['loader']
    loader.load()
    parent.session_notes = SessionNotes._get_or_create('Session Notes', data=loader.data, parent=parent)


class LoadSessionNotes:
    def __init__(self, one, eid):
        self.one = one
        self.eid = eid
        self.data = None

    def load(self):
        sess = self.one.alyx.rest('sessions', 'read', id=self.eid)
        if sess['notes']:
            self.data = sess['notes'][0]['text']
        if not self.data:
            self.data = sess['narrative']
        if not self.data:
            self.data = 'No notes for this session'

    def reset(self):
        self.data = None


class SessionNotes(PopupWindow):
    def __init__(self, title, data=None, parent=None):
        self.data = data
        super().__init__(title, parent=parent, size=(200, 100), graphics=False)

    def setup(self):
        notes = QtWidgets.QTextEdit()
        notes.setReadOnly(True)
        notes.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
        notes.setText(self.data)
        self.layout.addWidget(notes)
