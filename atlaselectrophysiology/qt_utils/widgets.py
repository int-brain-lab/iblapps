from abc import ABC, abstractmethod
from qtpy import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import random


class GridTabSwitcher(QtWidgets.QWidget):
    custom_signal = QtCore.Signal(str)
    def __init__(self):

        super().__init__()
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # Layouts and containers
        self.main_layout = QtWidgets.QVBoxLayout(self)

        self.panels = []
        # Tab widget
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setTabPosition(QtWidgets.QTabWidget.South)
        self.tab_widget.hide()

        # TODO add rounded corners
        self.tab_widget.setStyleSheet("""
        QTabBar::tab:selected {
            background-color: #2c3e50;
            color: white;
            font-weight: bold;
        }
        """)

        # Splitter widget
        self.splitter_main = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        #self.splitter_main.setContentsMargins(0, 0, 0, 0)
        self.splitter_top = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter_bottom = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Splitter layout
        self.grid_layout = True

    def initialise(self, panels, names, headers=()):
        self.headers = headers
        self.panels = panels
        self.panel_names = list(names)

        if len(self.panels) == 4:
            self.splitter_main.addWidget(self.splitter_top)
            self.splitter_main.addWidget(self.splitter_bottom)

        self.add_splitter_layout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

    def add_splitter_layout(self):

        if len(self.panels) == 1:
            self.splitter_main.addWidget(self.panels[0])
        elif len(self.panels) == 4:
            self.splitter_top.addWidget(self.panels[0])
            self.splitter_top.addWidget(self.panels[1])
            self.splitter_bottom.addWidget(self.panels[2])
            self.splitter_bottom.addWidget(self.panels[3])
            self.splitter_top.setSizes([1] * self.splitter_top.count())
            self.splitter_bottom.setSizes([1] * self.splitter_bottom.count())
        else:
            return

        self.splitter_main.show()
        for panel in self.panels:
            panel.show()

        self.main_layout.addWidget(self.splitter_main)

    def delete_widgets(self):
        if self.grid_layout:
            self.remove_splitter_layout(delete=True)
        else:
            self.remove_tab_layout(delete=True)
        self.panels = []

    def remove_header(self):
        if self.headers:
            for panel, header in zip(self.panels, self.headers):
                panel.layout().removeWidget(header)

    def add_header(self):
        if self.headers:
            for panel, header in zip(self.panels, self.headers):
                panel.layout().insertWidget(0, header)

    def remove_splitter_layout(self, delete=False):

        if len(self.panels) == 1:
            splitters = [self.splitter_main]
        elif len(self.panels) == 4:
            splitters = [self.splitter_top, self.splitter_bottom]
        else:
            return

        for splitter in splitters:
            for i in reversed(range(splitter.count())):
                widget = splitter.widget(i)
                widget.setParent(None)
                if delete:
                    del widget

        self.main_layout.removeWidget(self.splitter_main)
        self.splitter_main.hide()

    def add_tab_layout(self):

        for i, w in enumerate(self.panels):
            self.tab_widget.addTab(w, f"{self.panel_names[i]}")
        self.main_layout.addWidget(self.tab_widget)
        self.tab_widget.show()

    def remove_tab_layout(self, delete=False):

        for i in reversed(range(self.tab_widget.count())):
            widget = self.tab_widget.widget(i)
            widget.setParent(None)
            if delete:
                del widget

        self.main_layout.removeWidget(self.tab_widget)
        self.tab_widget.hide()

    def toggle_layout(self):
        self.tab_widget.blockSignals(True)
        if self.grid_layout:
            # Switch to tab layout
            self.remove_splitter_layout()
            self.remove_header()
            self.add_tab_layout()
        else:
            # Switch to grid layout
            self.remove_tab_layout()
            self.add_header()
            self.add_splitter_layout()
            # Emit so we can signal that we have to add the lines for the fit as we now show 4 displays
            self.custom_signal.emit("lala")

        self.grid_layout = not self.grid_layout
        self.tab_widget.blockSignals(False)



class PopupWindow(QtWidgets.QWidget):
    closed = QtCore.Signal(QtWidgets.QWidget)
    leaveWidget = QtCore.Signal(QtWidgets.QWidget)
    enterWidget = QtCore.Signal(QtWidgets.QWidget)

    @classmethod
    def _instances(cls):
        app = QtWidgets.QApplication.instance()
        return [w for w in app.topLevelWidgets() if isinstance(w, cls)]

    @classmethod
    def _get_or_create(cls, title: str, **kwargs):
        window = next((w for w in cls._instances() if w.isVisible() and w.windowTitle() == title), None)
        if window is None:
            window = cls(title, **kwargs)
        else:
            window.showNormal()
            window.activateWindow()
        return window

    def __init__(self, title, parent=None, size=(300, 300), graphics=True):
        super(PopupWindow, self).__init__(parent)
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowStaysOnTopHint)
        self.resize(*size)
        self.move(random.randrange(30) + 1000, random.randrange(30) + 200)

        if graphics:
            self.popup_widget = pg.GraphicsLayoutWidget()
        else:
            self.popup_widget = QtWidgets.QWidget()
            self.layout = QtWidgets.QGridLayout()
            self.popup_widget.setLayout(self.layout)

        # Top-level layout for the whole popup window
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.popup_widget)
        self.setLayout(self.main_layout)

        # If parent is destroyed, close this popup
        if parent is not None:
            parent.destroyed.connect(self.close)

        self.setWindowTitle(title)
        self.setup()
        self.show()

    @abstractmethod
    def setup(self):
        """Abstract method to be implemented by subclasses."""
        pass

    def closeEvent(self, event):
        self.closed.emit(self)
        super().closeEvent(event)

    def leaveEvent(self, event):
        self.leaveWidget.emit(self)
        super().leaveEvent(event)

    def enterEvent(self, event):
        self.enterWidget.emit(self)
        super().leaveEvent(event)
