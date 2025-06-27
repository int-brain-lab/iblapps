import pandas as pd
from pathlib import Path
from one.api import ONE
from one.alf.path import ALFPath


one = ONE()
eid = '6a720380-8969-4cf0-9247-3d4ea300ee74'

session_path = one.eid2path(eid)
session = session_path.session_path_short()

probes = ['probe00', 'probe01', 'probe02']
shanks = ['a', 'b', 'c', 'd']




# session
# probe
# shank
# alignment

df = []

for probe in probes:
    for shank in shanks:
        data = {'session': session,
                'probe': probe + shank,
                'local_path': Path(session_path),
                'is_quarter': False}
        df.append(pd.DataFrame([data]))

quarter_path = ALFPath('/Users/admin/Downloads/quarterdensity/Subjects/KM_027/2024-12-13/002')
session_quarter = quarter_path.session_path_short()
for probe in probes:
    for shank in shanks:
        data = {'session': session_quarter,
                'probe': probe + shank,
                'local_path': quarter_path.joinpath('alf', probe + shank, 'iblsorter'),
                'is_quarter': True}
        df.append(pd.DataFrame([data]))


df = pd.concat(df)







from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QTextEdit,
    QWidget, QPushButton, QVBoxLayout
)
from PyQt5.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dock Grid and Tab Mode")

        # Central widget with toggle button
        central_widget = QWidget()
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        self.toggle_button = QPushButton("Toggle View Mode")
        layout.addWidget(self.toggle_button)


        self.toggle_button.clicked.connect(self.toggle_mode)

        # Create docks
        self.docks = [self.create_dock(f"Dock {i+1}") for i in range(4)]
        self.in_tab_mode = False
        self.setup_grid_mode()  # Start in grid

    def create_dock(self, title):
        dock = QDockWidget(title, self)
        dock.setWidget(QTextEdit(f"Content of {title}"))
        dock.setFeatures(QDockWidget.DockWidgetMovable)
        return dock

    def setup_grid_mode(self):
        # Start fresh (only once)
        for dock in self.docks:
            self.addDockWidget(Qt.LeftDockWidgetArea, dock)
            dock.show()

        self.splitDockWidget(self.docks[0], self.docks[1], Qt.Horizontal)
        self.splitDockWidget(self.docks[0], self.docks[2], Qt.Vertical)
        self.splitDockWidget(self.docks[1], self.docks[3], Qt.Vertical)

    def setup_tab_mode(self):
        # Tabify all docks under dock 0
        for dock in self.docks[1:]:
            self.tabifyDockWidget(self.docks[0], dock)
            dock.show()

        # Show all tabs
        for dock in self.docks:
            dock.show()
        self.docks[0].raise_()

    def toggle_mode(self):
        if self.in_tab_mode:
            self.setup_grid_mode()
        else:
            self.setup_tab_mode()
        self.in_tab_mode = not self.in_tab_mode


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.resize(1000, 700)
    window.show()
    app.exec_()

# splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
#
# # Create your central widget (e.g., main display area)
# main_widget = QtWidgets.QWidget()
# main_layout = QtWidgets.QVBoxLayout(main_widget)
# main_layout.addLayout(self.interaction_layout3)
# main_layout.addWidget(items.fig_slice_area)
# main_layout.addLayout(self.interaction_layout1)
# main_layout.addWidget(items.fig_fit)
# main_layout.addLayout(self.interaction_layout2)
#
# # Wrap docks in a QWidget container
# dock_container = QtWidgets.QWidget()
# dock_layout = QtWidgets.QVBoxLayout(dock_container)
# dock_layout.setContentsMargins(0, 0, 0, 0)
#
# # Add all docks as children of the dock container
# dock_area = QtWidgets.QMainWindow()
# dock_area.setDockNestingEnabled(True)
# dock_area.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.docks[0])
# dock_area.splitDockWidget(self.docks[0], self.docks[1], QtCore.Qt.Horizontal)
# dock_area.splitDockWidget(self.docks[0], self.docks[2], QtCore.Qt.Vertical)
# dock_area.splitDockWidget(self.docks[1], self.docks[3], QtCore.Qt.Vertical)
# #
# # for i, dock in enumerate(self.docks):
# #     dock_area.addDockWidget(QtCore.Qt.TopDockWidgetArea, dock)
# dock_layout.addWidget(dock_area)
#
# # Add widgets to splitter
# splitter.addWidget(dock_container)  # Left side: docks
# splitter.addWidget(main_widget)  # Right side: main content
#
# # Set 3/4 width for docks, 1/4 for main content
# splitter.setSizes([1200, 400])  # Or [75, 25] proportional
#
# self.setCentralWidget(splitter)



import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QSplitter, QTextEdit, QVBoxLayout,
    QTabWidget, QLabel
)
from PyQt5.QtCore import Qt


import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QSplitter, QTextEdit, QVBoxLayout,
    QTabWidget
)
from PyQt5.QtCore import Qt


class GridTabWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Grid ↔ Tab Toggle Example")
        self.resize(1000, 600)

        # Create the 4 content panels
        self.panel_widgets = [QTextEdit(f"Panel {i + 1}") for i in range(4)]

        # Main layout with no margins or spacing
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.setLayout(self.main_layout)

        # Initialize both views
        self.init_split_layout()
        self.init_tab_layout()

        # Show grid layout by default
        self.show_grid_layout = True
        self.main_layout.addWidget(self.grid_widget)

    def init_split_layout(self):
        self.h_splitter = QSplitter(Qt.Vertical)

        top_splitter = QSplitter(Qt.Horizontal)
        bottom_splitter = QSplitter(Qt.Horizontal)

        top_splitter.addWidget(self.panel_widgets[0])
        top_splitter.addWidget(self.panel_widgets[1])
        bottom_splitter.addWidget(self.panel_widgets[2])
        bottom_splitter.addWidget(self.panel_widgets[3])

        self.h_splitter.addWidget(top_splitter)
        self.h_splitter.addWidget(bottom_splitter)

        self.grid_widget = self.h_splitter

    def init_tab_layout(self):
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.South)  # Tabs at bottom
        self.tab_widget.setContentsMargins(0, 0, 0, 0)

        for i, panel in enumerate(self.panel_widgets):
            self.tab_widget.addTab(panel, f"Panel {i + 1}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_T:
            self.toggle_layout()

    def toggle_layout(self):
        current_widget = self.grid_widget if self.show_grid_layout else self.tab_widget
        self.main_layout.removeWidget(current_widget)
        current_widget.setParent(None)

        if self.show_grid_layout:
            self.main_layout.addWidget(self.tab_widget)
        else:
            self.main_layout.addWidget(self.grid_widget)

        self.show_grid_layout = not self.show_grid_layout


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GridTabWidget()
    window.show()
    sys.exit(app.exec_())



import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QTextEdit,
    QVBoxLayout, QTabWidget, QLabel
)
from PyQt5.QtCore import Qt


class GridTabWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Shared widgets
        self.panels = [QLabel(f"Shared Panel {i + 1}") for i in range(4)]

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.grid_widget = self.create_split_layout()
        self.tab_widget = self.create_tab_layout()

        self.show_grid_layout = True
        self.main_layout.addWidget(self.grid_widget)

    def create_split_layout(self):
        h_splitter = QSplitter(Qt.Vertical)

        top_splitter = QSplitter(Qt.Horizontal)
        bottom_splitter = QSplitter(Qt.Horizontal)

        # Important: reparent and add shared widgets
        top_splitter.addWidget(self.panels[0])
        top_splitter.addWidget(self.panels[1])
        bottom_splitter.addWidget(self.panels[2])
        bottom_splitter.addWidget(self.panels[3])

        h_splitter.addWidget(top_splitter)
        h_splitter.addWidget(bottom_splitter)

        return h_splitter

    def create_tab_layout(self):
        tab_widget = QTabWidget()
        tab_widget.setTabPosition(QTabWidget.South)
        return tab_widget

    def toggle_layout(self):
        current = self.grid_widget if self.show_grid_layout else self.tab_widget
        self.main_layout.removeWidget(current)
        current.hide()

        if self.show_grid_layout:
            # Remove from splitter
            self._remove_widgets_from_splitter(self.grid_widget)
            # Add to tab
            for i, w in enumerate(self.panels):
                self.tab_widget.addTab(w, f"Panel {i + 1}")
            new_layout = self.tab_widget
        else:
            # Remove from tab
            for i in reversed(range(self.tab_widget.count())):
                self.tab_widget.removeTab(i)
            # Rebuild splitter and reassign self.grid_widget
            self.grid_widget = self.create_split_layout()
            new_layout = self.grid_widget

        self.main_layout.addWidget(new_layout)
        new_layout.show()

        self.show_grid_layout = not self.show_grid_layout

    def _remove_widgets_from_splitter(self, splitter):
        """Recursively remove all widgets from a splitter."""
        for i in reversed(range(splitter.count())):
            widget = splitter.widget(i)
            if isinstance(widget, QSplitter):
                self._remove_widgets_from_splitter(widget)
            splitter.widget(i).setParent(None)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shared Widgets: Grid ↔ Tab View")
        self.resize(1200, 800)

        self.grid_tab_widget = GridTabWidget()
        self.setCentralWidget(self.grid_tab_widget)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_T:
            self.grid_tab_widget.toggle_layout()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())



from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QSplitter, QTextEdit, QTabWidget
)
from PyQt5.QtCore import Qt
import sys


class GridTabSwitcher(QWidget):
    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)

        # Layouts and containers
        self.main_layout = QVBoxLayout(self)

        # Persistent widgets to reuse
        self.panels = [QTextEdit(f"Panel {i + 1}") for i in range(4)]
        for p in self.panels:
            p.setMinimumSize(100, 100)

        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.South)
        self.tab_widget.hide()

        # Splitter widget
        self.splitter_main = QSplitter(Qt.Vertical)
        self.splitter_top = QSplitter(Qt.Horizontal)
        self.splitter_bottom = QSplitter(Qt.Horizontal)
        self.splitter_main.addWidget(self.splitter_top)
        self.splitter_main.addWidget(self.splitter_bottom)

        # Splitter layout
        self.show_grid_layout = True

        self.add_splitter_layout()

    def add_splitter_layout(self):

        self.splitter_top.addWidget(self.panels[0])
        self.splitter_top.addWidget(self.panels[1])
        self.splitter_bottom.addWidget(self.panels[2])
        self.splitter_bottom.addWidget(self.panels[3])

        self.splitter_main.show()
        for panel in self.panels:
            panel.show()

        self.main_layout.addWidget(self.splitter_main)

    def remove_splitter_layout(self):

        for i in reversed(range(self.splitter_top.count())):
            widget = self.splitter_top.widget(i)
            widget.setParent(None)

        for i in reversed(range(self.splitter_top.count())):
            widget = self.splitter_bottom.widget(i)
            widget.setParent(None)

        self.main_layout.removeWidget(self.splitter_main)
        self.splitter_main.hide()

    def add_tab_layout(self):

        for i, w in enumerate(self.panels):
            self.tab_widget.addTab(w, f"Panel {i + 1}")
        self.main_layout.addWidget(self.tab_widget)
        self.tab_widget.show()

    def remove_tab_layout(self):

        for i in reversed(range(self.tab_widget.count())):
            widget = self.tab_widget.widget(i)
            widget.setParent(None)

        self.main_layout.removeWidget(self.tab_widget)
        self.tab_widget.hide()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_T:
            self.toggle_layout()

    def toggle_layout(self):
        if self.show_grid_layout:
            # Switch to tab layout
            self.remove_splitter_layout()
            self.add_tab_layout()
        else:
            # Switch to grid layout
            self.remove_tab_layout()
            self.add_splitter_layout()

        self.show_grid_layout = not self.show_grid_layout


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grid to Tabs Example")
        self.resize(800, 600)

        self.switcher = GridTabSwitcher()
        self.setCentralWidget(self.switcher)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())




import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import numpy as np
import sys
import atlaselectrophysiology.qt_utils.ColorBar as cb


class LUTControllerExample(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        layout = QtWidgets.QVBoxLayout(self)

        # Create four ImageItems in separate GraphicsLayouts
        self.views = []
        self.images = []
        img_data = np.random.random(size=(100, 100))


        fig_slice_area = pg.GraphicsLayoutWidget()
        fig_slice_layout = pg.GraphicsLayout()
        fig_slice_area.addItem(fig_slice_layout)
        layout.addWidget(fig_slice_area)

        color_bar = cb.ColorBar('cividis')
        lut = color_bar.getColourMap()

        for _ in range(4):
            # view = pg.GraphicsLayoutWidget()
            # images_layout.addItem(view)
            vb = pg.ViewBox()
            fig_slice_layout.addItem(vb)
            # vb = view.addViewBox()
            vb.setAspectLocked(True)
            img = pg.ImageItem(img_data)
            img.setLookupTable(lut)
            vb.addItem(img)
            self.images.append(img)
            self.views.append(vb)

        # Create one HistogramLUTItem and show only the gradient editor
        self.histLUT = pg.HistogramLUTItem()
        fig_slice_layout.addItem(self.histLUT)

        # Set one image to histogramLUT for range reference (doesn't have to be displayed)
        self.histLUT.setImageItem(self.images[0])
        self.histLUT.gradient.setColorMap(color_bar.map)

        # Now connect the histogramLUT's lookup table to all images
        #self.histLUT.sigLevelsChanged.connect(self.update_levels)

        #self.update_levels()  # initial update

    def update_levels(self):
        levels = self.histLUT.getLevels()
        lut = self.histLUT.getLookupTable()

        for img in self.images:
            img.setLevels(levels)
            img.setLookupTable(lut)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = LUTControllerExample()
    win.show()
    sys.exit(app.exec_())




from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QComboBox, QToolBar, QWidget, QHBoxLayout, QAction
)
from PyQt5.QtCore import Qt
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ComboBox on Menu Bar Right")

        # Create a QToolBar to act as the menu bar
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        # Left-aligned actions
        toolbar.addAction("File")
        toolbar.addAction("Edit")

        # Spacer to push following widgets to the right
        spacer = QWidget()
        spacer.setSizePolicy(
            spacer.sizePolicy().Expanding, spacer.sizePolicy().Preferred)
        toolbar.addWidget(spacer)

        # Right-aligned combo box
        combo = QComboBox()
        combo.addItems(["Option 1", "Option 2", "Option 3"])
        toolbar.addWidget(combo)

        self.setGeometry(100, 100, 600, 400)

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())



from PyQt5.QtWidgets import QApplication, QMainWindow, QToolBar, QToolButton, QMenu
from PyQt5.QtCore import Qt, QEvent

class HoverMenuButton(QToolButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPopupMode(QToolButton.InstantPopup)  # show menu on click immediately

    def enterEvent(self, event):
        super().enterEvent(event)
        if self.menu():
            # Show menu immediately on hover (like QMenuBar)
            self.showMenu()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        # Optional: you can close menu when hover leaves (usually handled by menu itself)
        # self.menu().close()

app = QApplication([])

window = QMainWindow()
toolbar = QToolBar()
window.addToolBar(toolbar)

# Create a menu
menu = QMenu()
menu.addAction("Option 1")
menu.addAction("Option 2")

# Create hover menu button
btn = HoverMenuButton()
btn.setText("File")
btn.setMenu(menu)
btn.setPopupMode(QToolButton.InstantPopup)  # show menu immediately on click

toolbar.addWidget(btn)

# Add another normal button (optional)
btn2 = QToolButton()
btn2.setText("Edit")
toolbar.addWidget(btn2)

window.show()
app.exec()

