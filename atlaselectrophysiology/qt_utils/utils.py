import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from typing import Any, Union, Optional, List, Dict, Tuple, Callable
import random
from abc import ABC, abstractmethod
import numpy as np


colours = ['#cc0000', '#6aa84f', '#1155cc', '#a64d79']

kpen_dot = pg.mkPen(color='k', style=QtCore.Qt.DotLine, width=2)
rpen_dot = pg.mkPen(color='r', style=QtCore.Qt.DotLine, width=2)
kpen_solid = pg.mkPen(color='k', style=QtCore.Qt.SolidLine, width=2)
bpen_solid = pg.mkPen(color='b', style=QtCore.Qt.SolidLine, width=3)


tab_style = {
    'selected': """QLabel {
                background-color: #2c3e50;
                border: 1px solid lightgrey;
                color: white;
                padding: 6px;
                font-weight: bold;
            }
            """,
    'deselected': """
            QLabel {
                background-color: rgb(240, 240, 240);
                border: 1px solid lightgrey;
                color: black;
                padding: 6px;
                font-weight: bold;
            }
            """
}

button_style = {
    'activated': """
    QPushButton {
        background-color: grey;
        border: 1px solid lightgrey;
        color: white;
        border-radius: 5px;  /* Rounded corners */
        padding: 2px;
    }
""",
    'deactivated': """
    QPushButton {
        background-color: white;
        border: 1px solid transparent;
        color: grey;
        border-radius: 5px;  /* Rounded corners */
        padding: 2px;
    }
"""
}

# TODO fix
def set_view(
        self,
        view: int,
        configure: bool = False
) -> None:
    """
    Update the layout and visual configuration of figure panels based on the selected view mode.

    Parameters
    ----------
    view : int
        The layout mode to use. Supported modes are:
            1 - img | line | probe
            2 - img | probe | line
            3 - probe | line | img
    configure : bool
        If True, update stored figure width and height dimensions before applying the layout.
    """

    if configure:
        self.fig_ax_width = self.fig_data_ax.width()
        self.fig_img_width = self.fig_img.width() - self.fig_ax_width
        self.fig_line_width = self.fig_line.width()
        self.fig_probe_width = self.fig_probe.width()
        self.slice_width = self.fig_slice.width()
        self.slice_height = self.fig_slice.height()
        self.slice_rect = self.fig_slice.viewRect()

    # Remove all existing layout items to start fresh
    for item in [self.fig_img_cb, self.fig_probe_cb, self.fig_img, self.fig_line, self.fig_probe]:
        self.fig_data_layout.removeItem(item)

    # Define configurations for each view mode
    layout_configs = {
        1: {
            'items': [
                (self.fig_img_cb, 0, 0),
                (self.fig_probe_cb, 0, 1, 1, 2),
                (self.fig_img, 1, 0),
                (self.fig_line, 1, 1),
                (self.fig_probe, 1, 2),
            ],
            'col_stretch': [(0, 6), (1, 2), (2, 1)],
            'row_stretch': [(0, 1), (1, 10)],
            'axis_target': self.fig_img,
            'sizes': lambda: (
                self.fig_img.setPreferredWidth(self.fig_img_width + self.fig_ax_width),
                self.fig_line.setPreferredWidth(self.fig_line_width),
                self.fig_probe.setFixedWidth(self.fig_probe_width)
            )
        },
        2: {
            'items': [
                (self.fig_img_cb, 0, 0),
                (self.fig_probe_cb, 0, 1, 1, 2),
                (self.fig_img, 1, 0),
                (self.fig_probe, 1, 1),
                (self.fig_line, 1, 2),
            ],
            'col_stretch': [(0, 6), (1, 1), (2, 2)],
            'row_stretch': [(0, 1), (1, 10)],
            'axis_target': self.fig_img,
            'sizes': lambda: (
                self.fig_img.setPreferredWidth(self.fig_img_width + self.fig_ax_width),
                self.fig_line.setPreferredWidth(self.fig_line_width),
                self.fig_probe.setFixedWidth(self.fig_probe_width)
            )
        },
        3: {
            'items': [
                (self.fig_probe_cb, 0, 0, 1, 2),
                (self.fig_img_cb, 0, 2),
                (self.fig_probe, 1, 0),
                (self.fig_line, 1, 1),
                (self.fig_img, 1, 2),
            ],
            'col_stretch': [(0, 1), (1, 2), (2, 6)],
            'row_stretch': [(0, 1), (1, 10)],
            'axis_target': self.fig_probe,
            'sizes': lambda: (
                self.fig_probe.setFixedWidth(self.fig_probe_width + self.fig_ax_width),
                self.fig_img.setPreferredWidth(self.fig_img_width),
                self.fig_line.setPreferredWidth(self.fig_line_width)
            )
        }
    }

    # Validate view and retrieve layout config
    config = layout_configs.get(view)
    if not config:
        raise ValueError(f"Unknown view mode: {view}")

    # Add layout items
    for item_args in config['items']:
        self.fig_data_layout.addItem(*item_args)

    # Apply column and row stretch factors
    for col, factor in config['col_stretch']:
        self.fig_data_layout.layout.setColumnStretchFactor(col, factor)
    for row, factor in config['row_stretch']:
        self.fig_data_layout.layout.setRowStretchFactor(row, factor)

    # Configure axes: only one figure shows the axis label
    for fig in [self.fig_img, self.fig_line, self.fig_probe]:
        if fig == config['axis_target']:
            utils.set_axis(fig, 'left', label='Distance from probe tip (um)')
        else:
            utils.set_axis(fig, 'left', show=False)

    # Apply size adjustments specific to view
    config['sizes']()

    # Force updates and axis correction
    for fig in [self.fig_img, self.fig_line, self.fig_probe]:
        fig.update()
    self.fig_img.setXRange(min=self.xrange[0] - 10, max=self.xrange[1] + 10, padding=0)
    self.reset_axis_button_pressed()



def toggle_plots(
        options_group: QtWidgets.QActionGroup
) -> None:
    """
    Cycle through image, line, probe and slice plots using keyboard shortcuts (Alt+1, Alt+2, Alt+3, Alt+4)
    Parameters
    ----------
    options_group : QActionGroup
        The group of QAction items representing plots to toggle through
    """
    current_act = options_group.checkedAction()
    actions = options_group.actions()
    current_idx = next(i for i, act in enumerate(actions) if act == current_act)
    next_idx = np.mod(current_idx + 1, len(actions))
    actions[next_idx].setChecked(True)
    actions[next_idx].trigger()



def remove_items(fig, items, delete=True):
    for item in items:
        fig.removeItem(item)
        if delete:
            del item
    return []


def set_axis(
        fig: Union[pg.PlotItem, pg.PlotWidget],
        ax: str,
        show: bool = True,
        label: Optional[str] =None,
        pen: str = 'k',
        ticks: bool = True
) -> pg.AxisItem:
    """
    Show, hide, and configure an axis on a PyQtGraph figure.

    Parameters
    ----------
    fig : pyqtgraph.PlotWidget or pyqtgraph.PlotItem
        The figure containing the axis to modify.
    ax : str
        The orientation of the axis. Must be one of {'left', 'right', 'top', 'bottom'}.
    show : bool, optional
        Whether to show the axis (default is True).
    label : str or None, optional
        The label text for the axis (default is None).
    pen : str, optional
        The color for the axis line and text (default is 'k' for black).
    ticks : bool, optional
        Whether to show axis ticks (default is True).

    Returns
    -------
    axis : pyqtgraph.AxisItem
        The configured axis object.
    """

    if ax not in {'left', 'right', 'top', 'bottom'}:
        raise ValueError(f"Invalid axis '{ax}'. Must be one of 'left', 'right', 'top', 'bottom'.")

    label = label or ''
    axis = fig.getAxis(ax) if isinstance(fig, pg.PlotItem) else fig.plotItem.getAxis(ax)

    if show:
        axis.show()
        axis.setPen(pen)
        axis.setTextPen(pen)
        axis.setLabel(label)
        if not ticks:
            axis.setTicks([[(0, ''), (0.5, ''), (1, '')]])
    else:
        axis.hide()

    return axis


def set_font(
        fig: Union[pg.PlotItem, pg.PlotWidget],
        ax: str,
        ptsize: int = 8,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> None:

    """
    Set the font size and optionally the axis width/height for a given axis in a PyQtGraph figure.

    Parameters
    ----------
    fig : pyqtgraph.PlotItem or pyqtgraph.PlotWidget
        The figure containing the axis to modify.
    ax : str
        The orientation of the axis. Must be one of {'left', 'right', 'top', 'bottom'}.
    ptsize : int, optional
        Point size for the axis font (default is 8).
    width : int, optional
        Width to set for the axis in pixels. Only applicable for vertical axes.
    height : int, optional
        Height to set for the axis in pixels. Only applicable for horizontal axes.
    """

    if ax not in {'left', 'right', 'top', 'bottom'}:
        raise ValueError(f"Invalid axis '{ax}'. Must be one of 'left', 'right', 'top', 'bottom'.")

    axis = fig.getAxis(ax) if isinstance(fig, pg.PlotItem) else fig.plotItem.getAxis(ax)

    font = QtGui.QFont()
    font.setPointSize(ptsize)
    axis.setStyle(tickFont=font)
    axis.setLabel(**{'font-size': f'{ptsize}pt'})

    if width is not None:
        axis.setWidth(width)
    if height is not None:
        axis.setHeight(height)


def create_line_style(colour=None) -> Tuple[QtGui.QPen, QtGui.QBrush]:
    """
    Generate a random line style (color and dash style) for reference lines.

    Returns
    -------
    pen : QtGui.QPen
        A pen object defining the line color, dash style, and width.
    brush : QtGui.QBrush
        A brush object with the same color as the pen for use with filled items.
    """
    colours = ['#000000', '#cc0000', '#6aa84f', '#1155cc', '#a64d79']
    styles = [QtCore.Qt.SolidLine, QtCore.Qt.DashLine, QtCore.Qt.DashDotLine]

    colour = colour or QtGui.QColor(random.choice(colours))
    style = random.choice(styles)

    pen = pg.mkPen(color=colour, style=style, width=3)
    brush = pg.mkBrush(color=colour)

    return pen, brush

def create_combobox(function, editable=False):

    model = QtGui.QStandardItemModel()
    combobox = QtWidgets.QComboBox()
    combobox.setModel(model)
    if editable:
        combobox.setLineEdit(QtWidgets.QLineEdit())
        completer = QtWidgets.QCompleter()
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        combobox.setCompleter(completer)
        combobox.completer().setModel(model)

    combobox.activated.connect(function)

    return model, combobox

def find_actions(text, action_group):
    for action in action_group.actions():
        if action.text() == text:
            return action


def add_actions(options, function, menu, group, set_checked=True):

    for i, option in enumerate(options):
        if set_checked:
            checked = True if i == 0 else False
        else:
            checked = False

        action = QtWidgets.QAction(option, checkable=True, checked=checked)
        action.triggered.connect(lambda _, o=option: function(o))
        menu.addAction(action)
        group.addAction(action)
        if i == 0:
            action_init = option

    return action_init


def remove_actions(action_group):
    for action in list(action_group.actions()):
        _ = action_group.removeAction(action)
        # TODO deleteLater?
        del _


def create_action_menu(menubar, options, function, title=None, set_checked=True, **kwargs):

    action_menu = kwargs.pop('action_menu', None)
    if action_menu is None:
        action_menu  = menubar.addMenu(title)
    action_group = kwargs.pop('action_group',  None)
    if action_group is None:
        action_group = QtWidgets.QActionGroup(action_menu)
        action_group.setExclusive(True)

    for i, option in enumerate(options):
        if set_checked:
            checked = True if i == 0 else False
        else:
            checked = False

        action = QtWidgets.QAction(option, checkable=True, checked=checked)
        action.triggered.connect(lambda _, o=option: function(o))
        action_menu.addAction(action)
        action_group.addAction(action)
        if i == 0:
            action_init = action

    return action_group, action_menu, action_init


def populate_lists(
        data: List[str],
        list_name: QtGui.QStandardItemModel,
        combobox: QtWidgets.QComboBox
) -> None:
    """
    Populate a combo box and its associated model with a list of string options.

    Parameters
    ----------
    data : List[str]
        A list of strings to add to the widget.
    list_name : QtGui.QStandardItemModel
        The model object to which items will be added.
    combobox : QtWidgets.QComboBox
        The combo box widget to be populated and configured.
    """
    list_name.clear()
    for dat in data:
        item = QtGui.QStandardItem(dat)
        item.setEditable(False)
        list_name.appendRow(item)

    # Ensure the drop-down menu is wide enough to display the longest string
    min_width = combobox.fontMetrics().width(max(data, key=len))
    min_width += combobox.view().autoScrollMargin()
    min_width += combobox.style().pixelMetric(QtWidgets.QStyle.PM_ScrollBarExtent)
    combobox.view().setMinimumWidth(min_width)

    # Set the default selected item to the first option, if available
    combobox.setCurrentIndex(0)


def add_menu(keys, data, callback):
    return


class PopupWindow(QtWidgets.QMainWindow):

    @classmethod
    def _instances(cls):
        app = QtWidgets.QApplication.instance()
        return [w for w in app.topLevelWidgets() if isinstance(w, cls)]

    @classmethod
    def _get_or_create(cls, title: str, **kwargs):
        window = next(filter(lambda e: e.isVisible() and e.windowTitle() == title,
                         cls._instances()), None)
        if window is None:
            window = cls(title, **kwargs)
        else:
            window.showNormal()
            window.activateWindow()
        return window

    closed = QtCore.pyqtSignal(QtWidgets.QMainWindow)
    moved = QtCore.pyqtSignal()

    def __init__(self, title, parent=None, size=(300, 300), graphics=True):
        super(PopupWindow, self).__init__()
        self.parent = parent
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Window)
        self.resize(size[0], size[1])
        self.move(random.randrange(30) + 1000, random.randrange(30) + 200)
        if graphics:
            self.popup_widget = pg.GraphicsLayoutWidget()
        else:
            self.popup_widget = QtWidgets.QWidget()
            self.layout = QtWidgets.QGridLayout()
            self.popup_widget.setLayout(self.layout)

        # TODO figure out need to add into closeEvent of main gui based on popups open
        self.parent.destroyed.connect(self.close)
        self.setCentralWidget(self.popup_widget)
        self.setWindowTitle(title)
        self.setup()
        self.show()

    @abstractmethod
    def setup(self):
        """

        :return:
        """

    def closeEvent(self, event):
        self.closed.emit(self)
        self.close()

    def leaveEvent(self, event):
        self.moved.emit()
