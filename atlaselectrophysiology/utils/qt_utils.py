import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from typing import Any, Union, Optional, List, Dict, Tuple, Callable
import random
from abc import ABC, abstractmethod


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


def create_line_style() -> Tuple[QtGui.QPen, QtGui.QBrush]:
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

    colour = QtGui.QColor(random.choice(colours))
    style = random.choice(styles)

    pen = pg.mkPen(color=colour, style=style, width=3)
    brush = pg.mkBrush(color=colour)

    return pen, brush



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

    # Ideally the get create would go here

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
