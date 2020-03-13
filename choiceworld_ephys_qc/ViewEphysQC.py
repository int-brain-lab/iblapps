import logging

from PyQt5 import QtCore, QtWidgets
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import pandas as pd

import choiceworld_ephys_qc.qt as qt

_logger = logging.getLogger('ibllib')


class DataFrameModel(QtCore.QAbstractTableModel):
    DtypeRole = QtCore.Qt.UserRole + 1000
    ValueRole = QtCore.Qt.UserRole + 1001

    def __init__(self, df=pd.DataFrame(), parent=None):
        super(DataFrameModel, self).__init__(parent)
        self._dataframe = df

    def setDataFrame(self, dataframe):
        self.beginResetModel()
        self._dataframe = dataframe.copy()
        self.endResetModel()

    def dataFrame(self):
        return self._dataframe

    dataFrame = QtCore.pyqtProperty(pd.DataFrame, fget=dataFrame, fset=setDataFrame)

    @QtCore.pyqtSlot(int, QtCore.Qt.Orientation, result=str)
    def headerData(self, section: int, orientation: QtCore.Qt.Orientation,
                   role: int = QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._dataframe.columns[section]
            else:
                return str(self._dataframe.index[section])
        return QtCore.QVariant()

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._dataframe.index)

    def columnCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return self._dataframe.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if (not index.isValid() or not (0 <= index.row() < self.rowCount() and
                                        0 <= index.column() < self.columnCount())):
            return QtCore.QVariant()
        row = self._dataframe.index[index.row()]
        col = self._dataframe.columns[index.column()]
        dt = self._dataframe[col].dtype

        val = self._dataframe.iloc[row][col]
        if role == QtCore.Qt.DisplayRole:
            return str(val)
        elif role == DataFrameModel.ValueRole:
            return val
        if role == DataFrameModel.DtypeRole:
            return dt
        return QtCore.QVariant()

    def roleNames(self):
        roles = {
            QtCore.Qt.DisplayRole: b'display',
            DataFrameModel.DtypeRole: b'dtype',
            DataFrameModel.ValueRole: b'value'
        }
        return roles


class PlotCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvasQTAgg.__init__(self, fig)
        self.setParent(parent)

        FigureCanvasQTAgg.setSizePolicy(
            self,
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)
        self.ax = self.figure.add_subplot(111)
        self.draw()


class PlotWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent=None)
        self.canvas = PlotCanvas()
        self.vbl = QtWidgets.QVBoxLayout()         # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)
        self.vbl.addWidget(NavigationToolbar2QT(self.canvas, self))

class GraphWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent=None)
        vLayout = QtWidgets.QVBoxLayout(self)
        hLayout = QtWidgets.QHBoxLayout()
        self.pathLE = QtWidgets.QLineEdit(self)
        hLayout.addWidget(self.pathLE)
        self.loadBtn = QtWidgets.QPushButton("Select File", self)
        hLayout.addWidget(self.loadBtn)
        vLayout.addLayout(hLayout)
        self.pandasTv = QtWidgets.QTableView(self)
        vLayout.addWidget(self.pandasTv)
        self.loadBtn.clicked.connect(self.loadFile)
        self.pandasTv.setSortingEnabled(True)
        self.pandasTv.doubleClicked.connect(self.tv_double_clicked)
        self.wplot = PlotWindow()
        self.wplot.show()
        print("toto")

    def loadFile(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", "",
                                                            "CSV Files (*.csv)")
        self.pathLE.setText(fileName)
        df = pd.read_csv(fileName)
        self.update_df(df)

    def update_df(self, df):
        model = DataFrameModel(df)
        self.pandasTv.setModel(model)
        self.wplot.canvas.draw()

    def tv_double_clicked(self):
        df = self.pandasTv.model()._dataframe
        ind = self.pandasTv.currentIndex()
        start = df.loc[ind.row()]['intervals_0']
        finish = df.loc[ind.row()]['intervals_1']
        dt = finish - start
        self.wplot.canvas.ax.set_xlim(start - dt / 10, finish + dt / 10)
        self.wplot.canvas.draw()


def viewqc(qc=None, title=None):
    qt.create_app()
    qcw = GraphWindow()
    qcw.setWindowTitle(title)
    if qc is not None:
        qcw.update_df(qc)
    qcw.show()
    return qcw
