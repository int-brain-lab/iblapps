from PyQt5 import QtCore, QtGui
import pyqtgraph as pg
import matplotlib
import numpy as np


class ColorBar(pg.GraphicsWidget):

    def __init__(self, cmap_name, parent=None):
        pg.GraphicsWidget.__init__(self)

        # Create colour map from matplotlib colourmap name
        self.cmap_name = cmap_name
        cmap = matplotlib.cm.get_cmap(self.cmap_name)
        if type(cmap) == matplotlib.colors.LinearSegmentedColormap:
            cbins = np.linspace(0.0, 1.0, 256)
            colors = (cmap(cbins)[np.newaxis, :, :3][0]).tolist()
        else:
            colors = cmap.colors
        colors = [c + [1.] for c in colors]
        positions = np.linspace(0, 1, len(colors))
        self.map = pg.ColorMap(positions, colors)
        self.lut = self.map.getLookupTable()
        self.grad = self.map.getGradient()

    def getColourMap(self):
        return self.lut

    def makeColourBar(self, width, height, fig, min=0, max=1, label='', lim=False):
        self.cbar = HorizontalBar(width, height, self.grad)
        ax = fig.getAxis('top')
        ax.setPen('k')
        ax.setStyle(stopAxisAtTick=((True, True)))
        # labelStyle = {'font-size': '8pt'}
        ax.setLabel(label)
        ax.setHeight(30)
        if lim:
            ax.setTicks([[(0, str(np.around(min, 2))), (width, str(np.around(max, 2)))]])
        else:
            ax.setTicks([[(0, str(np.around(min, 2))), (width / 2, str(np.around((max - min) / 2, 2))),
                        (width, str(np.around(max, 2)))], [(width / 4, str(np.around((max - min) / 4, 2))), 
                        (3 * width / 4, str(np.around(3 * (max - min) / 4, 2)))]])
        fig.setXRange(0, width)
        fig.setYRange(0, height)

        return self.cbar


class HorizontalBar(pg.GraphicsWidget):
    def __init__(self, width, height, grad):
        pg.GraphicsWidget.__init__(self)
        self.width = width
        self.height = height
        self.grad = grad
        p = QtGui.QPainter()

    def paint(self, p, *args):
        p.setPen(QtCore.Qt.NoPen)
        self.grad.setStart(0, self.height / 2)
        self.grad.setFinalStop(self.width, self.height / 2)
        p.setBrush(pg.QtGui.QBrush(self.grad))
        p.drawRect(QtCore.QRectF(0, 0, self.width, self.height))


class VerticalBar(pg.GraphicsWidget):
    def __init__(self, width, height, grad):
        pg.GraphicsWidget.__init__(self)
        self.width = width
        self.height = height
        self.grad = grad
        p = QtGui.QPainter()

    def paint(self, p, *args):
        p.setPen(QtCore.Qt.NoPen)
        self.grad.setStart(self.width / 2, self.height)
        self.grad.setFinalStop(self.width / 2, 0)
        p.setBrush(pg.QtGui.QBrush(self.grad))
        p.drawRect(QtCore.QRectF(0, 0, self.width, self.height))
