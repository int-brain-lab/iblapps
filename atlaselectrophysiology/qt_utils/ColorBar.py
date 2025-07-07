from qtpy import QtCore, QtGui
import pyqtgraph as pg
import matplotlib
import numpy as np
from pyqtgraph.functions import makeARGB


class ColorBar(pg.GraphicsWidget):

    def __init__(self, cmap_name, width=20, height=5, plot_item=None, cbin=256, orientation='horizontal'):
        pg.GraphicsWidget.__init__(self)

        # Set dimensions
        self.width = width
        self.height = height

        # Set orientation
        self.orientation = orientation

        # Create colour map from matplotlib colourmap name
        self.cmap_name = cmap_name
        self.map, self.lut, self.grad = self.getColor(self.cmap_name, cbin=cbin)

        # Create plot item to place the colorbar
        self.plot = plot_item
        if self.plot:
            self.plot.setXRange(0, self.width)
            self.plot.setYRange(0, self.height)
            self.plot.addItem(self)

        QtGui.QPainter()

        self.ticks = None


    @staticmethod
    def getColor(cmap_name, cbin=256):
        cmap = matplotlib.cm.get_cmap(cmap_name)
        if type(cmap) == matplotlib.colors.LinearSegmentedColormap:
            cbins = np.linspace(0.0, 1.0, cbin)
            colors = (cmap(cbins)[np.newaxis, :, :3][0]).tolist()
        else:
            colors = cmap.colors
        colors = [(np.array(c) * 255).astype(int).tolist() + [255.] for c in colors]
        positions = np.linspace(0, 1, len(colors))
        map = pg.ColorMap(positions, colors)
        lut = map.getLookupTable()
        grad = map.getGradient()

        return map, lut, grad


    def paint(self, p, *args):
        p.setPen(QtCore.Qt.NoPen)
        self.grad.setStart(0, self.height / 2)
        self.grad.setFinalStop(self.width, self.height / 2)
        p.setBrush(pg.QtGui.QBrush(self.grad))
        p.drawRect(QtCore.QRectF(0, 0, self.width, self.height))

    def getBrush(self, data, levels=None):
        if levels is None:
            levels = [np.min(data), np.max(data)]
        brush_rgb, _ = makeARGB(data[:, np.newaxis], levels=levels, lut=self.lut, useRGBA=True)
        brush = [QtGui.QColor(*col) for col in np.squeeze(brush_rgb)]
        return brush

    def getColourMap(self):
        return self.lut

    def setLevels(self, levels, label=None, nticks=2):
        self.levels = levels
        self.ticks = self.get_ticks(nticks)
        self.label = label
        self.setAxis(ticks=self.ticks, label=label)

    def setAxis(self, ticks=None, label=None, loc=None, extent=30):

        if loc is None:
            loc = 'top' if self.orientation == 'horizontal' else 'left'
        ticks = ticks
        ax = self.plot.getAxis(loc)
        ax.show()
        ax.setStyle(stopAxisAtTick=((True, True)), autoExpandTextSpace=True)
        if self.orientation == 'horizontal':
            ax.setHeight(extent)
        else:
            ax.setWidth(extent)
        if ticks:
            ax.setPen('k')
            ax.setTextPen('k')
            ax.setTicks([ticks])
        else:
            ax.setTextPen('w')
            ax.setPen('w')
        # Note this has to come after the setPen above otherwise overwritten
        ax.setLabel(label, color='k')

    def get_ticks(self, n=3):
        """
        Set ticks on a given axis, either with just min/max if `lim` is True,
        or evenly spaced `n` ticks otherwise.
        """
        extent = self.width if self.orientation == 'horizontal' else self.height
        offset = (0.005 * extent)

        # Set n number of ticks
        ticks = []
        for i in range(n):
            frac = i / (n - 1)
            pos = frac * extent
            val = self.levels[0] + frac * (self.levels[1] - self.levels[0])
            val = int(val) if np.abs(val) > 1 else np.round(val, 1)
            if i == 0:
                pos += offset
            elif i == n-1:
                pos -= offset
            ticks.append((pos, str(val)))

        return ticks
