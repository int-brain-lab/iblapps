import pyqtgraph as pg
from matplotlib import cm
from PyQt5 import QtCore, QtGui
import numpy as np


class ColorBar(pg.GraphicsWidget):
    
    def __init__(self, cmap_name, parent=None):
        pg.GraphicsWidget.__init__(self)

        # Create colour map from matplotlib colourmap name
        self.cmap_name = cmap_name
        cmap = cm.get_cmap(self.cmap_name)
        colors = cmap.colors
        colors = [c + [1.] for c in colors]
        positions = np.linspace(0, 1, len(colors))
        self.map = pg.ColorMap(positions, colors)
        self.lut = self.map.getLookupTable()
        self.grad = self.map.getGradient()

        #Create colour bar object
        self.layout = QtGui.QGraphicsGridLayout()
        #self.layout = pg.GraphicsLayout()
        self.layout.setContentsMargins(1, 1, 1, 1)
        self.setLayout(self.layout)
        self.layout.setHorizontalSpacing(0)
        self.layout.setVerticalSpacing(0)
  
    def getColourMap(self):
        return self.lut

    def makeHColourBar(self, width, height, scale=10, min=0, max=1, label='', lim=False):
        width = width / scale
        height = height / scale
        self.bar = HorizontalBar(width, height, self.grad)
        self.axis = pg.AxisItem(orientation='top', parent=self, pen='k')
        self.label = pg.LabelItem()
        self.layout.addItem(self.bar, 2, 1)
        self.layout.addItem(self.axis, 1, 1)
        self.layout.addItem(self.label, 0, 1)
        self.axis.setWidth(width)
        self.axis.setRange(min, max)
        if lim:
            self.axis.setTicks([[(min, str(np.around(min, 1))), (max, str(np.around(max, 1)))]])
        #self.axis.setLabel('')
        self.label.setText(label, color='k', bold=True)
        self.scale(scale, -1 * scale)
    
    def makeVColourBar(self, width, height, scale=10, min=0, max=1, label='', lim=False):
        width = width / scale
        height = height / scale
        self.bar = VerticalBar(width, height, self.grad)
        self.axis = pg.AxisItem(orientation='left', parent=self, pen='k')
        self.label = pg.LabelItem()
        self.layout.addItem(self.bar, 1, 1)
        self.layout.addItem(self.axis, 1, 0)
        self.layout.addItem(self.label, 0, 0, 1, 2)
        self.axis.setHeight(height)
        self.axis.setRange(min, max)
        if lim:
            self.axis.setTicks([[(min, str(np.around(min, 1))), (max, str(np.around(max, 1)))]])
        #self.axis.setLabel('')
        self.label.setText(label, color='k', bold=True)
        self.scale(scale, -1 * scale)

    def getColourBarAxis(self):
        return self.axis

class HorizontalBar(pg.GraphicsWidget):

    def __init__(self, width, height, grad):
        pg.GraphicsWidget.__init__(self)
        self.width = width
        self.height = height
        self.grad = grad
        
        self.vb = pg.ViewBox(parent=self)
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
        
        self.vb = pg.ViewBox(parent=self)
        p = QtGui.QPainter()
        
    def paint(self, p, *args):
        p.setPen(QtCore.Qt.NoPen)
        self.grad.setStart(self.width / 2, self.height)
        self.grad.setFinalStop(self.width / 2, 0)
        p.setBrush(pg.QtGui.QBrush(self.grad))
        p.drawRect(QtCore.QRectF(0, 0, self.width, self.height))





        

     