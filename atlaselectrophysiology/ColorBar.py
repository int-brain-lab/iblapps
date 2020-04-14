import pyqtgraph as pg
from matplotlib import cm
from PyQt5 import QtWidgets, QtCore, QtGui
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
        self.layout.setContentsMargins(1, 1, 1, 1)
        self.setLayout(self.layout)
        self.layout.setHorizontalSpacing(0)
        self.layout.setVerticalSpacing(0)
        self.axis = pg.AxisItem(orientation='top', parent=self, pen='k')


    
        
    def getColourMap(self):
        return self.lut

    def makeColourBar(self, width, height, min = 0, max = 1, label=''):
        width = width/10
        height = height/10
        self.bar = Bar(width, height, self.grad)
        self.layout.addItem(self.bar, 2, 1)
        self.layout.addItem(self.axis, 1, 1)
        self.axis.setWidth(width)
        self.axis.setRange(min, max)
        self.axis.setLabel(label)
    
        #self.rotate(180)
        self.scale(10, -10)


    def getColourBarAxis(self):
        return self.axis



        


class Bar(pg.GraphicsWidget):
    
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





        

     