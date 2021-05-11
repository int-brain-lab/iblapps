import pyqtgraph as pg
from pyqtgraph import debug as debug


class AdaptedAxisItem(pg.AxisItem):
    def __init__(self, orientation, parent=None,):
        pg.AxisItem.__init__(self, orientation, parent=parent)

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        profiler = debug.Profiler()

        p.setRenderHint(p.Antialiasing, False)
        p.setRenderHint(p.TextAntialiasing, True)

        # draw long line along axis
        pen, p1, p2 = axisSpec
        p.setPen(pen)
        p.drawLine(p1, p2)
        p.translate(0.5, 0)  # resolves some damn pixel ambiguity

        # draw ticks
        for pen, p1, p2 in tickSpecs:
            p.setPen(pen)
            p.drawLine(p1, p2)
        profiler('draw ticks')

        # Draw all text
        if self.style['tickFont'] is not None:
            p.setFont(self.style['tickFont'])
        p.setPen(self.textPen())
        for rect, flags, text in textSpecs:
            p.drawText(rect, int(flags), text)

        profiler('draw text')


def replace_axis(plot_item, orientation='left', pos=(2, 0)):

    new_axis = AdaptedAxisItem(orientation, parent=plot_item)
    oldAxis = plot_item.axes[orientation]['item']
    plot_item.layout.removeItem(oldAxis)
    oldAxis.unlinkFromView()
    #
    new_axis.linkToView(plot_item.vb)
    plot_item.axes[orientation] = {'item': new_axis, 'pos': pos}
    plot_item.layout.addItem(new_axis, *pos)
    new_axis.setZValue(-1000)
    new_axis.setFlag(new_axis.ItemNegativeZStacksBehindParent)
