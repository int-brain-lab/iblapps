import pyqtgraph as pg
from pyqtgraph import debug as debug
import numpy as np


class AdaptedAxisItem(pg.AxisItem):
    def __init__(self, orientation, parent=None,):
        pg.AxisItem.__init__(self, orientation, parent=parent)
        self.style['hideOverlappingLabels'] = False

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

    def generateDrawSpecs(self, p):
        """
        Calls tickValues() and tickStrings() to determine where and how ticks should
        be drawn, then generates from this a set of drawing commands to be
        interpreted by drawPicture().
        """
        profiler = debug.Profiler()
        if self.style['tickFont'] is not None:
            p.setFont(self.style['tickFont'])
        bounds = self.mapRectFromParent(self.geometry())

        linkedView = self.linkedView()
        if linkedView is None or self.grid is False:
            tickBounds = bounds
        else:
            tickBounds = linkedView.mapRectToItem(self, linkedView.boundingRect())

        if self.orientation == 'left':
            span = (bounds.topRight(), bounds.bottomRight())
            tickStart = tickBounds.right()
            tickStop = bounds.right()
            tickDir = -1
            axis = 0
        elif self.orientation == 'right':
            span = (bounds.topLeft(), bounds.bottomLeft())
            tickStart = tickBounds.left()
            tickStop = bounds.left()
            tickDir = 1
            axis = 0
        elif self.orientation == 'top':
            span = (bounds.bottomLeft(), bounds.bottomRight())
            tickStart = tickBounds.bottom()
            tickStop = bounds.bottom()
            tickDir = -1
            axis = 1
        elif self.orientation == 'bottom':
            span = (bounds.topLeft(), bounds.topRight())
            tickStart = tickBounds.top()
            tickStop = bounds.top()
            tickDir = 1
            axis = 1
        else:
            raise ValueError("self.orientation must be in ('left', 'right', 'top', 'bottom')")
        # print tickStart, tickStop, span

        ## determine size of this item in pixels
        points = list(map(self.mapToDevice, span))
        if None in points:
            return
        lengthInPixels = pg.Point(points[1] - points[0]).length()
        if lengthInPixels == 0:
            return

        # Determine major / minor / subminor axis ticks
        if self._tickLevels is None:
            tickLevels = self.tickValues(self.range[0], self.range[1], lengthInPixels)
            tickStrings = None
        else:
            ## parse self.tickLevels into the formats returned by tickLevels() and tickStrings()
            tickLevels = []
            tickStrings = []
            for level in self._tickLevels:
                values = []
                strings = []
                tickLevels.append((None, values))
                tickStrings.append(strings)
                for val, strn in level:
                    values.append(val)
                    strings.append(strn)

        ## determine mapping between tick values and local coordinates
        dif = self.range[1] - self.range[0]
        if dif == 0:
            xScale = 1
            offset = 0
        else:
            if axis == 0:
                xScale = -bounds.height() / dif
                offset = self.range[0] * xScale - bounds.height()
            else:
                xScale = bounds.width() / dif
                offset = self.range[0] * xScale

        xRange = [x * xScale - offset for x in self.range]
        xMin = min(xRange)
        xMax = max(xRange)

        profiler('init')

        tickPositions = []  # remembers positions of previously drawn ticks

        ## compute coordinates to draw ticks
        ## draw three different intervals, long ticks first
        tickSpecs = []
        for i in range(len(tickLevels)):
            tickPositions.append([])
            ticks = tickLevels[i][1]

            ## length of tick
            tickLength = self.style['tickLength'] / ((i * 0.5) + 1.0)

            lineAlpha = self.style["tickAlpha"]
            if lineAlpha is None:
                lineAlpha = 255 / (i + 1)
                if self.grid is not False:
                    lineAlpha *= self.grid / 255. * pg.functions.clip_scalar((0.05 * lengthInPixels / (len(ticks) + 1)), 0., 1.)
            elif isinstance(lineAlpha, float):
                lineAlpha *= 255
                lineAlpha = max(0, int(round(lineAlpha)))
                lineAlpha = min(255, int(round(lineAlpha)))
            elif isinstance(lineAlpha, int):
                if (lineAlpha > 255) or (lineAlpha < 0):
                    raise ValueError("lineAlpha should be [0..255]")
            else:
                raise TypeError("Line Alpha should be of type None, float or int")

            for v in ticks:
                ## determine actual position to draw this tick
                x = (v * xScale) - offset
                if x < xMin or x > xMax:  ## last check to make sure no out-of-bounds ticks are drawn
                    tickPositions[i].append(None)
                    continue
                tickPositions[i].append(x)

                p1 = [x, x]
                p2 = [x, x]
                p1[axis] = tickStart
                p2[axis] = tickStop
                if self.grid is False:
                    p2[axis] += tickLength * tickDir
                tickPen = self.pen()
                color = tickPen.color()
                color.setAlpha(int(lineAlpha))
                tickPen.setColor(color)
                tickSpecs.append((tickPen, pg.Point(p1), pg.Point(p2)))
        profiler('compute ticks')

        if self.style['stopAxisAtTick'][0] is True:
            minTickPosition = min(map(min, tickPositions))
            if axis == 0:
                stop = max(span[0].y(), minTickPosition)
                span[0].setY(stop)
            else:
                stop = max(span[0].x(), minTickPosition)
                span[0].setX(stop)
        if self.style['stopAxisAtTick'][1] is True:
            maxTickPosition = max(map(max, tickPositions))
            if axis == 0:
                stop = min(span[1].y(), maxTickPosition)
                span[1].setY(stop)
            else:
                stop = min(span[1].x(), maxTickPosition)
                span[1].setX(stop)
        axisSpec = (self.pen(), span[0], span[1])

        textOffset = self.style['tickTextOffset'][axis]  ## spacing between axis and text
        # if self.style['autoExpandTextSpace'] is True:
        # textWidth = self.textWidth
        # textHeight = self.textHeight
        # else:
        # textWidth = self.style['tickTextWidth'] ## space allocated for horizontal text
        # textHeight = self.style['tickTextHeight'] ## space allocated for horizontal text

        textSize2 = 0
        lastTextSize2 = 0
        textRects = []
        textSpecs = []  ## list of draw

        # If values are hidden, return early
        if not self.style['showValues']:
            return (axisSpec, tickSpecs, textSpecs)

        for i in range(min(len(tickLevels), self.style['maxTextLevel'] + 1)):
            ## Get the list of strings to display for this level
            if tickStrings is None:
                spacing, values = tickLevels[i]
                strings = self.tickStrings(values, self.autoSIPrefixScale * self.scale, spacing)
            else:
                strings = tickStrings[i]

            if len(strings) == 0:
                continue

            ## ignore strings belonging to ticks that were previously ignored
            for j in range(len(strings)):
                if tickPositions[i][j] is None:
                    strings[j] = None

            ## Measure density of text; decide whether to draw this level
            rects = []
            for s in strings:
                if s is None:
                    rects.append(None)
                else:
                    br = p.boundingRect(pg.QtCore.QRectF(0, 0, 100, 100), pg.QtCore.Qt.AlignmentFlag.AlignCenter, s)
                    ## boundingRect is usually just a bit too large
                    ## (but this probably depends on per-font metrics?)
                    br.setHeight(br.height() * 0.8)

                    rects.append(br)
                    textRects.append(rects[-1])

            if len(textRects) > 0:
                ## measure all text, make sure there's enough room
                if axis == 0:
                    textSize = np.sum([r.height() for r in textRects])
                    textSize2 = np.max([r.width() for r in textRects])
                else:
                    textSize = np.sum([r.width() for r in textRects])
                    textSize2 = np.max([r.height() for r in textRects])
            else:
                textSize = 0
                textSize2 = 0

            if i > 0:  ## always draw top level
                ## If the strings are too crowded, stop drawing text now.
                ## We use three different crowding limits based on the number
                ## of texts drawn so far.
                textFillRatio = float(textSize) / lengthInPixels
                finished = False
                for nTexts, limit in self.style['textFillLimits']:
                    if len(textSpecs) >= nTexts and textFillRatio >= limit:
                        finished = True
                        break
                if finished:
                    break

            lastTextSize2 = textSize2

            # spacing, values = tickLevels[best]
            # strings = self.tickStrings(values, self.scale, spacing)
            # Determine exactly where tick text should be drawn
            for j in range(len(strings)):
                vstr = strings[j]
                if vstr is None:  ## this tick was ignored because it is out of bounds
                    continue
                x = tickPositions[i][j]
                # textRect = p.boundingRect(QtCore.QRectF(0, 0, 100, 100), QtCore.Qt.AlignmentFlag.AlignCenter, vstr)
                textRect = rects[j]
                height = textRect.height()
                width = textRect.width()
                # self.textHeight = height
                offset = max(0, self.style['tickLength']) + textOffset

                rect = pg.QtCore.QRectF()
                if self.orientation == 'left':
                    alignFlags = pg.QtCore.Qt.AlignmentFlag.AlignRight | pg.QtCore.Qt.AlignmentFlag.AlignVCenter
                    rect = pg.QtCore.QRectF(tickStop - offset - width, x - (height / 2), width, height)
                elif self.orientation == 'right':
                    alignFlags = pg.QtCore.Qt.AlignmentFlag.AlignLeft | pg.QtCore.Qt.AlignmentFlag.AlignVCenter
                    rect = pg.QtCore.QRectF(tickStop + offset, x - (height / 2), width, height)
                elif self.orientation == 'top':
                    alignFlags = pg.QtCore.Qt.AlignmentFlag.AlignHCenter | pg.QtCore.Qt.AlignmentFlag.AlignBottom
                    rect = pg.QtCore.QRectF(x - width / 2., tickStop - offset - height, width, height)
                elif self.orientation == 'bottom':
                    alignFlags = pg.QtCore.Qt.AlignmentFlag.AlignHCenter | pg.QtCore.Qt.AlignmentFlag.AlignTop
                    rect = pg.QtCore.QRectF(x - width / 2., tickStop + offset, width, height)

                textFlags = alignFlags | pg.QtCore.Qt.TextFlag.TextDontClip
                # p.setPen(self.pen())
                # p.drawText(rect, textFlags, vstr)

                # br = self.boundingRect()
                # if not br.contains(rect):
                #     continue

                textSpecs.append((rect, textFlags, vstr))
        profiler('compute text')

        ## update max text size if needed.
        self._updateMaxTextSize(lastTextSize2)

        return (axisSpec, tickSpecs, textSpecs)


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
