import pyqtgraph as pg
import numpy as np
from typing import Dict, Optional, Any
from qtpy import QtGui
import atlaselectrophysiology.qt_utils.utils as utils
import atlaselectrophysiology.qt_utils.ColorBar as cb


def set_xaxis_range(fig, data, label=True):
    fig.setXRange(min=data['xrange'][0], max=data['xrange'][1], padding=0)
    if label:
        utils.set_axis(fig, 'bottom', label=data['xaxis'])


def set_yaxis_range(fig, shank_items):
    fig.setYRange(min=shank_items.probe_tip - shank_items.probe_extra, max=shank_items.probe_top +
                                                                           shank_items.probe_extra, padding=shank_items.pad)




def plot_histology(fig: pg.PlotWidget, data: Dict[str, Any], shank_items, ax: str ='left') -> None:
    """
    Plot histology regions on the given figure, shows brain regions intersecting with the probe track.

    Parameters
    ----------
    fig : pyqtgraph.PlotWidget
        The figure widget on which to plot the histology regions.
    data : dict
        A dictionary containing histology data with the following keys:
        - 'axis_label' : list of tuple(float, str)
            Tick labels and positions for the axis.
        - 'region' : list of tuple(float, float)
            Start and end coordinates for each histology region in micrometres.
        - 'colour' : list of tuple(int, int, int)
            RGB color tuples for each region.
    ax : 'left' or 'right', optional
        Orientation of the axis on which to add labels. 'left' for the main histology figure (fig_hist),
        and 'right' for the reference figure (fig_hist_ref). Default is 'left'.
    """
    # TODO more controlled???
    fig.clear()
    shank_items.hist_regions[ax] = np.empty((0, 1), dtype=object)

    # Configure axis ticks and appearance
    axis = fig.getAxis(ax)
    axis.setTicks([data['axis_label']])
    axis.setZValue(10)
    utils.set_axis(fig, 'bottom', pen='w', label='blank')

    # Plot each histology region
    for ir, reg in enumerate(data['region']):
        colour = QtGui.QColor(*data['colour'][ir])
        region = pg.LinearRegionItem(values=(reg[0], reg[1]),
                                     orientation=pg.LinearRegionItem.Horizontal,
                                     brush=colour,
                                     movable=False)
        bound = pg.InfiniteLine(pos=reg[0], angle=0, pen='w')
        fig.addItem(region)
        fig.addItem(bound)
        # Keep track of each histology LinearRegionItem for label pressed interaction
        shank_items.hist_regions[ax] = np.vstack([shank_items.hist_regions[ax], region])

    # Add additional boundary for final region
    bound = pg.InfiniteLine(pos=data['region'][-1][1], angle=0, pen='w')
    fig.addItem(bound)

    # Add dotted probe track limits (tip and top) as horizontal reference lines

    set_yaxis_range(fig, shank_items)




def plot_scale_factor(shank_items) -> None:
    """
    Plot the scale factor applied to brain regions along the probe track, displayed alongside
    the histology figure. This visualizes regional scale adjustments as colored horizontal bands.
    """

    shank_items.fig_scale.clear()
    shank_items.scale_regions = np.empty((0, 1))
    shank_items.scale_factor = shank_items.scale_data['scale']
    scale_factor = shank_items.scale_data['scale'] - 0.5

    color_bar = cb.ColorBar('seismic')
    cbar = color_bar.makeColourBar(20, 5, shank_items.fig_scale_cb, min=0.5, max=1.5, label='Scale Factor')
    colours = color_bar.map.mapToQColor(scale_factor)
    shank_items.fig_scale_cb.addItem(cbar)

    for ir, reg in enumerate(shank_items.scale_data['region']):
        region = pg.LinearRegionItem(values=(reg[0], reg[1]),
                                     orientation=pg.LinearRegionItem.Horizontal,
                                     brush=colours[ir], movable=False)
        bound = pg.InfiniteLine(pos=reg[0], angle=0, pen=colours[ir])

        shank_items.fig_scale.addItem(region)
        shank_items.fig_scale.addItem(bound)
        shank_items.scale_regions = np.vstack([shank_items.scale_regions, region])

    bound = pg.InfiniteLine(pos=shank_items.scale_data['region'][-1][1], angle=0, pen=colours[-1])
    shank_items.fig_scale.addItem(bound)

    set_yaxis_range(shank_items.fig_scale, shank_items)
    utils.set_axis(shank_items.fig_scale, 'bottom', pen='w', label='blank')

    return shank_items


def plot_slice_orig(shank_items, shank_data, data: Dict):
    """
    Plot a slice image representing histology data, with optional label or intensity image.

    Displays the histology slice on the figure widget with appropriate color mapping,
    overlays trajectory lines, and updates channel plotting.

    Parameters
    ----------
    data : dict
        Dictionary containing slice image data and metadata. Expected keys include:
        - 'label' (np.ndarray): image data for annotation slice
        - 'hist_rd', 'hist_gr', 'ccf' (np.ndarray): image data for histology slice
        - 'scale' (tuple[float, float]): scaling factors for x and y axes
        - 'offset' (tuple[float, float]): offsets for x and y axes
    img_type : 'label', 'hist_rd', 'hist_gr', 'hist_cb' or 'ccf'
        The key of the image data to plot. If 'label', displays a labeled slice without LUT.
        For other types, a color lookup table and histogram are applied.
    """

    # If no histology we can't do alignment

    # Clear previous display
    shank_items.fig_slice.clear()
    shank_items.slice_chns = pg.ScatterPlotItem()
    shank_items.slice_lines = []

    # Create image item with data to display
    img = pg.ImageItem()
    img.setImage(data['slice'])

    # Construct QTransform from scale and offset data
    transform = [data['scale'][0], 0., 0., 0.,
                 data['scale'][1], 0.,
                 data['offset'][0], data['offset'][1], 1.]
    img.setTransform(QtGui.QTransform(*transform))

    label_img = data.get('label', False)
    if label_img:
        shank_items.fig_slice_layout.removeItem(shank_items.slice_item)
        shank_items.fig_slice_layout.addItem(shank_items.fig_slice_hist_alt, 0, 1)
        shank_items.slice_item = shank_items.fig_slice_hist_alt
    else:
        color_bar = cb.ColorBar('cividis')
        lut = color_bar.getColourMap()
        img.setLookupTable(lut)

        shank_items.fig_slice_layout.removeItem(shank_items.slice_item)
        shank_items.fig_slice_hist = pg.HistogramLUTItem()
        shank_items.fig_slice_hist.axis.hide()
        shank_items.fig_slice_hist.setImageItem(img)
        shank_items.fig_slice_hist.gradient.setColorMap(color_bar.map)
        shank_items.fig_slice_hist.autoHistogramRange()
        shank_items.fig_slice_layout.addItem(shank_items.fig_slice_hist, 0, 1)
        hist_levels = shank_items.fig_slice_hist.getLevels()
        hist_val, hist_count = img.getHistogram()
        upper_idx = np.where(hist_count > 10)[0][-1]
        upper_val = hist_val[upper_idx]
        if hist_levels[0] != 0:
            shank_items.fig_slice_hist.setLevels(min=hist_levels[0], max=upper_val)
        shank_items.slice_item = shank_items.fig_slice_hist

    shank_items.fig_slice.addItem(img)

    # Plot the trajectory line
    shank_items.traj_line = pg.PlotCurveItem()
    shank_items.traj_line.setData(x=shank_data.loaders['align'].xyz_track[:, 0],
                                  y=shank_data.loaders['align'].xyz_track[:, 2], pen=utils.kpen_solid)
    shank_items.fig_slice.addItem(shank_items.traj_line)

    # Plot channels
    shank_items.fig_slice.addItem(shank_items.slice_chns)
    plot_channels(shank_items, shank_data)


def plot_slice(shank_items, shank_data, data: Dict):
    """
    Plot a slice image representing histology data, with optional label or intensity image.

    Displays the histology slice on the figure widget with appropriate color mapping,
    overlays trajectory lines, and updates channel plotting.

    Parameters
    ----------
    data : dict
        Dictionary containing slice image data and metadata. Expected keys include:
        - 'label' (np.ndarray): image data for annotation slice
        - 'hist_rd', 'hist_gr', 'ccf' (np.ndarray): image data for histology slice
        - 'scale' (tuple[float, float]): scaling factors for x and y axes
        - 'offset' (tuple[float, float]): offsets for x and y axes
    img_type : 'label', 'hist_rd', 'hist_gr', 'hist_cb' or 'ccf'
        The key of the image data to plot. If 'label', displays a labeled slice without LUT.
        For other types, a color lookup table and histogram are applied.
    """

    # If no histology we can't do alignment

    # Clear previous display
    shank_items.fig_slice.clear()
    shank_items.slice_chns = pg.ScatterPlotItem()
    shank_items.slice_lines = []

    # Create image item with data to display
    img = pg.ImageItem()
    img.setImage(data['slice'])

    # Construct QTransform from scale and offset data
    transform = [data['scale'][0], 0., 0., 0.,
                 data['scale'][1], 0.,
                 data['offset'][0], data['offset'][1], 1.]
    img.setTransform(QtGui.QTransform(*transform))

    label_img = data.get('label', False)
    if not label_img:
        color_bar = cb.ColorBar('cividis')
        lut = color_bar.getColourMap()
        img.setLookupTable(lut)
    else:
        color_bar = None

    shank_items.fig_slice.addItem(img)
    shank_items.fig_slice.autoRange()

    # Plot the trajectory line
    shank_items.traj_line = pg.PlotCurveItem()
    shank_items.traj_line.setData(x=shank_data.loaders['align'].xyz_track[:, 0],
                                  y=shank_data.loaders['align'].xyz_track[:, 2], pen=utils.kpen_solid)
    shank_items.fig_slice.addItem(shank_items.traj_line)

    # Plot channels
    shank_items.fig_slice.addItem(shank_items.slice_chns)
    plot_channels(shank_items, shank_data)

    return img,  color_bar




def plot_channels(shank_items, shank_data) -> None:
    """
    Plot electrode channels and alignment lines on the histological slice.

    Displays the locations of electrode channels and probe track reference lines
    (alignment lines) on the histology slice view.
    """

    # If no histology we can't do alignment

    xyz_channels = shank_data.loaders['align'].xyz_channels
    track_lines = shank_data.loaders['align'].track_lines

    # Clear existing reference lines
    shank_items.slice_lines = utils.remove_items(shank_items.fig_slice, shank_items.slice_lines)

    shank_items.slice_chns.setData(x=xyz_channels[:, 0], y=xyz_channels[:, 2], pen='r', brush='r')

    for ref_line in track_lines:
        line = pg.PlotCurveItem()
        line.setData(x=ref_line[:, 0], y=ref_line[:, 2], pen=utils.kpen_dot)
        shank_items.fig_slice.addItem(line)
        shank_items.slice_lines.append(line)



def plot_line(shank_items, data: Optional[Dict]) -> None:
    """
    Plot a 1D line plot of electrophysiological data on the figure.

    Parameters
    ----------
    data : dict, optional
        Dictionary containing the line plot data and display parameters. Expected keys:
            - 'x' : np.ndarray of float
                X-coordinates of the data points.
            - 'y' : np.ndarray of float
                Y-coordinates of the data points.
            - 'xrange' : tuple[float, float]
                Display range for the x-axis.
            - 'xaxis' : str
                Label for the x-axis.
    """

    if not data:
        print('data for this plot not available')
        return
    # Clear existing plots
    shank_items.line_plots = utils.remove_items(shank_items.fig_line, shank_items.line_plots)

    # Create line plot at add to figure
    line = pg.PlotCurveItem()
    line.setData(x=data['x'], y=data['y'])
    line.setPen(utils.kpen_solid)
    shank_items.fig_line.addItem(line)
    shank_items.line_plots.append(line)

    # Set axis ranges
    set_xaxis_range(shank_items.fig_line, data)
    set_yaxis_range(shank_items.fig_line, shank_items)


def plot_probe(shank_items, data: Optional[Dict]) -> None:
    """
    Plot a 2D image representing the probe geometry, including individual channel banks and optional boundaries.

    Parameters
    ----------
    data : dict
        Dictionary containing information to plot the probe geometry.
        Expected keys:
            img : list of np.ndarray
                Image data arrays, one for each channel bank.
            scale : list of np.ndarray
                Scaling to apply to each image, as [xscale, yscale].
            offset : list of np.ndarray
                Offset to apply to each image, as [xoffset, yoffset].
                Offset to apply to each image, as [xoffset, yoffset]
            levels : np.ndarray
                Two-element array specifying color bar limits [min, max].
            cmap : str
                Colormap to use.
            xrange : np.ndarray
                Two-element array for x-axis limits [min, max].
            title : str
                Label for the color bar.
    """
    if not data:
        print('data for this plot not available')
        return

    shank_items.probe_plots = utils.remove_items(shank_items.fig_probe, shank_items.probe_plots)
    shank_items.probe_cbars = utils.remove_items(shank_items.fig_probe_cb, shank_items.probe_cbars)
    shank_items.probe_bounds = utils.remove_items(shank_items.fig_probe, shank_items.probe_bounds)

    # Create colorbar and add to figure
    color_bar = cb.ColorBar(data['cmap'])
    lut = color_bar.getColourMap()
    cbar = color_bar.makeColourBar(20, 5, shank_items.fig_probe_cb, min=data['levels'][0],
                                   max=data['levels'][1], label=data['title'], lim=False)
    shank_items.fig_probe_cb.addItem(cbar)
    shank_items.probe_cbars.append(cbar)

    # Create image plots per shank and add to figure
    for img, scale, offset in zip(data['img'], data['scale'], data['offset']):
        image = pg.ImageItem()
        image.setImage(img)
        transform = [scale[0], 0., 0., 0., scale[1], 0., offset[0], offset[1], 1.]
        image.setTransform(QtGui.QTransform(*transform))
        image.setLookupTable(lut)
        image.setLevels((data['levels'][0], data['levels'][1]))
        shank_items.fig_probe.addItem(image)
        shank_items.probe_plots.append(image)

    # Set axis ranges
    set_xaxis_range(shank_items.fig_probe, data, label=False)
    set_yaxis_range(shank_items.fig_probe, shank_items)
    # Add in a fake label so that the appearence is the same as other plots
    utils.set_axis(shank_items.fig_probe, 'bottom', pen='w', label='blank')

    # Optionally plot horizontal boundary lines
    bounds = data.get('boundaries', None)
    if bounds is not None:
        for bound in bounds:
            line = pg.InfiniteLine(pos=bound, angle=0, pen='w')
            shank_items.fig_probe.addItem(line)
            shank_items.probe_bounds.append(line)


def plot_image(shank_items, data: Optional[Dict]) -> None:
    """
    Plot a 2D image of electrophysiology data

    Parameters
    ----------
    data : dict
        Dictionary containing image and display metadata. Expected keys:
            img : np.ndarray
                2D image array of shape (nx, ny), representing the electrophysiological data.
            scale : np.ndarray
                Two-element array [xscale, yscale] for scaling the image along each axis.
            offset : np.ndarray
                Two-element array [xoffset, yoffset] specifying the image translation.
            levels : np.ndarray
                Two-element array [min, max] specifying the color scaling limits.
            cmap : str
                Name of the colormap to use.
            xrange : np.ndarray
                Two-element array [xmin, xmax] for x-axis limits.
            xaxis : str
                Label for the x-axis.
            title : str
                Title or label for the colorbar.
    """
    if not data:
        print('data for this plot not available')
        return

    shank_items.img_plots = utils.remove_items(shank_items.fig_img, shank_items.img_plots)
    shank_items.img_cbars = utils.remove_items(shank_items.fig_img_cb, shank_items.img_cbars)
    # TODO check if I need this
    # qt_utils.set_axis(self.fig_img_cb, 'top', pen='w')

    # Create image item and add to figure
    image = pg.ImageItem()
    image.setImage(data['img'])
    transform = [data['scale'][0], 0., 0., 0., data['scale'][1], 0., data['offset'][0], data['offset'][1], 1.]
    image.setTransform(QtGui.QTransform(*transform))
    shank_items.fig_img.addItem(image)
    shank_items.img_plots.append(image)

    # If applicable create colorbar and add to figure
    cmap = data.get('cmap', [])
    if cmap:
        color_bar = cb.ColorBar(data['cmap'])
        lut = color_bar.getColourMap()
        image.setLookupTable(lut)
        image.setLevels((data['levels'][0], data['levels'][1]))
        cbar = color_bar.makeColourBar(20, 5, shank_items.fig_img_cb, min=data['levels'][0],
                                       max=data['levels'][1], label=data['title'])
        shank_items.fig_img_cb.addItem(cbar)
        shank_items.img_cbars.append(cbar)
    else:
        image.setLevels((1, 0))

    # Set axis ranges
    set_xaxis_range(shank_items.fig_img, data)
    set_yaxis_range(shank_items.fig_img, shank_items)

    # Store plot references
    shank_items.ephys_plot = image


def plot_scatter(shank_items, data: Optional[Dict]) -> None:
    """
    Plot a 2D scatter plot of electrophysiological data on the figure.

    Parameters
    ----------
    data : dict, optional
        Dictionary containing the scatter plot data and display parameters. Expected keys:
            - 'x' : np.ndarray of float
                X-coordinates of the data points.
            - 'y' : np.ndarray of float
                Y-coordinates of the data points.
            - 'size' : np.ndarray of float
                Size of each point.
            - 'symbol' : np.ndarray of str
                Symbol type (e.g., 'o', 't', etc.) for each point.
            - 'colours' : np.ndarray
                Either array of floats for colormap or array of QColor objects.
            - 'pen' : str or QPen
                Pen used to outline each point.
            - 'levels' : tuple[float, float]
                Min and max levels used for the colormap.
            - 'title' : str
                Title for the colorbar.
            - 'xrange' : tuple[float, float]
                Display range for the x-axis.
            - 'xaxis' : str
                Label for the x-axis.
            - 'cmap' : str
                Name of the colormap to use.
            - 'cluster' : bool
                If True, enables interactive cluster selection on click.
    """

    if not data:
        print('data for this plot not available')
        return

    # Clear existing plot
    shank_items.img_plots = utils.remove_items(shank_items.fig_img, shank_items.img_plots)
    shank_items.img_cbars = utils.remove_items(shank_items.fig_img_cb, shank_items.img_cbars)
    # TODO check if I need this
    # qt_utils.set_axis(self.fig_img_cb, 'top', pen='w')

    size = data['size'].tolist()
    symbol = data['symbol'].tolist()

    # Create colorbar and add to figure
    color_bar = cb.ColorBar(data['cmap'])
    cbar = color_bar.makeColourBar(20, 5, shank_items.fig_img_cb, min=np.min(data['levels'][0]),
                                   max=np.max(data['levels'][1]), label=data['title'])
    shank_items.fig_img_cb.addItem(cbar)
    shank_items.img_cbars.append(cbar)

    # Determine brush type based on given data
    if isinstance(np.any(data['colours']), QtGui.QColor):
        brush = data['colours'].tolist()
    else:
        brush = color_bar.getBrush(data['colours'], levels=list(data['levels']))

    # Create scatter plot and add to figure
    plot = pg.ScatterPlotItem()
    plot.setData(x=data['x'], y=data['y'], symbol=symbol, size=size, brush=brush, pen=data['pen'])
    shank_items.fig_img.addItem(plot)
    shank_items.img_plots.append(plot)

    # Set axis ranges
    set_xaxis_range(shank_items.fig_img, data)
    set_yaxis_range(shank_items.fig_img, shank_items)

    # Store plot references
    shank_items.ephys_plot = plot



    # def plot_histology_nearby(self, fig, ax='right', movable=False):
    #     # TODO
    #
    #     """
    #     Plots histology figure - brain regions that intersect with probe track
    #     :param fig: figure on which to plot
    #     :type fig: pyqtgraph PlotWidget
    #     :param ax: orientation of axis, must be one of 'left' (fig_hist) or 'right' (fig_hist_ref)
    #     :type ax: string
    #     :param movable: whether probe reference lines can be moved, True for fig_hist, False for
    #                     fig_hist_ref
    #     :type movable: Bool
    #     """
    #
    #     # If no histology we can't plot histology
    #     if not self.histology_exists:
    #         return
    #
    #     fig.clear()
    #     self.hist_ref_regions = np.empty((0, 1))
    #     axis = fig.getAxis(ax)
    #     axis.setTicks([self.hist_data_ref['axis_label']])
    #     axis.setZValue(10)
    #
    #     utils.set_axis(fig, 'bottom', label='dist to boundary (um)')
    #     fig.setXRange(min=0, max=100)
    #     fig.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top + self.probe_extra,
    #                   padding=self.pad)
    #
    #     # Plot nearby regions
    #     for ir, (x, y, c) in enumerate(zip(self.hist_nearby_x, self.hist_nearby_y,
    #                                        self.hist_nearby_col)):
    #         colour = QtGui.QColor(c)
    #         plot = pg.PlotCurveItem()
    #         plot.setData(x=x, y=y * 1e6, fillLevel=10, fillOutline=True)
    #         plot.setBrush(colour)
    #         plot.setPen(colour)
    #         fig.addItem(plot)
    #
    #     for ir, (x, y, c) in enumerate(zip(self.hist_nearby_parent_x, self.hist_nearby_parent_y,
    #                                        self.hist_nearby_parent_col)):
    #         colour = QtGui.QColor(c)
    #         colour.setAlpha(70)
    #         plot = pg.PlotCurveItem()
    #         plot.setData(x=x, y=y * 1e6, fillLevel=10, fillOutline=True)
    #         plot.setBrush(colour)
    #         plot.setPen(colour)
    #         fig.addItem(plot)
    #
    #     # Add dotted lines to plot to indicate region along probe track where electrode
    #     # channels are distributed
    #     self.tip_pos = pg.InfiniteLine(pos=self.probe_tip, angle=0, pen=self.kpen_dot,
    #                                    movable=movable)
    #     self.top_pos = pg.InfiniteLine(pos=self.probe_top, angle=0, pen=self.kpen_dot,
    #                                    movable=movable)
    #     # Add lines to figure
    #     fig.addItem(self.tip_pos)
    #     fig.addItem(self.top_pos)