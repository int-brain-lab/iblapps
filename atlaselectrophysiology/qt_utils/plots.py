import pyqtgraph as pg
import numpy as np
from typing import Dict, Optional, Any
from qtpy import QtGui
import atlaselectrophysiology.qt_utils.utils as utils
import atlaselectrophysiology.qt_utils.ColorBar as cb
from dataclasses import asdict, is_dataclass
from atlaselectrophysiology.loaders.plot_loader import ScatterData, ImageData, LineData, ProbeData


def set_xaxis_range(fig, data, label=True):
    if is_dataclass(data):
        data = asdict(data)
    fig.setXRange(min=data['xrange'][0], max=data['xrange'][1], padding=0)
    if label:
        utils.set_axis(fig, 'bottom', label=data['xaxis'])


def set_yaxis_range(fig, data):
    if is_dataclass(data):
        data = asdict(data)
    fig.setYRange(min=data['yrange'][0], max=data['yrange'][1], padding=data['pad'])


def plot_fit(fit_plot, fit_plot_lin, fit_scatter, data) -> None:
    """
    Plot the scale factor and offset applied to channels along the depth of the probe track,
    relative to the original positions of the channels.
    """

    if len(data['x']) > 2:
        fit_plot.setData(x=data['x'], y=data['y'])
        fit_scatter.setData(x=data['x'],y=data['y'])

        if np.any(data['depth_lin']):
            fit_plot_lin.setData(x=data['depth'], y=data['depth_lin'])
        else:
            fit_plot_lin.setData()
    else:
        fit_plot.setData()
        fit_scatter.setData()
        fit_plot_lin.setData()



def plot_histology(fig: pg.PlotWidget, data: Dict[str, Any], ax: str ='left') -> None:
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

    hist_regions = np.empty((0, 1), dtype=object)

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
        hist_regions = np.vstack([hist_regions, region])

    # Add additional boundary for final region
    bound = pg.InfiniteLine(pos=data['region'][-1][1], angle=0, pen='w')
    fig.addItem(bound)

    # Add dotted probe track limits (tip and top) as horizontal reference lines

    set_yaxis_range(fig, data)

    return hist_regions




def plot_scale_factor(fig_scale, fig_scale_cb, data) -> None:
    """
    Plot the scale factor applied to brain regions along the probe track, displayed alongside
    the histology figure. This visualizes regional scale adjustments as colored horizontal bands.
    """

    scale_regions = np.empty((0, 1))

    cbar = cb.ColorBar('seismic', plot_item=fig_scale_cb)
    colours = cbar.map.mapToQColor(data['scale_factor'])
    cbar.setLevels((0, 1.5), label='Scale Factor')

    # color_bar = cb.ColorBar('seismic')
    # cbar = color_bar.makeColourBar(20, 5, fig_scale_cb, min=0.5, max=1.5, label='Scale Factor')
    # colours = color_bar.map.mapToQColor(data['scale_factor'])
    # fig_scale_cb.addItem(cbar)

    for ir, reg in enumerate(data['region']):
        region = pg.LinearRegionItem(values=(reg[0], reg[1]),
                                     orientation=pg.LinearRegionItem.Horizontal,
                                     brush=colours[ir], movable=False)
        bound = pg.InfiniteLine(pos=reg[0], angle=0, pen=colours[ir])

        fig_scale.addItem(region)
        fig_scale.addItem(bound)
        scale_regions = np.vstack([scale_regions, region])

    bound = pg.InfiniteLine(pos=data['region'][-1][1], angle=0, pen=colours[-1])
    fig_scale.addItem(bound)

    set_yaxis_range(fig_scale, data)
    utils.set_axis(fig_scale, 'bottom', pen='w', label='blank')

    return scale_regions


def plot_slice(fig_slice, data: Dict, data_chns):
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

    fig_slice.addItem(img)
    fig_slice.autoRange()

    # Plot the trajectory line
    traj_line = pg.PlotCurveItem()
    traj_line.setData(x=data_chns['x'], y=data_chns['y'], pen=utils.kpen_solid)
    fig_slice.addItem(traj_line)

    return img, color_bar, traj_line




def plot_channels(fig_slice, data, colour='r') -> None:
    """
    Plot electrode channels and alignment lines on the histological slice.

    Displays the locations of electrode channels and probe track reference lines
    (alignment lines) on the histology slice view.
    """

    # If no histology we can't do alignment

    # Clear existing reference lines
    slice_chns = pg.ScatterPlotItem()
    slice_chns.setData(x=data['xyz_channels'][:, 0], y=data['xyz_channels'][:, 2], pen=colour, brush=colour)
    fig_slice.addItem(slice_chns)

    slice_lines = []
    for ref_line in data['track_lines']:
        line = pg.PlotCurveItem()
        line.setData(x=ref_line[:, 0], y=ref_line[:, 2], pen=utils.kpen_dot)
        fig_slice.addItem(line)
        slice_lines.append(line)

    return slice_lines, slice_chns



def plot_line(fig_line, data: Optional[Dict]) -> None:
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
    if isinstance(data, LineData):
        data = asdict(data)
    # Create line plot at add to figure
    line = pg.PlotCurveItem()
    line.setData(x=data['x'], y=data['y'])
    line.setPen(utils.kpen_solid)
    fig_line.addItem(line)

    # Set colour of xaxis
    utils.set_axis(fig_line, 'bottom', pen='k')

    # Set axis ranges
    set_xaxis_range(fig_line, data)
    set_yaxis_range(fig_line, data)

    return line


def plot_probe(fig_probe, fig_probe_cb, data: Optional[Dict], levels=None) -> None:
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
    if isinstance(data, ProbeData):
        data = asdict(data)
    # Create colorbar and add to figure
    # color_bar = cb.ColorBar(data['cmap'])
    # lut = color_bar.getColourMap()
    # cbar = color_bar.makeColourBar(20, 5, fig_probe_cb, min=data['levels'][0],
    #                                max=data['levels'][1], label=data['title'], lim=False)
    # fig_probe_cb.addItem(cbar)

    if levels is None:
        levels = data['levels']
    else:
        levels = levels

    cbar = cb.ColorBar(data['cmap'], plot_item=fig_probe_cb)
    cbar.setLevels(levels, label=data['title'])
    lut = cbar.getColourMap()

    # Create image plots per shank and add to figure
    image_items = []
    for img, scale, offset in zip(data['img'], data['scale'], data['offset']):
        image = pg.ImageItem()
        image.setImage(img)
        transform = [scale[0], 0., 0., 0., scale[1], 0., offset[0], offset[1], 1.]
        image.setTransform(QtGui.QTransform(*transform))
        image.setLookupTable(lut)
        image.setLevels((levels[0], levels[1]))
        fig_probe.addItem(image)
        image_items.append(image)

    # Set axis ranges
    set_xaxis_range(fig_probe, data, label=False)
    set_yaxis_range(fig_probe, data)
    # Add in a fake label so that the appearence is the same as other plots
    utils.set_axis(fig_probe, 'bottom', pen='w', label='blank')

    # Optionally plot horizontal boundary lines
    bound_items = []
    bounds = data.get('boundaries', None)
    if bounds is not None:
        for bound in bounds:
            line = pg.InfiniteLine(pos=bound, angle=0, pen='w')
            fig_probe.addItem(line)
            bound_items.append(line)

    return image_items, cbar, bound_items



def plot_scatter(fig_img, fig_img_cb,  data: Optional[Dict], levels=None) -> None:
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
    if isinstance(data, ScatterData):
        data = asdict(data)
    # Clear existing plot
    # TODO check if I need this
    # qt_utils.set_axis(self.fig_img_cb, 'top', pen='w')

    size = data['size'].tolist()
    symbol = data['symbol'].tolist()

    # Create colorbar and add to figure
    # color_bar = cb.ColorBar(data['cmap'])
    # cbar = color_bar.makeColourBar(20, 5, fig_img_cb, min=np.min(data['levels'][0]),
    #                                max=np.max(data['levels'][1]), label=data['title'])
    # fig_img_cb.addItem(cbar)

    if levels is None:
        levels = data['levels']
    else:
        levels = levels

    cbar = cb.ColorBar( data['cmap'], plot_item=fig_img_cb)
    cbar.setLevels(levels, label=data['title'])


    # Determine brush type based on given data
    if isinstance(np.any(data['colours']), QtGui.QColor):
        brush = data['colours'].tolist()
    else:
        #brush = color_bar.getBrush(data['colours'], levels=list(data['levels']))
        brush = cbar.getBrush(data['colours'], levels=list(levels))

    # Create scatter plot and add to figure
    plot = pg.ScatterPlotItem()
    plot.setData(x=data['x'], y=data['y'], symbol=symbol, size=size, brush=brush, pen=data['pen'])
    fig_img.addItem(plot)

    # Set colour of xaxis
    utils.set_axis(fig_img, 'bottom', pen='k')

    # Set axis ranges
    set_xaxis_range(fig_img, data)
    set_yaxis_range(fig_img, data)

    return plot, cbar


def plot_image(fig_img, fig_img_cb, data: Optional[Dict], levels=None) -> None:
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

    if isinstance(data, ImageData):
        data = asdict(data)

    # TODO check if I need this
    # qt_utils.set_axis(self.fig_img_cb, 'top', pen='w')
    if levels is None:
        levels = data['levels']
    else:
        levels = levels

    # Create image item and add to figure
    image = pg.ImageItem()
    image.setImage(data['img'])
    transform = [data['scale'][0], 0., 0., 0., data['scale'][1], 0., data['offset'][0], data['offset'][1], 1.]
    image.setTransform(QtGui.QTransform(*transform))
    fig_img.addItem(image)

    # If applicable create colorbar and add to figure
    cmap = data.get('cmap', [])
    if cmap:
        # color_bar = cb.ColorBar(data['cmap'])
        # lut = color_bar.getColourMap()
        # image.setLookupTable(lut)
        # image.setLevels((data['levels'][0], data['levels'][1]))
        # cbar = color_bar.makeColourBar(20, 5, fig_img_cb, min=data['levels'][0],
        #                                max=data['levels'][1], label=data['title'])
        # fig_img_cb.addItem(cbar)

        cbar = cb.ColorBar( data['cmap'], plot_item=fig_img_cb)
        lut = cbar.getColourMap()
        image.setLookupTable(lut)
        image.setLevels((levels[0], levels[1]))
        cbar.setLevels(levels, label=data['title'])

    else:
        image.setLevels((1, 0))

    # Set colour of xaxis
    utils.set_axis(fig_img, 'bottom', pen='k')

    set_xaxis_range(fig_img, data)
    set_yaxis_range(fig_img, data)

    return image, cbar
