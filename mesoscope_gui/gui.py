# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import glob
import json
from math import pow, sqrt, exp
import os
import os.path as op
from pathlib import Path
import sys
from typing import List

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QListWidget, QLabel,
    QScrollBar, QPushButton, QWidget, QMenu, QAction, QSplitter, QListView, QAbstractItemView,
    QLineEdit, QTreeView, QGraphicsOpacityEffect, QGraphicsBlurEffect, QSpinBox, QComboBox)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QKeySequence, QPen
from superqt import QRangeSlider

import imageio.v3 as iio
import numpy as np


# -------------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------------

WIDTH = 1024
HEIGHT = 768
DEFAULT_LEFT_PANEL_WIDTH = 250
DEFAULT_POINT_COUNT = 3
MAX_LEFT_PANEL_WIDTH = 500
DEFAULT_GLOB = "*stack*.tif"
RADIUS = 48
COLORS = (
    (224, 54, 0),
    (91, 200, 0),
    (4, 100, 141),
)
WHITE = (255, 255, 255)
STROKE_WIDTH = 8
BORDER_WIDTH = 4
MAX_SIZE = RADIUS - STROKE_WIDTH - 2 * BORDER_WIDTH
MIN_DISTANCE = .1
SNAPSHOT_RESCALING_FACTOR = .075


# -------------------------------------------------------------------------------------------------
# Util functions
# -------------------------------------------------------------------------------------------------

def set_widget_opaque(widget, is_opaque):
    if is_opaque:
        widget.setGraphicsEffect(None)
    else:
        opacity_effect = QGraphicsOpacityEffect()
        opacity_effect.setOpacity(0.5)
        widget.setGraphicsEffect(opacity_effect if not is_opaque else None)


def set_blur(widget, blur_amount):
    if blur_amount == 0:
        widget.setGraphicsEffect(None)
    else:
        blur_effect = QGraphicsBlurEffect()
        blur_effect.setBlurRadius(blur_amount)
        widget.setGraphicsEffect(blur_effect)


def rescale(img, scale=1):
    new_h = int(img.shape[0] * scale)
    new_w = int(img.shape[1] * scale)
    row_indices = (np.linspace(0, img.shape[0] - 1, new_h)).astype(int)
    col_indices = (np.linspace(0, img.shape[1] - 1, new_w)).astype(int)
    return img[np.ix_(row_indices, col_indices)]


def inset(img, inset, loc='tl'):
    if inset is None:
        return img
    assert img.ndim == 2
    new_h, new_w = inset.shape[:2]
    if loc == 'tl':
        img[:new_h, :new_w] = inset
    if loc == 'tr':
        img[:new_h, -new_w:] = inset
    return img


def normalize_image(img, v0, v1):
    h = .001
    vmin, vmax = np.quantile(img, [h, 1-h])
    d = vmax - vmin

    v0 /= 99.0
    v1 /= 99.0
    vmin += d * v0
    vmax -= d * (1.0 - v1)

    img = (img - vmin) / (vmax - vmin)
    img = np.clip(img, 0, 1)
    img = np.floor(img * 255).astype(np.uint8)

    return img


class PointWidget:
    def __init__(self, color, opacity=255, parent=None):
        self.color = color
        self.opacity = opacity
        self.size = MAX_SIZE

        self.widget = None
        if parent is not None:
            self.set_widget(parent)

    def set_size(self, size):
        self.size = min(MAX_SIZE, max(0, size))
        self.update()

    def set_opacity(self, opacity):
        self.opacity = opacity
        self.update()

    def pixmap(self):
        s = RADIUS
        offsets = (0, int(1.5*BORDER_WIDTH))
        colors = (WHITE, self.color)
        opacities = (255, self.opacity)
        widths = (STROKE_WIDTH + BORDER_WIDTH, STROKE_WIDTH)

        pixmap = QPixmap(RADIUS, RADIUS)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setBrush(Qt.NoBrush)

        for color, opacity, width, b in zip(colors, opacities, widths, offsets):
            painter.setPen(QPen(QColor(*color, opacity), width))
            painter.drawLine(b, s // 2, s-b, s // 2)
            painter.drawLine(s // 2, b, s // 2, s-b)

        painter.end()
        return pixmap

    def set_widget(self, parent):
        r = RADIUS
        self.widget = QLabel(parent)
        self.widget.setFixedSize(r, r)
        self.widget.setPixmap(self.pixmap())
        self.widget.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.clear()
        self.widget.show()

    def clear(self):
        self.widget.move(-100, -100)

    def move(self, x, y):
        r = RADIUS
        self.widget.move(x - r // 2, y - r // 2)

    def update(self):
        self.widget.setPixmap(self.pixmap())


# -------------------------------------------------------------------------------------------------
# Image loader
# -------------------------------------------------------------------------------------------------

class Loader:
    def __init__(self):
        self._image_list = []
        self._current_index = -1
        self.pattern = ''
        self.root_dir = None

    def set_root_dir(self, root_dir: Path):
        assert root_dir
        self.root_dir = root_dir
        # self.update()

    def set_glob(self, pattern: str):
        self.pattern = pattern
        # self.update()

    def update(self):
        pattern = '**/' + self.pattern
        self._image_list = glob.glob(pattern, root_dir=self.root_dir, recursive=True)
        self._image_list = [Path(_) for _ in self._image_list]
        self._current_index = 0 if self._image_list else -1

    @property
    def image_list(self) -> List[Path]:
        return self._image_list

    @property
    def current_image(self) -> Path:
        if 0 <= self._current_index < len(self._image_list):
            return self._image_list[self._current_index]

    @property
    def points_file(self) -> Path:
        if self.current_image:
            name = self.current_image.stem.split('.')[0]
            return self.current_image.with_name(name + ".points.json")

    @property
    def current_index(self):
        return self._current_index

    def set_index(self, index):
        self._current_index = index

    def can_next(self):
        return self._current_index < len(self._image_list) - 1

    def can_prev(self):
        return self._current_index > 0

    def next(self):
        if self.can_next():
            self._current_index += 1

    def prev(self):
        if self.can_prev():
            self._current_index -= 1

    @property
    def snapshot_before(self):
        path = (self.current_image / '../../../snapshots/WF_before.png').resolve()
        if path:
            return path

    @property
    def snapshot_after(self):
        path = (self.current_image / '../../../snapshots/WF_after.png').resolve()
        if path:
            return path


# -------------------------------------------------------------------------------------------------
# Mescoscope GUI
# -------------------------------------------------------------------------------------------------

class MesoscopeGUI(QMainWindow):

    # Initialization
    # ---------------------------------------------------------------------------------------------

    def __init__(self):
        self.loader = Loader()

        self.widget_range_slider = None

        # NOTE: we keep these in memory to avoid reloading them whenever we load a new slice
        self.WF_before = None
        self.WF_after = None

        self.image_stack = None
        self.pixmap = None
        self.clear_points_struct()

        super().__init__()
        self.init_ui()

    def make_point_widget(self, idx):
        color = COLORS[idx]
        pw = PointWidget(color, parent=self.widget_image)
        w = pw.widget
        w.mousePressEvent = lambda event: self.start_drag(event, w)
        w.mouseMoveEvent = lambda event: self.drag_point(event, w)
        w.mouseReleaseEvent = lambda event: self.end_drag(event, w, idx)
        return pw

    def create_points_widgets(self, n=DEFAULT_POINT_COUNT):
        self.points_widgets = [self.make_point_widget(idx) for idx in range(n)]

    def clear_points_struct(self, n=DEFAULT_POINT_COUNT):
        self.points = [{} for idx in range(n)]  # point_idx, stack_idx, coords

    def update_points(self):
        [p.update() for p in self.points_widgets]
        [self.update_point_filter(idx) for idx in range(self.point_count)]

    # UI
    # ---------------------------------------------------------------------------------------------

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        row1 = self.create_row1()
        main_layout.addLayout(row1)

        row2 = self.create_row2()
        main_layout.addLayout(row2)

        row3 = self.create_row3()
        main_layout.addLayout(row3)

        self.create_menu()
        self.create_bindings()
        self.create_points_widgets()

        self.setWindowTitle('Mesoscope GUI')
        self.resize(WIDTH, HEIGHT)
        self.show()

    def create_menu(self):
        # Menu.
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')

        # Open root directory.
        open_action = QAction('Open', self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_dialog)
        file_menu.addAction(open_action)

        # Open JSON points.
        open_json_action = QAction('Open JSON points', self)
        open_json_action.triggered.connect(self.open_json_dialog)
        file_menu.addAction(open_json_action)

        # Quit.
        quit_action = QAction('Quit', self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def create_row1(self):
        layout = QHBoxLayout()
        layout.setSpacing(10)

        # Glob pattern.
        self.widget_glob = QLineEdit()
        self.widget_glob.setText(DEFAULT_GLOB)
        self.widget_glob.setFixedWidth(250)
        layout.addWidget(self.widget_glob)

        # Dropdown with list of images.
        self.widget_dropdown = QComboBox()
        layout.addWidget(self.widget_dropdown)

        # Previous button.
        self.widget_prev_button = QPushButton("<")
        self.widget_prev_button.setFixedWidth(40)
        self.widget_prev_button.setEnabled(False)
        layout.addWidget(self.widget_prev_button)

        # Next button.
        self.widget_next_button = QPushButton(">")
        self.widget_next_button.setFixedWidth(40)
        self.widget_next_button.setEnabled(False)
        layout.addWidget(self.widget_next_button)

        layout.setContentsMargins(10, 0, 10, 0)
        return layout

    def create_row2(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(10)

        range_slider_layout = QHBoxLayout()

        # Range slider.
        self.widget_range_slider = QRangeSlider(Qt.Horizontal)
        self.widget_range_slider.setMaximumHeight(20)
        self.widget_range_slider.setValue((0, 100))
        range_slider_layout.addWidget(self.widget_range_slider)

        image_layout = QHBoxLayout()

        # Slice widget.
        self.widget_slice_label = QLabel("Slice #0")
        self.widget_slice_label.setAlignment(Qt.AlignRight)
        self.widget_slice_label.setMaximumHeight(20)
        range_slider_layout.addWidget(self.widget_slice_label)

        layout.addLayout(range_slider_layout)

        # Image.
        self.widget_image = QLabel()
        self.widget_image.setAlignment(Qt.AlignCenter)
        self.widget_image.installEventFilter(self) # for mouse wheel scroll
        self.widget_image.setScaledContents(True)
        self.widget_image.resizeEvent = self.on_resized
        self.widget_image.mousePressEvent = self.add_point_at_click
        self.widget_image.setMinimumSize(1, 1)
        image_layout.addWidget(self.widget_image)

        # Scrollbar.
        self.widget_scrollbar = QScrollBar(Qt.Vertical)
        # self.widget_scrollbar.setFixedWidth(20)
        image_layout.addWidget(self.widget_scrollbar)

        layout.addLayout(image_layout)
        return layout

    def create_row3(self):
        layout = QHBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 0, 10, 0)

        # Cortical depth.
        self.widget_cortical_depth = QSpinBox()
        self.widget_cortical_depth.setRange(0, 1000)
        self.widget_cortical_depth.setSingleStep(10)
        self.widget_cortical_depth.setValue(0)
        self.widget_cortical_depth.setPrefix("cortical depth: ")
        self.widget_cortical_depth.setSuffix(" microns")
        layout.addWidget(self.widget_cortical_depth)

        # Move down.
        self.widget_move_down_button = QPushButton("Move down")
        layout.addWidget(self.widget_move_down_button)

        # Move to current depth.
        self.widget_move_to_depth_button = QPushButton("Move to current depth")
        layout.addWidget(self.widget_move_to_depth_button)

        # Move up.
        self.widget_move_up_button = QPushButton("Move up")
        layout.addWidget(self.widget_move_up_button)

        layout.addStretch()
        return layout

    def create_bindings(self):
        self.widget_glob.returnPressed.connect(self.on_glob)
        self.widget_dropdown.currentTextChanged.connect(self.on_select)
        self.widget_cortical_depth.valueChanged.connect(self.on_cortical_depth)
        self.widget_range_slider.valueChanged.connect(self.on_slider)
        self.widget_scrollbar.valueChanged.connect(self.on_scrollbar)
        self.widget_move_down_button.clicked.connect(self.on_move_down)
        self.widget_move_to_depth_button.clicked.connect(self.on_move_to_current_depth)
        self.widget_move_up_button.clicked.connect(self.on_move_up)
        self.widget_prev_button.clicked.connect(self.on_prev)
        self.widget_next_button.clicked.connect(self.on_next)

    # Event handlers
    # ---------------------------------------------------------------------------------------------

    def on_glob(self):
        self.loader.set_glob(self.widget_glob.text())
        self.update_files()

    def on_select(self):
        self.loader.set_index(self.widget_dropdown.currentIndex())
        self.on_loader()

    def on_loader(self):
        self.widget_prev_button.setEnabled(self.loader.can_prev())
        self.widget_next_button.setEnabled(self.loader.can_next())

        # Load the range values, which should come BEFORE image loading
        self.load_points()

        # Load the image.
        if self.loader.current_image:
            self.WF_before = self.load_snapshot(self.loader.snapshot_before)
            self.WF_after = self.load_snapshot(self.loader.snapshot_after)
            self.image_stack = self.load_image_stack(self.loader.current_image)

    def on_cortical_depth(self):
        self.save_cortical_depth()

    def on_slider(self):
        if self.image_stack is None:
            return
        stack_idx = self.get_stack()
        img = self.image_stack[stack_idx, ...]
        self.set_image(img)
        self.save_range()

    def on_scrollbar(self):
        if self.image_stack is None:
            return
        self.set_stack()
        stack_idx = self.get_stack()
        img = self.image_stack[stack_idx, ...]
        self.set_image(img)

    def on_move_down(self):
        pass

    def on_move_to_current_depth(self):
        pass

    def on_move_up(self):
        pass

    def on_prev(self):
        self.loader.prev()
        self.widget_dropdown.setCurrentIndex(self.loader.current_index)
        self.on_loader()

    def on_next(self):
        self.loader.next()
        self.widget_dropdown.setCurrentIndex(self.loader.current_index)
        self.on_loader()

    # File opening
    # ---------------------------------------------------------------------------------------------

    def open_dialog(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.DirectoryOnly)

        # dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        # file_view = dialog.findChild(QListView, 'listView')

        # # to make it possible to select multiple directories:
        # if file_view:
        #     file_view.setSelectionMode(QAbstractItemView.MultiSelection)
        # f_tree_view = dialog.findChild(QTreeView)
        # if f_tree_view:
        #     f_tree_view.setSelectionMode(QAbstractItemView.MultiSelection)

        if dialog.exec():
            paths = dialog.selectedFiles()
            paths = sorted(paths)
            self.open(paths)

    def open_json_dialog(self):
        json_file, _ = QFileDialog.getOpenFileName(
            self,
            "Open JSON Points File",
            "",
            "JSON Files (*.json)"
        )
        if json_file:
            self.load_points(json_file)

    def open(self, paths):
        if not paths:
            return
        # NOTE: only take the first path.
        path = paths[0]
        path = Path(path).resolve()

        self.loader.set_glob(self.widget_glob.text())
        self.loader.set_root_dir(path)
        self.update_files()

    def update_files(self):
        self.loader.update()
        self.widget_dropdown.clear()
        self.widget_dropdown.addItems([str(_) for _ in self.loader.image_list])

    def navigate(self, direction):
        if direction > 0:
            self.loader.next()
        elif direction < 0:
            self.loader.prev()
        self.widget_dropdown.setCurrentIndex(self.loader.current_index)

    # Image setting
    # ---------------------------------------------------------------------------------------------

    def set_image(self, img):
        assert img.ndim == 2
        v0, v1 = self.get_range()

        # Slice image.
        img = normalize_image(img, v0, v1)

        # Add the reference images.
        img = inset(img, self.WF_before, loc='tl')
        img = inset(img, self.WF_after, loc='tr')

        qimg = QImage(
            img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
        self.pixmap = QPixmap.fromImage(qimg)
        self.widget_image.setPixmap(self.pixmap)

    # Image stack
    # ---------------------------------------------------------------------------------------------

    @property
    def stack_count(self):
        return self.image_stack.shape[0] if self.image_stack is not None else 0

    def load_image_stack(self, img_path):
        img = iio.imread(img_path).astype(np.float32)  # shape: nstacks, h, w
        assert img.ndim == 3
        stack_count = img.shape[0]

        # Number of images in the stack.
        self.widget_scrollbar.setMaximum(stack_count - 1)

        # Default stack: middle.
        mid_stack = stack_count // 2

        # Set the default stack.
        stack_idx = mid_stack if not self.points else self.points[0].get('stack_idx', mid_stack)
        self.set_stack(stack_idx)

        # Set the image.
        self.set_image(img[stack_idx, ...])

        # Updates.
        self.update_margins()
        self.update_points()

        return img

    def load_snapshot(self, path):
        if path.exists():
            img = iio.imread(path)
            img = np.rot90(img)
            img = img.mean(axis=-1)
            img = rescale(img, SNAPSHOT_RESCALING_FACTOR)
            return img

    # Scrollbar
    # ---------------------------------------------------------------------------------------------

    def get_stack(self):
        return self.widget_scrollbar.value()

    def set_stack(self, idx=None):
        if idx is not None:
            idx = max(self.widget_scrollbar.minimum(), min(self.widget_scrollbar.maximum(), idx))
            self.widget_scrollbar.setValue(idx)

        idx = self.widget_scrollbar.value()
        s = f"Slice #{idx + 1:03d} / {self.stack_count}"
        self.widget_slice_label.setText(s)

    # Value range
    # ---------------------------------------------------------------------------------------------

    def get_range(self):
        if self.widget_range_slider is None:
            return None, None
        vmin, vmax = self.widget_range_slider.value()
        return vmin, vmax

    # Cortical depth
    # ---------------------------------------------------------------------------------------------

    def get_cortical_depth(self):
        return self.widget_cortical_depth.value()

    def set_cortical_depth(self, value):
        self.widget_cortical_depth.setValue(value)

    # Coordinate transforms
    # ---------------------------------------------------------------------------------------------

    def to_relative(self, x, y):
        label_width, label_height = self.widget_image.width(), self.widget_image.height()
        pixmap = self.widget_image.pixmap()

        if not pixmap:
            return None, None

        # Get the aspect ratio for the pixmap and label
        pixmap_width, pixmap_height = pixmap.width(), pixmap.height()
        label_aspect_ratio = label_width / label_height
        pixmap_aspect_ratio = pixmap_width / pixmap_height

        # Calculate scaled dimensions of the pixmap to fit within the label, preserving aspect ratio
        if label_aspect_ratio > pixmap_aspect_ratio:
            scaled_width = int(label_height * pixmap_aspect_ratio)
            scaled_height = label_height
        else:
            scaled_width = label_width
            scaled_height = int(label_width / pixmap_aspect_ratio)

        # Calculate margins on each side
        x_margin = (label_width - scaled_width) // 2
        y_margin = (label_height - scaled_height) // 2

        # Convert from pixel coordinates to relative [0,1] coordinates
        xr = (x - x_margin) / scaled_width
        yr = (y - y_margin) / scaled_height

        return xr, yr

    def to_absolute(self, xr, yr):
        label_width, label_height = self.widget_image.width(), self.widget_image.height()
        pixmap = self.widget_image.pixmap()

        if not pixmap:
            return None, None

        # Get the aspect ratio for the pixmap and label
        pixmap_width, pixmap_height = pixmap.width(), pixmap.height()
        label_aspect_ratio = label_width / label_height
        pixmap_aspect_ratio = pixmap_width / pixmap_height

        # Calculate scaled dimensions of the pixmap to fit within the label, preserving aspect ratio
        if label_aspect_ratio > pixmap_aspect_ratio:
            scaled_width = int(label_height * pixmap_aspect_ratio)
            scaled_height = label_height
        else:
            scaled_width = label_width
            scaled_height = int(label_width / pixmap_aspect_ratio)

        # Calculate margins on each side
        x_margin = (label_width - scaled_width) // 2
        y_margin = (label_height - scaled_height) // 2

        # Convert from relative [0,1] coordinates back to pixel coordinates
        x = x_margin + int(xr * scaled_width)
        y = y_margin + int(yr * scaled_height)

        return x, y

    def update_margins(self):
        pixmap = self.pixmap
        if not pixmap:
            return
        size = self.widget_image.size()
        w, h = size.width(), size.height()
        pw = pixmap.width()
        ph = pixmap.height()

        if (w * ph > h * pw):
            m = (w - (pw * h / ph)) / 2
            m = int(m)
            self.widget_image.setContentsMargins(m, 0, m, 0)
        else:
            m = (h - (ph * w / pw)) / 2
            m = int(m)
            self.widget_image.setContentsMargins(0, m, 0, m)

    # Adding points
    # ---------------------------------------------------------------------------------------------

    def set_point_position(self, point_idx, xr, yr, stack_idx):
        if not self.points or point_idx >= self.point_count:
            return

        self.points[point_idx]['coords'] = (xr, yr)
        self.points[point_idx]['stack_idx'] = stack_idx

        self.update_point_position(point_idx)
        self.update_point_filter(point_idx)

    def update_point_position(self, idx):
        if not self.points or idx >= self.point_count:
            return
        xr, yr = self.points[idx].get('coords', (None, None))
        if xr is None:
            return
        x, y = self.to_absolute(xr, yr)
        if x is not None and y is not None:
            self.points_widgets[idx].move(x, y)

    def update_point_filter(self, idx):
        if not self.points or idx >= self.point_count:
            return
        w = self.points_widgets[idx].widget
        stack_idx = self.points[idx].get('stack_idx', -1)
        blur = abs(self.get_stack() - stack_idx)
        blur = 0 if blur == 0 else 2 + .5 * blur * blur
        set_blur(w, blur)

    def add_point_at_click(self, event):
        x, y = event.pos().x(), event.pos().y()
        point_idx = next((i for i, point in enumerate(self.points) if not point), None)
        if point_idx is None:
            return
        if 0 <= point_idx and point_idx < self.point_count:
            xr, yr = self.to_relative(x, y)
            self.set_point_position(point_idx, xr, yr, self.get_stack())
            self.save_points()

    @property
    def point_count(self):
        return len(self.points)

    # Stack navigation
    # ---------------------------------------------------------------------------------------------

    def move_points(self, absolute=None, relative=None):
        if not self.points:
            return

        # Move the current scrollbar.
        if relative is not None:
            self.set_stack(self.get_stack() + relative)
        elif absolute is not None:
            self.set_stack(absolute)

        # Move all points.
        for idx in range(self.point_count):
            if 'stack_idx' in self.points[idx]:
                if relative is not None:
                    self.points[idx]['stack_idx'] += relative
                elif absolute is not None:
                    self.points[idx]['stack_idx'] = absolute
                self.update_point_filter(idx)

        # Save the points.
        self.save_points()

    def move_down(self, ev):
        self.move_points(relative=-1)

    def move_to_current_depth(self, ev):
        self.move_points(absolute=self.get_stack())

    def move_up(self, ev):
        self.move_points(relative=-1)

    # Points drag and drop
    # ---------------------------------------------------------------------------------------------

    def _widget_idx(self, w):
        widgets = [pw.widget for pw in self.points_widgets]
        assert w in widgets
        point_idx = widgets.index(w)
        return point_idx

    def start_drag(self, event, w):
        self.drag_offset = event.pos()
        w.raise_()

        # Set the point's stack idx to the current stack
        idx = self._widget_idx(w)
        if 0 <= idx and idx < self.point_count:
            self.points[idx]['stack_idx'] = self.get_stack()
            self.update_point_filter(idx)

    def drag_point(self, event, w):
        if not self.points:
            return
        idx = self._widget_idx(w)
        new_pos = w.pos() + event.pos() - self.drag_offset
        x, y = self.to_relative(new_pos.x(), new_pos.y())
        for i in range(self.point_count):
            if i == idx:
                continue
            coords = self.points[i].get('coords', None)
            if coords is None:
                continue
            x_ = coords[0]
            y_ = coords[1]
            d = sqrt((x - x_) ** 2 + (y - y_) ** 2)
            if d <= MIN_DISTANCE:
                print(f"Warning: points {idx} and {i} are too close ({d: .3f} <= {MIN_DISTANCE})")
            #     x = x_ + MIN_DISTANCE * (x-x_) / d
            #     y = y_ + MIN_DISTANCE * (y-y_) / d
            #     x, y = self.to_absolute(x, y)
            #     new_pos = QPoint(x, y)
        w.move(new_pos)

    def end_drag(self, event, w, point_idx):
        if point_idx >= self.point_count:
            return
        r = RADIUS
        x, y = w.x() + r // 2, w.y() + r // 2
        xr, yr = self.to_relative(x, y)
        if xr < 0 or xr > 1 or yr < 0 or yr > 1:
            print(f"Deleting point {point_idx}")
            self.points[point_idx] = {}
        else:
            self.points[point_idx]['coords'] = xr, yr
        self.save_points()

    # I/O
    # ---------------------------------------------------------------------------------------------

    def load(self, points_file=None):
        points_file = points_file or self.loader.points_file
        if points_file and op.exists(points_file):
            with open(points_file, 'r') as f:
                return json.load(f)
        return {}

    def save_fields(self, points_file=None, **kwargs):
        points_file = points_file or self.loader.points_file
        if not points_file:
            return
        data = self.load(points_file=points_file)
        data.update(**kwargs)
        with open(points_file, 'w') as f:
            json.dump(data, f, indent=2)
            f.write('\n')

    # Points file
    # ---------------------------------------------------------------------------------------------

    def load_points(self, points_file=None):
        # Reset the points.
        self.clear_points_struct()
        for p in self.points_widgets:
            p.clear()

        # Load the points.
        data = self.load(points_file=points_file)
        self.points = data.get('points', [])

        # Range.
        vrange = data.get('range', (0, 99))
        self.widget_range_slider.setValue(tuple(vrange))

        # Cortical depth.
        self.set_cortical_depth(data.get('cortical_depth', 0))

        # Update the points position on the image.
        for point_idx in range(self.point_count):
            self.update_point_position(point_idx)
            self.update_point_filter(point_idx)

    def save_points(self):
        self.save_fields(points=self.points)

    def save_range(self):
        self.save_fields(range=self.get_range())

    def save_cortical_depth(self):
        self.save_fields(cortical_depth=self.get_cortical_depth())

    # Event handling
    # ---------------------------------------------------------------------------------------------

    def on_resized(self, ev):
        self.update_margins()
        for point_idx in range(self.point_count):
            self.update_point_position(point_idx)

    def eventFilter(self, obj, event):
        if obj == self.widget_image and event.type() == event.Wheel:
            delta = event.angleDelta().y() // 120
            new_value = self.widget_scrollbar.value() - delta
            new_value = max(0, min(self.widget_scrollbar.maximum(), new_value))
            self.widget_scrollbar.setValue(new_value)
            return True
        return super().eventFilter(obj, event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MesoscopeGUI()
    if sys.argv[1:]:
        gui.open(sys.argv[1:])
    sys.exit(app.exec_())
