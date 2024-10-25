# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import json
from math import pow, sqrt, exp
import os
import os.path as op
from pathlib import Path
import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QListWidget, QLabel,
    QScrollBar, QPushButton, QWidget, QMenu, QAction, QSplitter, QListView, QAbstractItemView,
    QTreeView, QGraphicsOpacityEffect, QGraphicsBlurEffect)
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
# Mescoscope GUI
# -------------------------------------------------------------------------------------------------

class MesoscopeGUI(QMainWindow):

    # Initialization
    # ---------------------------------------------------------------------------------------------

    def __init__(self):
        self.current_folder_idx = 0
        self.folder_paths = []

        self.stack_count = 0
        self.current_stack_idx = 0
        self.range_slider = None

        self.pixmap = None
        self.clear_points_struct()

        super().__init__()
        self.init_ui()
        self.init_points_widgets()

    def make_point_widget(self, idx):
        color = COLORS[idx]
        pw = PointWidget(color, parent=self.image_label)
        w = pw.widget
        w.mousePressEvent = lambda event: self.start_drag(event, w)
        w.mouseMoveEvent = lambda event: self.drag_point(event, w)
        w.mouseReleaseEvent = lambda event: self.end_drag(event, w, idx)
        return pw

    def init_points_widgets(self):
        self.points_widgets = [self.make_point_widget(idx) for idx in range(3)]

    def clear_points_struct(self):
        self.points = [{} for idx in range(3)]  # point_idx, stack_idx, coords

    def update_points(self):
        [p.update() for p in self.points_widgets]
        [self.update_point_filter(idx) for idx in range(3)]

    # UI
    # ---------------------------------------------------------------------------------------------

    def init_ui(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')

        open_action = QAction('Open', self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_dialog)
        file_menu.addAction(open_action)

        open_json_action = QAction('Open JSON points', self)
        open_json_action.triggered.connect(self.open_json_dialog)
        file_menu.addAction(open_json_action)

        quit_action = QAction('Quit', self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)

        self.splitter = QSplitter(Qt.Horizontal)
        self.layout.addWidget(self.splitter)

        # Folder list
        self.folder_list = QListWidget()
        self.folder_list.setMaximumWidth(200)
        self.folder_list.currentRowChanged.connect(self.update_folder)
        self.folder_list.itemClicked.connect(lambda: self.select_folder(self.folder_list.currentRow()))
        self.splitter.addWidget(self.folder_list)

        self.image_layout = QVBoxLayout()

        # Slice label.
        self.slice_index_label = QLabel("Slice #0")
        self.slice_index_label.setAlignment(Qt.AlignRight)
        self.slice_index_label.setMaximumHeight(20)
        self.image_layout.addWidget(self.slice_index_label)

        # Range slider.
        self.range_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.range_slider.setMaximumHeight(20)
        self.range_slider.setValue((0, 100))
        self.range_slider.valueChanged.connect(
            lambda ev: (self.update_image(), self.save_points()))
        self.image_layout.addWidget(self.range_slider)

        # Points.
        self.image_label = QLabel()
        self.image_label.installEventFilter(self) # for mouse wheel scroll
        self.image_label.setScaledContents(True)
        self.image_label.resizeEvent = self.on_resized
        self.image_label.mousePressEvent = self.add_point_at_click
        self.image_label.setMinimumSize(1, 1)
        self.image_layout.addWidget(self.image_label)

        # Stack scrollbar.
        self.scrollbar = QScrollBar(Qt.Vertical)
        self.scrollbar.setMaximumWidth(20)
        self.scrollbar.valueChanged.connect(self.update_image)

        # Image.
        self.image_widget = QWidget()
        self.image_widget.setLayout(self.image_layout)
        self.splitter.addWidget(self.image_widget)

        self.splitter.addWidget(self.scrollbar)

        # Bottom buttons
        self.nav_layout = QHBoxLayout()

        # Move to current depth button
        self.all_button = QPushButton('Move to current depth')
        self.all_button.clicked.connect(self.move_to_current_depth)
        self.nav_layout.addWidget(self.all_button)

        # Previous button
        self.prev_button = QPushButton('Previous')
        self.prev_button.setEnabled(False)
        self.prev_button.clicked.connect(lambda: self.navigate(-1))
        self.nav_layout.addWidget(self.prev_button)

        # Next button
        self.next_button = QPushButton('Next')
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(lambda: self.navigate(1))
        self.nav_layout.addWidget(self.next_button)

        self.layout.addLayout(self.nav_layout)

        self.setWindowTitle('Mesoscope GUI')
        self.resize(WIDTH, HEIGHT)
        self.show()

    # Folder opening
    # ---------------------------------------------------------------------------------------------

    def open_dialog(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.DirectoryOnly)

        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        file_view = dialog.findChild(QListView, 'listView')

        # to make it possible to select multiple directories:
        if file_view:
            file_view.setSelectionMode(QAbstractItemView.MultiSelection)
        f_tree_view = dialog.findChild(QTreeView)
        if f_tree_view:
            f_tree_view.setSelectionMode(QAbstractItemView.MultiSelection)

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
            self.folder_paths = self.find_image_folders(paths)

            self.folder_list.clear()
            # NOTE: remove "/reference" at the end
            self.folder_list.addItems([f[:-10] for f in self.folder_paths])

            self.current_folder_idx = 0
            self.update_folder()

    def find_image_folders(self, root_folders):
        image_folders = []
        for root_folder in root_folders:
            root_path = Path(root_folder)
            for subdir in root_path.rglob('reference'):
                if subdir.is_dir() and any(f.name.startswith("referenceImage.meta") for f in subdir.iterdir()):
                    image_folders.append(str(subdir))
        return image_folders

    def update_folder(self):
        self.current_stack_idx = 0
        self.load_points()
        # NOTE: load_points also loads the range values, which should come BEFORE image loading
        self.load_image_stack()
        self.update_image()

        self.prev_button.setEnabled(self.current_folder_idx > 0)
        self.next_button.setEnabled(self.current_folder_idx < len(self.folder_paths) - 1)

    def select_folder(self, folder_idx):
        self.current_folder_idx = folder_idx
        self.update_folder()

    def navigate(self, direction):
        self.current_folder_idx += direction
        self.folder_list.setCurrentRow(self.current_folder_idx)

    def load_image_stack(self):
        folder = self.folder_paths[self.current_folder_idx]
        stack_file = next(f for f in os.listdir(folder) if f.startswith("referenceImage.stack") and f.endswith(".tif"))
        self.stack_path = op.join(folder, stack_file)

        self.image_stack = iio.imread(self.stack_path)  # shape: nstacks, h, w
        assert self.image_stack.ndim == 3
        self.stack_count = self.image_stack.shape[0]
        self.scrollbar.setMaximum(self.stack_count - 1)
        self.image_stack = self.image_stack.astype(np.float32)

    def normalize_image(self, img):
        h = .001
        vmin, vmax = np.quantile(img, [h, 1-h])
        d = vmax - vmin

        v0, v1 = self.get_range()
        v0 /= 99.0
        v1 /= 99.0
        vmin += d * v0
        vmax -= d * (1.0 - v1)

        img = (img - vmin) / (vmax - vmin)
        img = np.clip(img, 0, 1)
        img = np.floor(img * 255).astype(np.uint8)

        return img

    def get_range(self):
        if self.range_slider is None:
            return None, None
        vmin, vmax = self.range_slider.value()
        return vmin, vmax

    # Coordinate transforms
    # ---------------------------------------------------------------------------------------------

    def to_relative(self, x, y):
        label_width, label_height = self.image_label.width(), self.image_label.height()
        pixmap = self.image_label.pixmap()

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
        label_width, label_height = self.image_label.width(), self.image_label.height()
        pixmap = self.image_label.pixmap()

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
        size = self.image_label.size()
        w, h = size.width(), size.height()
        pw = pixmap.width()
        ph = pixmap.height()

        if (w * ph > h * pw):
            m = (w - (pw * h / ph)) / 2
            m = int(m)
            self.image_label.setContentsMargins(m, 0, m, 0)
        else:
            m = (h - (ph * w / pw)) / 2
            m = int(m)
            self.image_label.setContentsMargins(0, m, 0, m)

    def update_image(self):
        self.current_stack_idx = self.scrollbar.value()
        if self.current_stack_idx >= self.stack_count:
            return
        img = self.image_stack[self.current_stack_idx, ...]
        img = self.normalize_image(img)

        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)

        self.pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(self.pixmap)

        self.slice_index_label.setText(
            f"Slice #{self.current_stack_idx + 1} / {self.stack_count}")

        self.update_margins()
        self.update_points()

    # Adding points
    # ---------------------------------------------------------------------------------------------

    def set_point_position(self, point_idx, xr, yr, stack_idx):
        self.points[point_idx]['coords'] = (xr, yr)
        self.points[point_idx]['stack_idx'] = stack_idx
        self.update_point_position(point_idx)
        self.update_point_filter(point_idx)

    def update_point_position(self, idx):
        xr, yr = self.points[idx].get('coords', (None, None))
        if xr is None:
            return
        x, y = self.to_absolute(xr, yr)
        if x is not None and y is not None:
            self.points_widgets[idx].move(x, y)

    def update_point_filter(self, idx):
        w = self.points_widgets[idx].widget
        stack_idx = self.points[idx].get('stack_idx', -1)
        blur = abs(self.current_stack_idx - stack_idx)
        blur = 0 if blur == 0 else 2 + .5 * blur * blur
        set_blur(w, blur)

    def add_point_at_click(self, event):
        x, y = event.pos().x(), event.pos().y()
        point_idx = next((i for i, point in enumerate(self.points) if not point), None)
        if point_idx is None:
            return
        assert 0 <= point_idx and point_idx < 3
        xr, yr = self.to_relative(x, y)
        self.set_point_position(point_idx, xr, yr, self.current_stack_idx)
        self.save_points()

    def move_to_current_depth(self, ev):
        for idx in range(3):
            self.points[idx]['stack_idx'] = self.current_stack_idx
            self.update_point_filter(idx)
        self.save_points()

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
        assert 0 <= idx and idx <= 2
        self.points[idx]['stack_idx'] = self.current_stack_idx
        self.update_point_filter(idx)

    def drag_point(self, event, w):
        idx = self._widget_idx(w)
        new_pos = w.pos() + event.pos() - self.drag_offset
        x, y = self.to_relative(new_pos.x(), new_pos.y())
        for i in range(3):
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
        r = RADIUS
        x, y = w.x() + r // 2, w.y() + r // 2
        xr, yr = self.to_relative(x, y)
        if xr < 0 or xr > 1 or yr < 0 or yr > 1:
            print(f"Deleting point {point_idx}")
            self.points[point_idx] = {}
        else:
            self.points[point_idx]['coords'] = xr, yr
        self.save_points()

    # Points file
    # ---------------------------------------------------------------------------------------------

    @property
    def points_file(self):
        return op.join(self.folder_paths[self.current_folder_idx], "referenceImage.points.json")

    def load_points(self, points_file=None):
        self.clear_points_struct()
        [p.clear() for p in self.points_widgets]

        points_file = points_file or self.points_file

        # Update the points structure.
        if op.exists(points_file):
            with open(points_file, 'r') as f:
                data = json.load(f)
            self.points = data['points']
            vrange = data.get('range', (0, 99))
            self.range_slider.setValue(tuple(vrange))

        # Update the points position on the image.
        for point_idx in range(3):
            self.update_point_position(point_idx)
            self.update_point_filter(point_idx)

        self.scrollbar.setValue(self.points[0].get('stack_idx', self.stack_count // 2))

    def save_points(self):
        if self.current_folder_idx >= len(self.folder_paths):
            return
        # print("Saving points", self.points)
        with open(self.points_file, 'w') as f:
            json.dump({'points': self.points, 'range': self.get_range()}, f, indent=2)
            f.write('\n')

    # Event handling
    # ---------------------------------------------------------------------------------------------

    def on_resized(self, ev):
        self.update_margins()
        for point_idx in range(3):
            self.update_point_position(point_idx)

    def eventFilter(self, obj, event):
        if obj == self.image_label and event.type() == event.Wheel:
            delta = event.angleDelta().y() // 120
            new_value = self.scrollbar.value() - delta
            new_value = max(0, min(self.scrollbar.maximum(), new_value))
            self.scrollbar.setValue(new_value)
            return True
        return super().eventFilter(obj, event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MesoscopeGUI()
    if sys.argv[1:]:
        gui.open(sys.argv[1:])
    sys.exit(app.exec_())
