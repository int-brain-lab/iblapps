# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import sys
import os
import json

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QListWidget, QLabel,
    QScrollBar, QPushButton, QWidget, QMenu, QAction, QSplitter, QListView, QAbstractItemView,
    QTreeView, QGraphicsOpacityEffect)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QKeySequence

import imageio.v3 as iio
import numpy as np


# -------------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------------

opacity_effect = QGraphicsOpacityEffect()
opacity_effect.setOpacity(0.5)

WIDTH = 1024
HEIGHT = 768
RADIUS = 20


# -------------------------------------------------------------------------------------------------
# Util functions
# -------------------------------------------------------------------------------------------------

def set_widget_opaque(widget, is_opaque):
    widget.setGraphicsEffect(opacity_effect if not is_opaque else None)


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

        self.pixmap = None
        self._clear_points_struct()

        super().__init__()
        self.init_ui()
        self._init_point_widgets()

    def _clear_points_struct(self):
        self.points = [{} for _ in range(3)]  # stack_idx, coords

    def _init_point_widgets(self):
        self.points_widgets = []
        for point_idx in range(3):
            color = ['red', 'green', 'blue'][point_idx]
            x = y = -100
            self._add_point_widget(x, y, color, point_idx)

    # UI
    # ---------------------------------------------------------------------------------------------

    def init_ui(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')

        open_action = QAction('Open', self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_dialog)
        file_menu.addAction(open_action)

        quit_action = QAction('Quit', self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)

        self.splitter = QSplitter(Qt.Horizontal)
        self.layout.addWidget(self.splitter)

        self.folder_list = QListWidget()
        self.folder_list.setMaximumWidth(200)
        self.folder_list.currentRowChanged.connect(self.update_folder)
        self.folder_list.itemClicked.connect(lambda: self.select_folder(self.folder_list.currentRow()))
        self.splitter.addWidget(self.folder_list)

        self.image_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.installEventFilter(self) # for mouse wheel scroll
        self.image_label.setScaledContents(True)
        self.image_label.resizeEvent = self.on_resized
        self.image_label.mousePressEvent = self.add_point_at_click
        self.image_label.setMinimumSize(1, 1)
        self.image_layout.addWidget(self.image_label)

        self.scrollbar = QScrollBar(Qt.Vertical)
        self.scrollbar.setMaximumWidth(20)
        self.scrollbar.valueChanged.connect(self.update_image)

        self.image_widget = QWidget()
        self.image_widget.setLayout(self.image_layout)
        self.splitter.addWidget(self.image_widget)

        self.splitter.addWidget(self.scrollbar)

        self.nav_layout = QHBoxLayout()
        self.prev_button = QPushButton('Previous')
        self.prev_button.clicked.connect(lambda: self.navigate(-1))
        self.next_button = QPushButton('Next')
        self.next_button.clicked.connect(lambda: self.navigate(1))
        self.nav_layout.addWidget(self.prev_button)
        self.nav_layout.addWidget(self.next_button)
        self.layout.addLayout(self.nav_layout)

        self.setWindowTitle('Mesoscope GUI')
        self.resize(WIDTH, HEIGHT)
        self.show()

    def _add_point_widget(self, x, y, color, point_idx):
        point_label = QLabel(self.image_label)
        r = 20
        point_label.setFixedSize(r, r)

        pixmap = QPixmap(r, r)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setBrush(QColor(color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, r, r)
        painter.end()

        point_label.setPixmap(pixmap)
        point_label.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        point_label.move(x - r // 2, y - r // 2)
        point_label.show()

        point_label.mousePressEvent = lambda event: self.start_drag(event, point_label)
        point_label.mouseMoveEvent = lambda event: self.drag_point(event, point_label)
        point_label.mouseReleaseEvent = lambda event: self.end_drag(event, point_label, point_idx)

        self.points_widgets.append(point_label)

    # Folder opening
    # ---------------------------------------------------------------------------------------------

    def open_dialog(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        # dialog.setOption(QFileDialog.ShowDirsOnly, True)
        # if dialog.exec_():
        #     self.open(dialog.selectedFiles())

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

    def open(self, paths):
            self.folder_paths = self.find_image_folders(paths)

            self.folder_list.clear()
            self.folder_list.addItems([os.path.relpath(f, os.path.dirname(self.folder_paths[0])) for f in self.folder_paths])

            self.current_folder_idx = 0
            self.update_folder()

    def find_image_folders(self, root_folders):
        image_folders = []
        for root_folder in root_folders:
            for subdir, _, files in os.walk(root_folder):
                if any(f.startswith("referenceImage.meta") for f in files):
                    image_folders.append(subdir)
        return image_folders

    def update_folder(self):
        self.current_stack_idx = 0
        self.load_image_stack()
        self.update_image()
        self.load_points()

        self.prev_button.setEnabled(self.current_folder_idx > 0)
        self.next_button.setEnabled(self.current_folder_idx < len(self.folder_paths) - 1)

    def select_folder(self, folder_index):
        self.current_folder_idx = folder_index
        self.update_folder()

    def navigate(self, direction):
        self.current_folder_idx += direction
        self.folder_list.setCurrentRow(self.current_folder_idx)

    def load_image_stack(self):
        folder = self.folder_paths[self.current_folder_idx]
        stack_file = next(f for f in os.listdir(folder) if f.startswith("referenceImage.stack") and f.endswith(".tif"))
        self.stack_path = os.path.join(folder, stack_file)

        self.image_stack = iio.imread(self.stack_path)  # shape: nstacks, h, w
        assert self.image_stack.ndim == 3
        self.stack_count = self.image_stack.shape[0]

        self.image_stack = self.image_stack.astype(np.float32)
        h = .01
        q0, q1 = np.quantile(self.image_stack, [h, 1-h])
        self.image_stack = (self.image_stack - q0) / (q1 - q0)
        self.image_stack = np.clip(self.image_stack, 0, 1)
        self.image_stack = np.floor(self.image_stack * 255).astype(np.uint8)

        self.scrollbar.setMaximum(self.stack_count - 1)
        self.scrollbar.setValue(self.stack_count // 2)

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
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)

        self.pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(self.pixmap)
        self.update_margins()

    # Adding points
    # ---------------------------------------------------------------------------------------------

    def clear_points(self):
        self._clear_points_struct()
        for widget, point in zip(self.points_widgets, self.points):
            widget.move(-100, -100)

    def set_point_position(self, point_idx, xr, yr, stack_idx):
        self.points[point_idx]['coords'] = (xr, yr)
        self.points[point_idx]['stack_idx'] = stack_idx
        self.update_point_position(point_idx)

    def update_point_position(self, point_idx):
        xr, yr = self.points[point_idx].get('coords', (None, None))
        if xr is None:
            return
        x, y = self.to_absolute(xr, yr)
        self.points_widgets[point_idx].move(x - RADIUS // 2, y - RADIUS // 2)

    def add_point_at_click(self, event):
        x, y = event.pos().x(), event.pos().y()
        point_idx = next((i for i, point in enumerate(self.points) if not point), None)
        if point_idx is None:
            return
        assert 0 <= point_idx and point_idx < 3
        xr, yr = self.to_relative(x, y)
        self.set_point_position(point_idx, xr, yr, self.current_stack_idx)
        self.save_points()

    # Points drag and drop
    # ---------------------------------------------------------------------------------------------

    def start_drag(self, event, point_label):
        self.drag_offset = event.pos()
        point_label.raise_()

    def drag_point(self, event, point_label):
        new_pos = point_label.pos() + event.pos() - self.drag_offset
        point_label.move(new_pos)

    def end_drag(self, event, point_label, point_idx):
        r = RADIUS
        x, y = point_label.x() + r // 2, point_label.y() + r // 2
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
        return os.path.join(self.folder_paths[self.current_folder_idx], "referenceImage.points.json")

    def load_points(self):
        self.clear_points()

        points_file = self.points_file

        # Update the points structure.
        if os.path.exists(points_file):
            with open(points_file, 'r') as f:
                data = json.load(f)
            self.points = data['points']

        # Update the points position on the image.
        for point_idx in range(3):
            self.update_point_position(point_idx)

    def save_points(self):
        if self.current_folder_idx >= len(self.folder_paths):
            return
        print(self.points)
        with open(self.points_file, 'w') as f:
            json.dump({'points': self.points}, f)

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
