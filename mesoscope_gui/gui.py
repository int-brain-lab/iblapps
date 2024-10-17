import sys
import os
import json

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QListWidget, QLabel,
    QScrollBar, QPushButton, QWidget, QMenu, QAction, QSplitter, QGraphicsOpacityEffect)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor

import imageio.v3 as iio
import numpy as np


opacity_effect = QGraphicsOpacityEffect()
opacity_effect.setOpacity(0.5)

RADIUS = 20


def set_widget_opaque(widget, is_opaque):
    widget.setGraphicsEffect(opacity_effect if not is_opaque else None)


class MesoscopeGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.current_folder_idx = 0
        self.folder_paths = []

        self.stack_count = 0
        self.current_stack_idx = 0


        self.pixmap = None

        self.init_ui()

        self.points = [{} for _ in range(3)]  # stack_idx, coords

        self.points_widgets = []
        for point_idx in range(3):
            color = ['red', 'green', 'blue'][point_idx]
            x = y = -100
            self.add_point_widget(x, y, color, point_idx)

    def init_ui(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')
        open_action = QAction('Open', self)
        open_action.triggered.connect(self.open_dialog)
        file_menu.addAction(open_action)

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
        self.image_label.resizeEvent = lambda event: self.update_margins()
        self.image_label.mousePressEvent = self.add_point_at_click
        self.image_label.setMinimumSize(1, 1)
        self.image_layout.addWidget(self.image_label)

        self.scrollbar = QScrollBar(Qt.Vertical)
        self.scrollbar.setMaximumWidth(20)
        self.scrollbar.valueChanged.connect(self.update_image)
        # self.image_layout.addWidget(self.scrollbar)

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
        self.resize(800, 600)
        self.show()

    def open_dialog(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        if dialog.exec_():
            self.open(dialog.selectedFiles())

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
        self.prev_button.setEnabled(self.current_folder_idx > 0)
        self.next_button.setEnabled(self.current_folder_idx < len(self.folder_paths) - 1)

    def select_folder(self, folder_index):
        self.current_folder_idx = folder_index
        self.update_folder()

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
        self.load_points()

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

    def to_relative(self, x, y):
        pixmap = self.pixmap
        if not pixmap:
            return
        size = self.image_label.size()
        w, h = size.width(), size.height()
        pw = pixmap.width()
        ph = pixmap.height()

        ih = self.image_stack.shape[1]
        iw = self.image_stack.shape[2]

        if (w * ph > h * pw):
            m = (w - (pw * h / ph)) / 2
            return ((x-m)/w, y/h)

        else:
            m = (h - (ph * w / pw)) / 2
            return ((x)/w, (y-m)/h)

    def update_image(self):
        self.current_stack_idx = self.scrollbar.value()
        if self.current_stack_idx >= self.stack_count:
            return
        img = self.image_stack[self.current_stack_idx, ...]
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)

        self.pixmap = QPixmap.fromImage(qimg)
        # pixmap = QPixmap.fromImage(qimg).scaled(
        #     self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # painter = QPainter(self.pixmap)
        # for p in self.points.get(self.current_stack_idx, []):
        #     opacity = 1.0 if p['stack_idx'] == self.current_stack_idx else 0.5
        #     painter.setOpacity(opacity)
        #     color = [QColor('red'), QColor('green'), QColor('blue')][p['point_idx']]
        #     painter.setPen(color)
        #     painter.setBrush(color)
        #     painter.drawEllipse(p['coords'][0] - 5, p['coords'][1] - 5, 10, 10)
        # painter.end()
        self.image_label.setPixmap(self.pixmap)

        self.update_margins()

    def navigate(self, direction):
        self.current_folder_idx += direction
        self.folder_list.setCurrentRow(self.current_folder_idx)

    def add_point_widget(self, x, y, color, point_idx):
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

    def add_point_at_click(self, event):
        x, y = event.pos().x(), event.pos().y()
        point_idx = next((i for i, point in enumerate(self.points) if not point), None)
        if point_idx is None:
            return
        assert 0 <= point_idx and point_idx < 3

        r = RADIUS
        self.points_widgets[point_idx].move(x - r // 2, y - r // 2)

        self.points[point_idx]['coords'] = self.to_relative(x, y)
        self.points[point_idx]['stack_idx'] = self.current_stack_idx
        self.save_points()

    def start_drag(self, event, point_label):
        self.drag_offset = event.pos()
        point_label.raise_()

    def drag_point(self, event, point_label):
        new_pos = point_label.pos() + event.pos() - self.drag_offset
        point_label.move(new_pos)

    def end_drag(self, event, point_label, point_idx):
        r = RADIUS
        x, y = point_label.x() + r // 2, point_label.y() + r // 2

        self.points[point_idx]['coords'] = self.to_relative(x, y)
        self.save_points()

    def clear_points(self):
        for widget in self.points_widgets:
            widget.move(-100, -100)

    def eventFilter(self, obj, event):
        if obj == self.image_label and event.type() == event.Wheel:
            delta = event.angleDelta().y() // 120
            new_value = self.scrollbar.value() - delta
            new_value = max(0, min(self.scrollbar.maximum(), new_value))
            self.scrollbar.setValue(new_value)
            return True
        return super().eventFilter(obj, event)

    @property
    def points_file(self):
        return os.path.join(self.folder_paths[self.current_folder_idx], "referenceImage.points.json")

    def load_points(self):
        points_file = self.points_file
        if not os.path.exists(points_file):
            return
        with open(points_file, 'r') as f:
            data = json.load(f)
        self.points = data['points']

    def save_points(self):
        if self.current_folder_idx >= len(self.folder_paths):
            return

        # DEBUG
        print(self.points)
        return

        with open(points_file, 'w') as f:
            json.dump({'points': self.points}, f)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MesoscopeGUI()
    if sys.argv[1:]:
        gui.open(sys.argv[1:])
    sys.exit(app.exec_())
