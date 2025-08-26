from qtpy import QtWidgets
from ibllib.qc.critical_reasons import CriticalInsertionNote


PLUGIN_NAME = "QC"

def setup(parent):
    parent.qc_dialog = QCDialog(parent)
    parent.qc_dialog.accepted.connect(lambda: callback(parent))


def display(parent, shank):
    parent.qc_dialog.setWindowTitle(f"QC assessment {shank}")
    parent.qc_dialog.exec_()


def callback(parent):

    align_qc = parent.qc_dialog.align_qc.currentText()
    ephys_qc = parent.qc_dialog.ephys_qc.currentText()
    ephys_desc = [btn.text() for btn in parent.qc_dialog.desc_buttons.buttons() if btn.isChecked()]
    resolve = parent.qc_dialog.resolve.currentData()
    if parent.loaddata.configs is None:
        upload = parent.loaddata.get_selected_shank().loaders['upload']
    else:
        upload = parent.loaddata.get_selected_shank()['dense'].loaders['upload']

    upload.get_qc_string(align_qc, ephys_qc, ephys_desc, resolve)
    return


class QCDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(QCDialog, self).__init__(parent)

        self.setWindowTitle('QC assessment')
        self.resize(300, 150)
        self.setup()

    def setup(self) -> None:

        # Alignment QC
        self.align_qc_label = QtWidgets.QLabel("Confidence of alignment:")
        self.align_qc = QtWidgets.QComboBox()
        self.align_qc.addItems(["High", "Medium", "Low"])

        # Ephys QC
        self.ephys_qc_label = QtWidgets.QLabel("QC for ephys recording:")
        self.ephys_qc = QtWidgets.QComboBox()
        self.ephys_qc.addItems(["Pass", "Warning", "Critical"])

        # Problem Descriptions
        self.desc_buttons = QtWidgets.QButtonGroup()
        self.desc_buttons.setExclusive(False)
        self.desc_group = QtWidgets.QGroupBox("Describe problem with recording:")
        self.desc_layout = QtWidgets.QVBoxLayout()

        for i, label in enumerate(CriticalInsertionNote.descriptions_gui):
            checkbox = QtWidgets.QCheckBox(label)
            self.desc_buttons.addButton(checkbox, i)
            self.desc_layout.addWidget(checkbox)
        self.desc_group.setLayout(self.desc_layout)

        # Force upload option
        self.resolve_label = QtWidgets.QLabel("Do you want to resolve this alignment with the current alignment?:")
        self.resolve = QtWidgets.QComboBox()
        self.resolve.addItem("No", False)
        self.resolve.addItem("Yes", True)

        # Dialog buttons
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.on_accept)
        self.button_box.rejected.connect(self.reject)

        # Assemble layout
        dialog_layout = QtWidgets.QVBoxLayout()
        dialog_layout.addWidget(self.align_qc_label)
        dialog_layout.addWidget(self.align_qc)
        dialog_layout.addWidget(self.ephys_qc_label)
        dialog_layout.addWidget(self.ephys_qc)
        dialog_layout.addWidget(self.desc_group)
        dialog_layout.addWidget(self.resolve_label)
        dialog_layout.addWidget(self.resolve)
        dialog_layout.addWidget(self.button_box)
        self.setLayout(dialog_layout)

    def on_accept(self) -> None:
        """Validation before accepting the dialog."""
        ephys_qc = self.ephys_qc.currentText()
        ephys_desc = [btn.text() for btn in self.desc_buttons.buttons() if btn.isChecked()]

        if ephys_qc != 'Pass' and not ephys_desc:
            QtWidgets.QMessageBox.warning(self, "Missing Information", "You must select a reason for QC choice")
            return
        self.accept()