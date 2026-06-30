from __future__ import annotations

from PySide6.QtWidgets import QDialog, QListWidget, QVBoxLayout, QPushButton, QDialogButtonBox

from upscalerr.util.win32 import enumerate_windows, WindowInfo


class GamePickerDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Game Window")
        self._selected: WindowInfo | None = None

        layout = QVBoxLayout(self)
        self.list_widget = QListWidget(self)
        layout.addWidget(self.list_widget)

        refresh = QPushButton("Refresh", self)
        refresh.clicked.connect(self._populate)
        layout.addWidget(refresh)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._populate()

    def _populate(self) -> None:
        self.list_widget.clear()
        for win in enumerate_windows():
            self.list_widget.addItem(win.title)
            item = self.list_widget.item(self.list_widget.count() - 1)
            item.setData(256, win)

    @property
    def selected(self) -> WindowInfo | None:
        item = self.list_widget.currentItem()
        if item is None:
            return None
        return item.data(256)
