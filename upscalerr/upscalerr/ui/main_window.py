from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from upscalerr.app.lifecycle import shutdown_runtime
from upscalerr.capture.window_target import WindowTarget
from upscalerr.ui.game_picker_dialog import GamePickerDialog
from upscalerr.util.win32 import WindowInfo, enumerate_windows, get_foreground_window


class MainWindow(QMainWindow):
    def __init__(self, app_core, parent=None) -> None:
        super().__init__(parent)
        self._core = app_core
        self._selected_window_hwnd: int | None = None

        self.setWindowTitle("Upscalerr")
        self.setMinimumSize(1120, 760)

        self._build_ui()
        self._sync_controls_from_core()
        self._refresh_windows(select_foreground=True)
        self._refresh_engine_status()
        self._update_status()

        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._update_status)
        self._stats_timer.start(400)

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(22, 22, 22, 22)
        root_layout.setSpacing(18)

        self.sidebar = self._make_panel("Sidebar")
        self.sidebar.setFixedWidth(340)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setSpacing(16)

        hero = self._make_panel("Hero")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setSpacing(8)
        title = QLabel("Upscalerr", self)
        title.setObjectName("AppTitle")
        subtitle = QLabel("A Lossless Scaling-style control panel for real-time game upscaling.", self)
        subtitle.setWordWrap(True)
        subtitle.setObjectName("AppSubtitle")
        self.state_badge = QLabel("Idle", self)
        self.state_badge.setObjectName("StatusBadge")
        self.engine_notice = QLabel("", self)
        self.engine_notice.setObjectName("MutedText")
        self.engine_notice.setWordWrap(True)
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)
        hero_layout.addWidget(self.state_badge, alignment=Qt.AlignLeft)
        hero_layout.addWidget(self.engine_notice)
        sidebar_layout.addWidget(hero)

        target_card = self._make_panel("Target")
        target_layout = QVBoxLayout(target_card)
        target_layout.setSpacing(10)

        target_header = self._section_header("Capture target", "Pick the game window to upscale")
        target_layout.addLayout(target_header)

        self.window_combo = QComboBox(self)
        self.window_combo.currentIndexChanged.connect(self._update_selected_window_details)
        target_layout.addWidget(self.window_combo)

        self.target_detail = QLabel("No window selected.", self)
        self.target_detail.setWordWrap(True)
        self.target_detail.setObjectName("MutedText")
        target_layout.addWidget(self.target_detail)

        button_row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh", self)
        self.refresh_btn.setObjectName("SecondaryButton")
        self.refresh_btn.clicked.connect(lambda: self._refresh_windows(select_foreground=False))
        self.foreground_btn = QPushButton("Use Foreground", self)
        self.foreground_btn.setObjectName("SecondaryButton")
        self.foreground_btn.clicked.connect(self._use_foreground_window)
        button_row.addWidget(self.refresh_btn)
        button_row.addWidget(self.foreground_btn)
        target_layout.addLayout(button_row)

        self.pick_btn = QPushButton("Choose from list", self)
        self.pick_btn.setObjectName("GhostButton")
        self.pick_btn.clicked.connect(self._open_picker)
        target_layout.addWidget(self.pick_btn)
        sidebar_layout.addWidget(target_card)

        pipeline_card = self._make_panel("Pipeline")
        pipeline_layout = QVBoxLayout(pipeline_card)
        pipeline_layout.setSpacing(10)
        pipeline_layout.addLayout(self._section_header("Runtime settings", "These values are applied before Start"))

        self.framegen_check = QCheckBox("Enable frame generation", self)
        self.transparent_input_check = QCheckBox("Transparent input overlay", self)
        pipeline_layout.addWidget(self.framegen_check)
        pipeline_layout.addWidget(self.transparent_input_check)

        self.capture_fps_spin = self._make_spinbox(30, 360, "fps")
        self.present_fps_spin = self._make_spinbox(30, 360, "fps")
        self.monitor_index_spin = self._make_spinbox(0, 7, "")

        pipeline_layout.addWidget(self._make_labeled_row("Capture FPS", self.capture_fps_spin))
        pipeline_layout.addWidget(self._make_labeled_row("Present FPS", self.present_fps_spin))
        pipeline_layout.addWidget(self._make_labeled_row("Monitor index", self.monitor_index_spin))

        self.scale_hint = QLabel("Scale is driven by the active engine configuration.", self)
        self.scale_hint.setObjectName("MutedText")
        self.scale_hint.setWordWrap(True)
        pipeline_layout.addWidget(self.scale_hint)
        sidebar_layout.addWidget(pipeline_card)

        action_card = self._make_panel("Actions")
        action_layout = QVBoxLayout(action_card)
        action_layout.setSpacing(10)
        action_layout.addLayout(self._section_header("Session control", "Start the overlay when the game is selected"))

        action_row = QHBoxLayout()
        self.start_btn = QPushButton("Start", self)
        self.start_btn.setObjectName("PrimaryButton")
        self.stop_btn = QPushButton("Stop", self)
        self.stop_btn.setObjectName("DangerButton")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        action_row.addWidget(self.start_btn)
        action_row.addWidget(self.stop_btn)
        action_layout.addLayout(action_row)

        self.hint_label = QLabel(
            "1. Select the game window\n2. Tune the runtime settings\n3. Click Start to launch the overlay",
            self,
        )
        self.hint_label.setObjectName("MutedText")
        self.hint_label.setWordWrap(True)
        action_layout.addWidget(self.hint_label)
        sidebar_layout.addWidget(action_card)
        sidebar_layout.addStretch(1)

        main_panel = self._make_panel("Main")
        main_layout = QVBoxLayout(main_panel)
        main_layout.setSpacing(18)

        top_strip = QHBoxLayout()
        banner = QVBoxLayout()
        banner.setSpacing(4)
        banner_title = QLabel("Game Upscaling Dashboard", self)
        banner_title.setObjectName("SectionTitle")
        banner_subtitle = QLabel("A compact launcher inspired by Lossless Scaling.", self)
        banner_subtitle.setObjectName("MutedText")
        banner_subtitle.setWordWrap(True)
        banner.addWidget(banner_title)
        banner.addWidget(banner_subtitle)
        top_strip.addLayout(banner)
        top_strip.addStretch(1)
        self.window_summary = QLabel("No capture target chosen", self)
        self.window_summary.setObjectName("SummaryChip")
        top_strip.addWidget(self.window_summary)
        main_layout.addLayout(top_strip)

        metrics_card = self._make_panel("Metrics")
        metrics_layout = QGridLayout(metrics_card)
        metrics_layout.setHorizontalSpacing(18)
        metrics_layout.setVerticalSpacing(12)
        metrics_layout.addWidget(self._metric_label("Frame"), 0, 0)
        metrics_layout.addWidget(self._metric_label("Inference"), 0, 1)
        metrics_layout.addWidget(self._metric_label("Present"), 0, 2)
        metrics_layout.addWidget(self._metric_label("VRAM"), 0, 3)

        self.frame_value = self._metric_value("0")
        self.inference_value = self._metric_value("0.0 ms")
        self.present_value = self._metric_value("0.0 FPS")
        self.vram_value = self._metric_value("0 MB")
        metrics_layout.addWidget(self.frame_value, 1, 0)
        metrics_layout.addWidget(self.inference_value, 1, 1)
        metrics_layout.addWidget(self.present_value, 1, 2)
        metrics_layout.addWidget(self.vram_value, 1, 3)
        main_layout.addWidget(metrics_card)

        instructions_card = self._make_panel("Quick Guide")
        instructions_layout = QVBoxLayout(instructions_card)
        instructions_layout.setSpacing(8)
        guide = QLabel(
            "This launcher keeps the core pipeline intact while giving you a cleaner front panel.\n"
            "Use the left column to pick the game, adjust runtime behavior, and start the overlay.",
            self,
        )
        guide.setWordWrap(True)
        guide.setObjectName("GuideText")
        instructions_layout.addWidget(guide)

        checklist = QLabel(
            "• Window capture uses the selected game title and HWND\n"
            "• Frame generation can be toggled before launch\n"
            "• FPS and monitor settings are applied from this panel",
            self,
        )
        checklist.setWordWrap(True)
        checklist.setObjectName("MutedText")
        instructions_layout.addWidget(checklist)
        main_layout.addWidget(instructions_card)
        main_layout.addStretch(1)

        root_layout.addWidget(self.sidebar)
        root_layout.addWidget(main_panel, 1)

        self.framegen_check.stateChanged.connect(self._push_controls_to_core)
        self.transparent_input_check.stateChanged.connect(self._push_controls_to_core)
        self.capture_fps_spin.valueChanged.connect(self._push_controls_to_core)
        self.present_fps_spin.valueChanged.connect(self._push_controls_to_core)
        self.monitor_index_spin.valueChanged.connect(self._push_controls_to_core)
        self.framegen_check.stateChanged.connect(self._refresh_engine_status)

    def _make_panel(self, object_name: str) -> QFrame:
        panel = QFrame(self)
        panel.setObjectName(object_name)
        panel.setProperty("panel", True)
        return panel

    def _section_header(self, title: str, subtitle: str) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setSpacing(2)
        heading = QLabel(title, self)
        heading.setObjectName("SectionTitle")
        sub = QLabel(subtitle, self)
        sub.setObjectName("MutedText")
        sub.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(sub)
        return layout

    def _make_labeled_row(self, label: str, widget: QWidget) -> QWidget:
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        text = QLabel(label, self)
        text.setMinimumWidth(110)
        layout.addWidget(text)
        layout.addWidget(widget, 1)
        return row

    def _make_spinbox(self, minimum: int, maximum: int, suffix: str) -> QSpinBox:
        spin = QSpinBox(self)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(1)
        if suffix:
            spin.setSuffix(f" {suffix}")
        return spin

    def _metric_label(self, text: str) -> QLabel:
        label = QLabel(text, self)
        label.setObjectName("MetricLabel")
        return label

    def _metric_value(self, text: str) -> QLabel:
        label = QLabel(text, self)
        label.setObjectName("MetricValue")
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        return label

    def _sync_controls_from_core(self) -> None:
        config = getattr(self._core, "_config", {})
        self.framegen_check.setChecked(bool(config.get("pipeline", {}).get("frame_generation", True)))
        self.transparent_input_check.setChecked(bool(config.get("overlay", {}).get("transparent_input", True)))
        self.capture_fps_spin.setValue(int(config.get("capture", {}).get("max_fps", 240)))
        self.present_fps_spin.setValue(int(config.get("pipeline", {}).get("target_present_fps", 120)))
        self.monitor_index_spin.setValue(int(config.get("overlay", {}).get("monitor_index", 0)))
        self._refresh_engine_status()

    def _push_controls_to_core(self, *_args) -> None:
        self._core.set_frame_generation(self.framegen_check.isChecked())
        self._core.set_capture_fps(self.capture_fps_spin.value())
        self._core.set_target_present_fps(self.present_fps_spin.value())
        self._core.set_monitor_index(self.monitor_index_spin.value())
        self._core.set_transparent_input(self.transparent_input_check.isChecked())
        self._refresh_engine_status()

    def _refresh_engine_status(self, *_args) -> None:
        warnings = self._core.engine_warnings()
        if not self._core.engine_ready():
            self.engine_notice.setText(
                "TensorRT engines are missing. Build them first, then relaunch the GUI."
            )
            self.start_btn.setEnabled(False)
        else:
            if warnings:
                self.engine_notice.setText(" ".join(warnings))
            else:
                self.engine_notice.setText("TensorRT engines are ready.")
            if not self._core.running and self._current_window_info() is not None:
                self.start_btn.setEnabled(True)

    def _refresh_windows(self, select_foreground: bool = False) -> None:
        current_hwnd = self._current_window_hwnd()
        self.window_combo.blockSignals(True)
        self.window_combo.clear()

        windows = enumerate_windows()
        if not windows:
            self.window_combo.addItem("No windows found", None)
            self.window_combo.setEnabled(False)
            self._selected_window_hwnd = None
            self.target_detail.setText("Open a game or app window, then press Refresh.")
        else:
            self.window_combo.setEnabled(True)
            for win in windows:
                self.window_combo.addItem(self._format_window_title(win), userData=win)
            target_hwnd = current_hwnd
            if select_foreground:
                fg = get_foreground_window()
                if fg is not None:
                    target_hwnd = fg.hwnd
            if target_hwnd is not None and not self._select_window_by_hwnd(target_hwnd):
                self.window_combo.setCurrentIndex(0)
            elif target_hwnd is None:
                self.window_combo.setCurrentIndex(0)

        self.window_combo.blockSignals(False)
        self._update_selected_window_details()

    def _use_foreground_window(self) -> None:
        foreground = get_foreground_window()
        if foreground is None:
            QMessageBox.information(self, "Upscalerr", "No foreground window could be detected.")
            return

        if not self._select_window_by_hwnd(foreground.hwnd):
            self.window_combo.addItem(self._format_window_title(foreground), userData=foreground)
            self.window_combo.setCurrentIndex(self.window_combo.count() - 1)
        self._update_selected_window_details()

    def _open_picker(self) -> None:
        dialog = GamePickerDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return

        selected = dialog.selected
        if selected is None:
            QMessageBox.warning(self, "Upscalerr", "Select a window before continuing.")
            return

        if not self._select_window_by_hwnd(selected.hwnd):
            self.window_combo.addItem(self._format_window_title(selected), userData=selected)
            self.window_combo.setCurrentIndex(self.window_combo.count() - 1)
        self._update_selected_window_details()

    def _select_window_by_hwnd(self, hwnd: int) -> bool:
        for idx in range(self.window_combo.count()):
            info = self.window_combo.itemData(idx)
            if isinstance(info, WindowInfo) and info.hwnd == hwnd:
                self.window_combo.setCurrentIndex(idx)
                return True
        return False

    def _current_window_hwnd(self) -> int | None:
        info = self.window_combo.currentData()
        if isinstance(info, WindowInfo):
            return info.hwnd
        return None

    def _current_window_info(self) -> WindowInfo | None:
        info = self.window_combo.currentData()
        return info if isinstance(info, WindowInfo) else None

    def _format_window_title(self, info: WindowInfo) -> str:
        return f"{info.title}  [HWND {info.hwnd}]"

    def _update_selected_window_details(self, *_args) -> None:
        info = self._current_window_info()
        if info is None:
            self._selected_window_hwnd = None
            self.window_summary.setText("No capture target chosen")
            self.target_detail.setText("Pick a game or app window to begin.")
            self.start_btn.setEnabled(False)
            return

        left, top, right, bottom = info.rect
        width = right - left
        height = bottom - top
        self._selected_window_hwnd = info.hwnd
        self.window_summary.setText(f"{width}x{height}")
        self.target_detail.setText(f"Selected: {info.title}\nPosition: {left}, {top} -> {right}, {bottom}")
        if not self._core.running:
            self.start_btn.setEnabled(self._core.engine_ready())

    def _on_start(self) -> None:
        self._push_controls_to_core()

        if not self._core.engine_ready():
            QMessageBox.warning(
                self,
                "Upscalerr",
                "TensorRT engines are missing. Build the engines first, then relaunch the GUI.",
            )
            self._refresh_engine_status()
            return

        info = self._current_window_info()
        if info is None:
            QMessageBox.warning(self, "Upscalerr", "Select a target window before starting.")
            return

        target = WindowTarget.from_window(info)
        try:
            self._core.start(target)
        except Exception as exc:
            QMessageBox.critical(self, "Upscalerr", str(exc))
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.state_badge.setText("Running")
        self._update_status()

    def _on_stop(self) -> None:
        self._core.stop()
        self.start_btn.setEnabled(self._core.engine_ready() and self._current_window_info() is not None)
        self.stop_btn.setEnabled(False)
        self.state_badge.setText("Idle")
        self._update_status()

    def _update_status(self) -> None:
        fs = self._core.frame_state
        ctx = self._core.device_context
        self.frame_value.setText(str(fs.frame_id))
        self.inference_value.setText(f"{fs.inference_ms:.2f} ms")
        self.present_value.setText(f"{fs.present_fps:.1f} FPS")
        self.vram_value.setText(f"{ctx.vram_used_mb():.0f} MB")
        if self._core.running:
            self.state_badge.setText("Running")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
        else:
            self.state_badge.setText("Idle")
            self.start_btn.setEnabled(self._core.engine_ready() and self._current_window_info() is not None)
            self.stop_btn.setEnabled(False)

    def closeEvent(self, event) -> None:
        shutdown_runtime(self._core)
        super().closeEvent(event)
