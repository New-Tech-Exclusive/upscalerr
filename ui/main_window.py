import sys
import os
from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot, QSize
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QGroupBox, QComboBox, QCheckBox,
    QPushButton, QListWidget, QListWidgetItem, QMessageBox,
    QFrame, QSizePolicy, QStackedWidget
)
from PySide6.QtGui import QIcon, QFont

import win32gui
import win32process
import win32con

from ipc_client import IpcClient


class TelemetryThread(QThread):
    stats_updated = Signal(float, float)

    def __init__(self, client: IpcClient, parent=None):
        super().__init__(parent)
        self.client = client
        self.running = True

    def run(self):
        while self.running:
            fps, latency = self.client.get_stats()
            self.stats_updated.emit(fps, latency)
            self.msleep(100)  # Poll every 100ms

    def stop(self):
        self.running = False


def make_separator():
    """Create a thin horizontal separator line."""
    sep = QFrame()
    sep.setObjectName("separator")
    sep.setFrameShape(QFrame.HLine)
    sep.setFixedHeight(1)
    return sep


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.client = IpcClient()
        self.is_scaling = False
        self.bound_hwnd = None
        self.bound_title = ""

        self.setWindowTitle("Upscalerr")
        self.setMinimumSize(400, 500)
        self.resize(420, 720)
        self.init_ui()

        # Load stylesheet with resolved resource paths
        res_dir = os.path.join(os.path.dirname(__file__), "resources")
        qss_path = os.path.join(res_dir, "style.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r") as f:
                qss = f.read()
            icons_dir = os.path.join(res_dir, "icons").replace("\\", "/")
            qss = qss.replace('url("ui/resources/icons/', f'url("{icons_dir}/')
            self.setStyleSheet(qss)

        # Start telemetry loop
        self.telemetry = TelemetryThread(self.client, self)
        self.telemetry.stats_updated.connect(self.update_stats)
        self.telemetry.start()

        # Timer to refresh open window targets
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.populate_windows)
        self.refresh_timer.start(3000)

    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root = QVBoxLayout(central_widget)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        # ───────────── Header ─────────────
        header_layout = QHBoxLayout()
        header_layout.setSpacing(0)

        title_block = QVBoxLayout()
        title_block.setSpacing(2)

        title_label = QLabel("UPSCALERR", self)
        title_label.setObjectName("appTitle")
        title_block.addWidget(title_label)

        subtitle_label = QLabel("NVIDIA AI Upscaler & Frame Generator", self)
        subtitle_label.setObjectName("appSubtitle")
        title_block.addWidget(subtitle_label)

        header_layout.addLayout(title_block)
        header_layout.addStretch()

        # Version badge
        version_label = QLabel("v0.1α", self)
        version_label.setStyleSheet(
            "color: #555580; font-size: 11px; font-weight: 600; "
            "padding: 2px 8px; border: 1px solid #2a2a4a; border-radius: 4px;"
        )
        header_layout.addWidget(version_label, alignment=Qt.AlignTop)

        root.addLayout(header_layout)
        root.addWidget(make_separator())

        # ───────────── Scaling Type Section ─────────────
        scaling_group = QGroupBox("Scaling Type", self)
        scaling_layout = QVBoxLayout(scaling_group)
        scaling_layout.setSpacing(10)

        self.scaling_combo = QComboBox(self)
        self.scaling_combo.addItems([
            "ESPCN (Real-Time CNN)",
            "Nearest Neighbor",
            "Bilinear",
            "Bicubic"
        ])
        self.scaling_combo.setToolTip("Select the spatial upscaling algorithm")
        scaling_layout.addWidget(self.scaling_combo)

        # Scale factor row
        factor_row = QHBoxLayout()
        factor_row.setSpacing(8)

        factor_label = QLabel("Scale Factor", self)
        factor_label.setObjectName("sectionLabel")
        factor_row.addWidget(factor_label)
        factor_row.addStretch()

        self.factor_combo = QComboBox(self)
        self.factor_combo.addItems(["2×", "3×", "4×"])
        self.factor_combo.setMinimumWidth(80)
        self.factor_combo.setToolTip("Output resolution multiplier")
        self.factor_combo.currentIndexChanged.connect(self.on_scale_changed)
        factor_row.addWidget(self.factor_combo)

        scaling_layout.addLayout(factor_row)

        # Upscaling toggle
        self.upscale_check = QCheckBox("Enable Spatial Upscaling", self)
        self.upscale_check.setChecked(True)
        self.upscale_check.setToolTip("Toggle the AI upscaling pipeline on/off")
        self.upscale_check.stateChanged.connect(self.on_upscale_toggled)
        scaling_layout.addWidget(self.upscale_check)

        root.addWidget(scaling_group)

        # ───────────── Frame Generation Section ─────────────
        framegen_group = QGroupBox("Frame Generation", self)
        framegen_layout = QVBoxLayout(framegen_group)
        framegen_layout.setSpacing(10)

        self.framegen_combo = QComboBox(self)
        self.framegen_combo.addItems([
            "FlowNet (Optical Flow)",
            "Off"
        ])
        self.framegen_combo.setToolTip("Select the frame generation method")
        self.framegen_combo.currentIndexChanged.connect(self.on_framegen_combo_changed)
        framegen_layout.addWidget(self.framegen_combo)

        self.framegen_check = QCheckBox("Enable Frame Generation", self)
        self.framegen_check.setChecked(True)
        self.framegen_check.setToolTip("Toggle AI frame generation on/off")
        self.framegen_check.stateChanged.connect(self.on_framegen_toggled)
        framegen_layout.addWidget(self.framegen_check)

        root.addWidget(framegen_group)

        # ───────────── Capture Target Section ─────────────
        target_group = QGroupBox("Capture Target", self)
        target_layout = QVBoxLayout(target_group)
        target_layout.setSpacing(8)

        # Currently bound target display
        self.target_display = QLabel("No window selected", self)
        self.target_display.setObjectName("targetLabel")
        self.target_display.setAlignment(Qt.AlignCenter)
        target_layout.addWidget(self.target_display)

        # Window list
        self.window_list = QListWidget(self)
        self.window_list.setMinimumHeight(150)
        self.window_list.setMaximumHeight(200)
        self.window_list.setToolTip("Select a window to bind it as the capture target")
        self.window_list.itemClicked.connect(self.on_window_selected)
        self.window_list.itemDoubleClicked.connect(self.on_window_selected)
        target_layout.addWidget(self.window_list)

        # Target buttons row
        target_btn_row = QHBoxLayout()
        target_btn_row.setSpacing(8)

        self.refresh_btn = QPushButton("↻ Refresh", self)
        self.refresh_btn.setObjectName("refreshButton")
        self.refresh_btn.setToolTip("Refresh the list of open windows")
        self.refresh_btn.clicked.connect(self.populate_windows)
        target_btn_row.addWidget(self.refresh_btn)

        self.bind_btn = QPushButton("Bind Selected", self)
        self.bind_btn.setToolTip("Bind the selected window as capture target")
        self.bind_btn.clicked.connect(self.bind_selected_window)
        target_btn_row.addWidget(self.bind_btn)

        target_layout.addLayout(target_btn_row)

        root.addWidget(target_group)

        # ───────────── Scale / Stop Button ─────────────
        self.action_stack = QStackedWidget(self)

        self.scale_btn = QPushButton("▶  SCALE", self)
        self.scale_btn.setObjectName("scaleButton")
        self.scale_btn.setToolTip("Start upscaling the target window")
        self.scale_btn.clicked.connect(self.on_scale_start)

        self.stop_btn = QPushButton("■  STOP", self)
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.setToolTip("Stop the upscaling pipeline")
        self.stop_btn.clicked.connect(self.on_scale_stop)

        self.action_stack.addWidget(self.scale_btn)
        self.action_stack.addWidget(self.stop_btn)
        self.action_stack.setCurrentIndex(0)
        root.addWidget(self.action_stack)

        # ───────────── Telemetry Stats Bar ─────────────
        root.addWidget(make_separator())

        stats_row = QHBoxLayout()
        stats_row.setSpacing(0)

        # FPS stat
        fps_block = QVBoxLayout()
        fps_block.setSpacing(0)
        fps_block.setAlignment(Qt.AlignCenter)
        self.fps_value = QLabel("--", self)
        self.fps_value.setObjectName("statsValue")
        self.fps_value.setAlignment(Qt.AlignCenter)
        fps_unit = QLabel("FPS", self)
        fps_unit.setObjectName("statsUnit")
        fps_unit.setAlignment(Qt.AlignCenter)
        fps_block.addWidget(self.fps_value)
        fps_block.addWidget(fps_unit)
        stats_row.addLayout(fps_block)

        # Vertical divider
        stats_row.addSpacing(20)

        # Latency stat
        latency_block = QVBoxLayout()
        latency_block.setSpacing(0)
        latency_block.setAlignment(Qt.AlignCenter)
        self.latency_value = QLabel("--", self)
        self.latency_value.setObjectName("statsValue")
        self.latency_value.setAlignment(Qt.AlignCenter)
        latency_unit = QLabel("ms", self)
        latency_unit.setObjectName("statsUnit")
        latency_unit.setAlignment(Qt.AlignCenter)
        latency_block.addWidget(self.latency_value)
        latency_block.addWidget(latency_unit)
        stats_row.addLayout(latency_block)

        stats_row.addSpacing(20)

        # Status indicator
        status_block = QVBoxLayout()
        status_block.setSpacing(0)
        status_block.setAlignment(Qt.AlignCenter)
        self.status_label = QLabel("IDLE", self)
        self.status_label.setObjectName("statusInactive")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_desc = QLabel("STATUS", self)
        status_desc.setObjectName("statsUnit")
        status_desc.setAlignment(Qt.AlignCenter)
        status_block.addWidget(self.status_label)
        status_block.addWidget(status_desc)
        stats_row.addLayout(status_block)

        root.addLayout(stats_row)

        # Populate target windows initially
        self.populate_windows()

    # ───────────── Window Enumeration ─────────────

    def get_windows_list(self):
        """Retrieves names and HWND values of all active window targets on the Desktop."""
        windows = []

        def enum_win_proc(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd).strip()
                if title:
                    style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
                    ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
                    
                    if not (style & win32con.WS_CHILD):
                        is_tool = (ex_style & win32con.WS_EX_TOOLWINDOW) != 0
                        is_app = (ex_style & win32con.WS_EX_APPWINDOW) != 0
                        
                        if (style & win32con.WS_CAPTION) or is_app:
                            if not is_tool:
                                windows.append((hwnd, title))
            return True

        win32gui.EnumWindows(enum_win_proc, None)
        return windows

    def populate_windows(self):
        """Populates targets QListWidget filtering out invalid items."""
        selected_hwnd = None
        selected_item = self.window_list.currentItem()
        if selected_item:
            selected_hwnd = selected_item.data(Qt.UserRole)

        self.window_list.clear()

        for hwnd, title in self.get_windows_list():
            if hwnd == int(self.winId()):
                continue

            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, hwnd)
            self.window_list.addItem(item)

            if hwnd == selected_hwnd:
                self.window_list.setCurrentItem(item)

    # ───────────── Actions ─────────────

    def bind_selected_window(self):
        selected_item = self.window_list.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "No Selection", "Please select a target window to bind.")
            return

        hwnd = selected_item.data(Qt.UserRole)
        success = self.client.set_target_window(hwnd)
        if success:
            self.bound_hwnd = hwnd
            self.bound_title = selected_item.text()
            # Truncate long titles for display
            display_title = self.bound_title if len(self.bound_title) <= 40 else self.bound_title[:37] + "..."
            self.target_display.setText(f"🎯  {display_title}")
            self.target_display.setToolTip(f"HWND: 0x{hwnd:016X}\n{self.bound_title}")
        else:
            QMessageBox.critical(self, "Binding Error", "Failed to communicate window handle binding to C++ backend.")

    def on_window_selected(self, item):
        self.bind_selected_window()

    def on_scale_start(self):
        """Begin the upscaling pipeline by delaying, finding the active window, and capturing it."""
        # 5-second countdown timer
        self.countdown = 5
        self.scale_btn.setEnabled(False)
        self.scale_btn.setText(f"SCALING IN {self.countdown}s...")
        
        self.countdown_timer = QTimer(self)
        
        def handle_tick():
            self.countdown -= 1
            if self.countdown > 0:
                self.scale_btn.setText(f"SCALING IN {self.countdown}s...")
            else:
                self.countdown_timer.stop()
                self.scale_btn.setEnabled(True)
                self.scale_btn.setText("▶  SCALE")
                
                # Get the foreground/focused window (excluding our own window)
                fg_hwnd = win32gui.GetForegroundWindow()
                our_hwnd = int(self.winId())
                
                # If the foreground window is our own window or invalid, try to find the next window
                if fg_hwnd == our_hwnd or not fg_hwnd:
                    # Fallback to the bound window if we can't find another active window
                    fg_hwnd = self.bound_hwnd
                
                if not fg_hwnd:
                    QMessageBox.warning(self, "No Window Found", "Could not identify a focused window to scale.")
                    return
                
                # Bind the active foreground window
                title = win32gui.GetWindowText(fg_hwnd)
                success = self.client.set_target_window(fg_hwnd)
                if success:
                    self.bound_hwnd = fg_hwnd
                    self.bound_title = title
                    display_title = title if len(title) <= 40 else title[:37] + "..."
                    self.target_display.setText(f"🎯  {display_title}")
                    
                    # Send configuration to backend
                    scale = self.factor_combo.currentIndex() + 2
                    self.client.set_scale_factor(scale)
                    self.client.toggle_upscale(self.upscale_check.isChecked())
                    self.client.toggle_frame_gen(self.framegen_check.isChecked())

                    self.is_scaling = True
                    self.action_stack.setCurrentIndex(1)  # Show Stop button
                    self.status_label.setText("ACTIVE")
                    self.status_label.setObjectName("statusActive")
                    self.status_label.setStyleSheet("")  # Force re-style
                    self.status_label.style().unpolish(self.status_label)
                    self.status_label.style().polish(self.status_label)
                else:
                    QMessageBox.critical(self, "Binding Error", "Failed to bind to focused window.")
        
        self.countdown_timer.timeout.connect(handle_tick)
        self.countdown_timer.start(1000)

    def on_scale_stop(self):
        """Stop the upscaling pipeline."""
        self.is_scaling = False
        self.action_stack.setCurrentIndex(0)  # Show Scale button
        self.status_label.setText("IDLE")
        self.status_label.setObjectName("statusInactive")
        self.status_label.setStyleSheet("")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

        # Disable both pipelines and clear the target window on backend
        self.client.toggle_upscale(False)
        self.client.toggle_frame_gen(False)
        self.client.set_target_window(0)

    def on_scale_changed(self, index):
        scale = index + 2  # 0->2x, 1->3x, 2->4x
        self.client.set_scale_factor(scale)

    def on_upscale_toggled(self, state):
        enabled = (state == Qt.Checked.value)
        self.client.toggle_upscale(enabled)

    def on_framegen_toggled(self, state):
        enabled = (state == Qt.Checked.value)
        self.client.toggle_frame_gen(enabled)

    def on_framegen_combo_changed(self, index):
        """Handle framegen combo box selection. Index 0 = FlowNet ON, Index 1 = Off."""
        enabled = (index == 0)  # FlowNet is the first option
        self.framegen_check.setChecked(enabled)
        self.client.toggle_frame_gen(enabled)

    @Slot(float, float)
    def update_stats(self, fps, latency):
        self.fps_value.setText(f"{fps:.1f}")
        self.latency_value.setText(f"{latency:.1f}")

    def closeEvent(self, event):
        self.telemetry.stop()
        self.telemetry.wait()
        # Graceful shutdown
        self.client.request_shutdown()
        event.accept()
