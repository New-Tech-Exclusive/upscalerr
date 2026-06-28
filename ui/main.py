import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
from main_window import MainWindow


def setup_resources():
    """
    Creates basic asset directories and fallback files for QSS theme alignment.
    Guarantees that QSS check indicator image requests don't crash or throw exceptions.
    """
    icon_dir = os.path.join(os.path.dirname(__file__), "resources", "icons")
    os.makedirs(icon_dir, exist_ok=True)

    svg_path = os.path.join(icon_dir, "check.svg")
    if not os.path.exists(svg_path):
        svg_content = """<?xml version="1.0" encoding="utf-8"?>
<svg version="1.1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
    <path fill="#ffffff" d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
</svg>"""
        with open(svg_path, "w") as f:
            f.write(svg_content)


def main():
    setup_resources()

    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion as base theme for clean dark styling
    app.setApplicationName("Upscalerr")
    app.setApplicationVersion("0.1")

    # Set default application font
    font = QFont("Segoe UI", 10)
    font.setHintingPreference(QFont.PreferFullHinting)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
