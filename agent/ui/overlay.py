"""
agent/ui/overlay.py
-------------------
Minimal transparent overlay HUD window.

Features:
  - Frameless, fully transparent background
  - Always on top of all other windows
  - Click-through / mouse pass-through (WindowTransparentForInput)
  - Simple status text rendering with outline stroke for legibility
  - Does NOT appear in taskbar (Qt.WindowType.Tool)

Dependencies:
  PyQt6
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget


# ---------------------------------------------------------------------------
# Internal helper: label that renders text with a stroke outline so it stays
# readable on any background without requiring an opaque backdrop.
# ---------------------------------------------------------------------------

class _StrokedLabel(QLabel):
    """QLabel variant that draws an outline stroke around its text."""

    _STROKE_COLOR = QColor("#000000")
    _STROKE_WIDTH = 3

    def paintEvent(self, event) -> None:  # noqa: N802
        text = self.text()
        if not text:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        metrics = self.fontMetrics()
        tw = metrics.horizontalAdvance(text)
        th = metrics.ascent()
        x = (self.width() - tw) / 2
        y = (self.height() + th - metrics.descent()) / 2

        path = QPainterPath()
        path.addText(x, y, self.font(), text)

        # Stroke (outline)
        pen = QPen(self._STROKE_COLOR)
        pen.setWidth(self._STROKE_WIDTH)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        # Fill (text color from palette / stylesheet)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.palette().text())
        painter.drawPath(path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class OverlayHUD(QWidget):
    """
    Transparent, always-on-top, click-through overlay window.

    Usage::

        app = QApplication(sys.argv)
        hud = OverlayHUD()
        hud.show()
        hud.update_status("Hello, world!")
        app.exec()
    """

    # Default appearance
    _FONT_FAMILY: str = "Segoe UI"
    _FONT_SIZE: int = 22
    _TEXT_COLOR: str = "#FFFFFF"

    def __init__(self) -> None:
        super().__init__()
        self._init_window()
        self._init_ui()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_window(self) -> None:
        """Configure transparent, always-on-top, click-through window."""
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint        # No title bar / border
            | Qt.WindowType.WindowStaysOnTopHint     # Always on top
            | Qt.WindowType.Tool                     # Hidden from taskbar
            | Qt.WindowType.WindowTransparentForInput  # Clicks pass through
        )
        # Transparent background — no opaque surface behind widgets
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def _init_ui(self) -> None:
        """Build the minimal layout: a single centred status label."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._status_label = _StrokedLabel("")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setWordWrap(True)
        self._status_label.setFont(
            QFont(self._FONT_FAMILY, self._FONT_SIZE, QFont.Weight.Bold)
        )
        self._status_label.setStyleSheet(f"color: {self._TEXT_COLOR};")

        layout.addWidget(self._status_label)
        self.setLayout(layout)

        # Default geometry: full-width strip centred horizontally, near bottom
        self._fit_to_screen()

    def _fit_to_screen(self) -> None:
        """Size and position the overlay to cover the primary screen."""
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(
            screen.x(),
            screen.y(),
            screen.width(),
            screen.height(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_status(self, text: str) -> None:
        """
        Update the text displayed in the overlay.

        Args:
            text: The status string to display. Pass an empty string to
                  clear the overlay.
        """
        self._status_label.setText(text)

    def show(self) -> None:  # noqa: D102
        """Make the overlay visible."""
        super().show()
