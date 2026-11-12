"""Code-generated icons for the EasyROB bot UI."""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap

__all__ = ["create_bot_icon"]


def create_bot_icon(size: int = 18) -> QIcon:
    icon_size = max(int(size or 18), 12)
    pixmap = QPixmap(icon_size, icon_size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    stroke = QColor("#1f2937")
    fill = QColor("#d7eef8")
    accent = QColor("#ff8a3d")
    eye = QColor("#0f172a")

    head_rect = pixmap.rect().adjusted(3, 4, -3, -3)
    painter.setPen(QPen(stroke, 1.4))
    painter.setBrush(fill)
    painter.drawRoundedRect(head_rect, 3.2, 3.2)

    painter.drawLine(
        QPointF(icon_size * 0.5, 1.7),
        QPointF(icon_size * 0.5, 4.5),
    )
    painter.setBrush(accent)
    painter.drawEllipse(QPointF(icon_size * 0.5, 1.9), 1.35, 1.35)

    painter.setBrush(eye)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(QPointF(icon_size * 0.37, icon_size * 0.48), 1.2, 1.2)
    painter.drawEllipse(QPointF(icon_size * 0.63, icon_size * 0.48), 1.2, 1.2)

    painter.setPen(QPen(stroke, 1.1))
    painter.drawLine(
        QPointF(icon_size * 0.37, icon_size * 0.68),
        QPointF(icon_size * 0.63, icon_size * 0.68),
    )
    painter.drawLine(
        QPointF(3.0, icon_size * 0.46),
        QPointF(1.4, icon_size * 0.38),
    )
    painter.drawLine(
        QPointF(icon_size - 3.0, icon_size * 0.46),
        QPointF(icon_size - 1.4, icon_size * 0.38),
    )

    painter.end()
    return QIcon(pixmap)
