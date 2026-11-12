"""Behavior checks for the Python side of the web image gallery."""

import pytest
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication

from gui_easyrob.tabs import images


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


class _OnlySelectedImage:
    def include_image(self, path, model):
        return model == "RF" and path.name.startswith("RF_")


def test_gallery_state_groups_and_filters_images(tmp_path):
    for folder, names in (
        ("CURATE", ("overview.png", "ignore.txt")),
        ("PREDICT", ("RF_plot.png", "GB_plot.png")),
    ):
        output = tmp_path / folder
        output.mkdir()
        for name in names:
            (output / name).write_bytes(b"image")

    state, paths = images.build_gallery_state(tmp_path, _OnlySelectedImage(), "RF")

    assert [stage["label"] for stage in state["stages"]] == ["PREDICT"]
    assert [item["name"] for item in state["stages"][0]["images"]] == ["RF_plot.png"]
    assert state["total"] == 1
    image_id = state["stages"][0]["images"][0]["id"]
    assert paths[image_id] == tmp_path / "PREDICT" / "RF_plot.png"
    assert state["stages"][0]["images"][0]["url"] == QUrl.fromLocalFile(str(paths[image_id])).toString()


def test_gallery_bridge_opens_only_a_listed_image(tmp_path, monkeypatch):
    selected = tmp_path / "selected.png"
    selected.write_bytes(b"image")
    opened = []
    monkeypatch.setattr(images.QDesktopServices, "openUrl", lambda url: opened.append(url) or True)
    bridge = images.ImageActionBridge()
    bridge.set_paths({"image-0": selected})

    bridge.performAction("image-0", "open")
    bridge.performAction("../selected.png", "open")
    bridge.performAction("image-0", "unknown")

    assert opened == [QUrl.fromLocalFile(str(selected))]


def test_gallery_bridge_copies_and_saves_the_selected_image(tmp_path, qapp, monkeypatch):
    image_path = tmp_path / "source.png"
    target_path = tmp_path / "saved.png"
    from PySide6.QtGui import QImage

    image = QImage(2, 2, QImage.Format.Format_RGB32)
    image.fill(0x336699)
    assert image.save(str(image_path))
    monkeypatch.setattr(
        images.QFileDialog,
        "getSaveFileName",
        lambda *args: (str(target_path), "Images (*.png *.jpg *.jpeg)"),
    )
    bridge = images.ImageActionBridge()
    bridge.set_paths({"image-0": image_path})

    bridge.performAction("image-0", "copy")
    assert not qapp.clipboard().pixmap().isNull()
    bridge.performAction("image-0", "save")
    assert target_path.is_file()
