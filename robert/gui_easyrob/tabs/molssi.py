"""
MolSSI databases tab for easyROB.

This module embeds the MolSSI descriptor database inside the GUI and provides
a complete download and post-processing pipeline.

Responsibilities:
- Host the MolSSI web interface (QWebEngineView)
- Intercept and manage file downloads
- Handle user-defined save locations
- Convert downloaded Excel files to CSV
- Integrate downloaded datasets into the easyROB workflow

Import strategy:
- Supports dual execution modes:
  1. Local (portable execution)
  2. Installed package (environment / entry point)

- Imports use a try/except fallback for compatibility.

Notes:
- Combines UI interaction (web view) with background processing (workers).
- Designed to keep the main window decoupled from download logic.

"""

try:
    from utils.utils_gui import (
        Path,
        QFileDialog,
        QHBoxLayout,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QTimer,
        QUrl,
        QVBoxLayout,
        QWebEngineDownloadRequest,
        QWebEngineView,
        QWidget,
        Qt,
        Signal,
    )
    from utils.molssi_utils import ExcelToCSVWorker

except ImportError:
    from robert.gui_easyrob.utils.utils_gui import (
        Path,
        QFileDialog,
        QHBoxLayout,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QTimer,
        QUrl,
        QVBoxLayout,
        QWebEngineDownloadRequest,
        QWebEngineView,
        QWidget,
        Qt,
        Signal,
    )
    from robert.gui_easyrob.utils.molssi_utils import ExcelToCSVWorker

# ---- Standard library ----
import os


class MolSSIDatabasesTab(QWidget):
    """
    Tab widget embedding the MolSSI descriptor databases web interface.

    This widget:
    - Hosts a QWebEngineView pointing to the MolSSI databases site
    - Intercepts downloads initiated from the web view
    - Allows saving descriptor files locally
    - Optionally converts downloaded Excel files to CSV
    """

    # Signal emitted when a test file download is requested
    load_test_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Context used to determine post-download behaviour
        self.current_download_context = "active"

        # Worker used for Excel -> CSV conversion
        self._excel_worker = None

        # State used during downloads
        self._pending_popup = None
        self._pending_path = None

        layout = QVBoxLayout(self)

        # --------------------------------------------------
        # WebView subclass: force single-window navigation
        # --------------------------------------------------
        class SingleWindowWebView(QWebEngineView):
            """
            Custom QWebEngineView that prevents opening external windows.
            Any request to open a new window is redirected to the same view.
            """

            def createWindow(self, webWindowType):
                tmp = QWebEngineView(self)
                tmp.setAttribute(Qt.WA_DeleteOnClose, True)
                tmp.urlChanged.connect(
                    lambda url: (self.setUrl(url), tmp.deleteLater())
                )
                return tmp

        # Base URL for MolSSI databases
        self.databases_home_url = QUrl("https://descriptor-libraries.molssi.org/")

        # --------------------------------------------------
        # Home bar
        # --------------------------------------------------
        home_bar = QWidget()
        home_bar.setContentsMargins(0, 0, 0, 0)
        home_bar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        home_layout = QHBoxLayout(home_bar)
        home_layout.setContentsMargins(0, 0, 0, 0)
        home_layout.setSpacing(2)

        home_button = QPushButton("🏠 MolSSI Databases")
        home_button.setToolTip("Return to the Databases start page")
        home_button.setCursor(Qt.PointingHandCursor)
        home_button.setFixedSize(150, 24)
        home_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        home_button.clicked.connect(
            lambda: self.web_view.setUrl(self.databases_home_url)
        )

        home_layout.addWidget(home_button)
        home_layout.addStretch()

        layout.addWidget(home_bar, 0, Qt.AlignLeft)

        # --------------------------------------------------
        # Web view
        # --------------------------------------------------
        self.web_view = SingleWindowWebView()
        self.web_view.setUrl(self.databases_home_url)
        layout.addWidget(self.web_view, 1)

        # --------------------------------------------------
        # Download handling
        # --------------------------------------------------
        profile = self.web_view.page().profile()
        profile.downloadRequested.connect(self._handle_download)

    # ==================================================
    # PUBLIC API
    # ==================================================
    def handle_external_download(self, path, context="active"):
        """
        Entry point used by EasyROB when a file has already been downloaded
        externally and should follow the same post-download pipeline.
        """
        self.current_download_context = context
        self._on_download_completed(path)

    # ==================================================
    # DOWNLOAD HANDLING
    # ==================================================
    def _handle_download(self, req: QWebEngineDownloadRequest):
        """
        Intercepts download requests coming from the embedded web view.
        Prompts the user for a save location and tracks download completion.
        """

        def open_dialog():
            suggested = (
                req.downloadFileName() or QUrl(req.url()).fileName() or "download"
            )

            path, _ = QFileDialog.getSaveFileName(self, "Save File", suggested)

            if not path:
                req.cancel()
                return

            req.setDownloadDirectory(os.path.dirname(path))
            req.setDownloadFileName(os.path.basename(path))
            req.accept()

            self._pending_path = path
            self._pending_popup = self._show_download_popup()

            req.stateChanged.connect(self._on_download_state_changed)

        # Ensure dialog is opened outside the WebEngine callback stack
        QTimer.singleShot(0, open_dialog)

    def _on_download_state_changed(self, state):
        """
        Reacts to state changes of a QWebEngine download.
        When completed, triggers the post-download pipeline.
        """
        if state == QWebEngineDownloadRequest.DownloadCompleted:
            if self._pending_popup:
                self._pending_popup.close()
                self._pending_popup.deleteLater()
                self._pending_popup = None

            if self._pending_path:
                self._on_download_completed(self._pending_path)
                self._pending_path = None

    # ==================================================
    # POST-DOWNLOAD PIPELINE
    # ==================================================
    def _on_download_completed(self, path):
        """
        Called once a file has been fully downloaded.
        Handles user confirmation and optional Excel-to-CSV conversion.
        """

        # Consume context and reset to default
        ctx = self.current_download_context
        self.current_download_context = "active"

        if ctx != "molssi_test":
            reply = QMessageBox.question(
                self,
                "Convert Excel to CSV",
                "The Excel file has been downloaded successfully.\n\n"
                "This file contains chemical descriptors and molecular identifiers.\n"
                "Converting it to CSV will create a simpler, universal file format that can be loaded in easyROB later.\n\n"
                "Note: If you plan to build predictive models, you will need to include a target "
                "(for example, an experimental property or value you want to predict).\n\n"
                "Do you want to convert this Excel file to CSV now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )

            if reply != QMessageBox.Yes:
                return

        self._convert_excel(path, ctx)

    # ==================================================
    # CONVERSION WORKER
    # ==================================================
    def _convert_excel(self, path, ctx):
        """
        Launches a background worker that converts an Excel file to CSV.
        Displays a non-blocking progress popup during conversion.
        """
        popup = QMessageBox(self)
        popup.setWindowTitle("Converting file")
        popup.setText("Converting Excel to CSV…\n\nPlease wait.")
        popup.setStandardButtons(QMessageBox.NoButton)
        popup.setModal(False)
        popup.show()

        worker = ExcelToCSVWorker(path)
        self._excel_worker = worker

        def finished(csv_path):
            popup.close()
            popup.deleteLater()

            if ctx == "molssi_test":
                self.load_test_molssi(csv_path, source=path)
                return

            QMessageBox.information(
                self, "Conversion completed", "Excel converted to CSV successfully."
            )

        def error(msg):
            popup.close()
            popup.deleteLater()
            QMessageBox.warning(self, "Conversion failed", msg)

        worker.finished.connect(finished)
        worker.error.connect(error)
        worker.start()

    # ==================================================
    # UI HELPERS
    # ==================================================
    def _show_download_popup(self):
        """
        Displays a non-modal popup informing the user that a download
        is in progress and continues in the background.
        """
        popup = QMessageBox(self)
        popup.setWindowTitle("Downloading file")
        popup.setText(
            "Downloading file…\n\n"
            "You can close this window.\n"
            "The download will continue in the background."
        )
        popup.setStandardButtons(QMessageBox.Ok)
        popup.setModal(False)
        popup.show()
        return popup

    def load_test_molssi(self, csv_path, source=None):
        """
        Finalizes a MolSSI test dataset.
        """

        # Ask main window to load this CSV as TEST
        self.load_test_requested.emit(csv_path)

        # -------------------------------------------------
        # Inform user
        # -------------------------------------------------
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("MolSSI test dataset loaded")
        msg.setText(
            "The MolSSI database has been successfully loaded as a test dataset.\n\n"
        )
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

        # -------------------
        # Cleanup raw Excel
        # -------------------
        if source:
            try:
                Path(source).unlink()
            except Exception:
                pass
