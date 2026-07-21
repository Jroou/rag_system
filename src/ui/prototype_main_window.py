"""
PROTOTYPE — throwaway code answering: "What should the PySide6 main window look like?"

Three variants switchable via combo box at bottom of window:
  A — Classic IDE: left sidebar (threads stacked over docs), right is chat
  B — Three-column: threads | chat | documents, each its own column
  C — Compact drawer: chat fills window, sidebar slides in/out via toggle

Run: uv run python src/ui/prototype_main_window.py
"""

import sys
from enum import Enum

from PySide6.QtCore import Qt, QMimeData, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenuBar,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QFrame,
    QSizePolicy,
)


# --- Domain stubs (fake data for visual prototype) ---

class DocStatus(Enum):
    READY = "ready"
    PROCESSING = "processing"
    FAILED = "failed"


FAKE_THREADS = [
    "How does chunking work?",
    "Explain hybrid retrieval",
    "Debug: empty results on PDF",
    "Compare reranker models",
    "Ingestion pipeline overview",
]

FAKE_DOCUMENTS = [
    {"name": "architecture.pdf", "status": DocStatus.READY, "chunks": 47, "chats": 3},
    {"name": "meeting_notes.docx", "status": DocStatus.PROCESSING, "chunks": 0, "chats": 0, "progress": 65},
    {"name": "api_spec.pdf", "status": DocStatus.READY, "chunks": 112, "chats": 1},
    {"name": "broken_file.txt", "status": DocStatus.FAILED, "chunks": 0, "chats": 0},
    {"name": "research_paper.pdf", "status": DocStatus.READY, "chunks": 83, "chats": 5},
]


# --- Reusable building blocks ---

class ThreadList(QListWidget):
    """Thread history panel."""

    def __init__(self):
        super().__init__()
        for t in FAKE_THREADS:
            self.addItem(t)
        self.setCurrentRow(0)


class DocumentPanel(QWidget):
    """Document panel with drop-zone, file list, status indicators."""

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Drop zone
        self._drop_zone = QLabel("Drop files here to ingest")
        self._drop_zone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_zone.setMinimumHeight(48)
        self._drop_zone.setStyleSheet(
            "border: 2px dashed #666; border-radius: 6px; color: #888; padding: 8px;"
        )
        layout.addWidget(self._drop_zone)

        # File list
        self._list = QListWidget()
        for doc in FAKE_DOCUMENTS:
            item = QListWidgetItem()
            item.setText(self._format_doc(doc))
            if doc["status"] == DocStatus.FAILED:
                item.setForeground(Qt.GlobalColor.red)
            elif doc["status"] == DocStatus.PROCESSING:
                item.setForeground(Qt.GlobalColor.yellow)
            self._list.addItem(item)
        layout.addWidget(self._list)

        # Progress bar (visible when processing)
        self._progress = QProgressBar()
        self._progress.setValue(65)
        self._progress.setFormat("Processing meeting_notes.docx — %p%")
        layout.addWidget(self._progress)

    def _format_doc(self, doc: dict) -> str:
        status_icon = {"ready": "✓", "processing": "⟳", "failed": "✗"}[doc["status"].value]
        line = f"{status_icon} {doc['name']}"
        if doc["status"] == DocStatus.READY:
            line += f"  [{doc['chunks']} chunks]"
            if doc["chats"] > 0:
                line += f"  used in {doc['chats']} chats"
        return line

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._drop_zone.setStyleSheet(
                "border: 2px solid #4a9; border-radius: 6px; color: #4a9; padding: 8px; background: #1a3a2a;"
            )

    def dragLeaveEvent(self, event):
        self._drop_zone.setStyleSheet(
            "border: 2px dashed #666; border-radius: 6px; color: #888; padding: 8px;"
        )

    def dropEvent(self, event: QDropEvent):
        self._drop_zone.setStyleSheet(
            "border: 2px dashed #666; border-radius: 6px; color: #888; padding: 8px;"
        )
        urls = event.mimeData().urls()
        for url in urls:
            item = QListWidgetItem(f"⟳ {url.toLocalFile().split('/')[-1]}  [ingesting...]")
            item.setForeground(Qt.GlobalColor.yellow)
            self._list.addItem(item)


class ChatArea(QWidget):
    """Chat display + input."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._display = QTextBrowser()
        self._display.setPlaceholderText("Chat messages will appear here...")
        self._display.append("<b>You:</b> How does the chunking strategy work?")
        self._display.append("")
        self._display.append(
            "<b>Assistant:</b> Documents are split into parent chunks (~1000 tokens) "
            "and child chunks (~200 tokens). Retrieval matches child chunks, then "
            "parent text is injected into the generation prompt for fuller context..."
        )
        layout.addWidget(self._display)

        # Input row
        input_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask a question...")
        self._input.returnPressed.connect(self._send)
        input_row.addWidget(self._input)

        self._send_btn = QPushButton("Send")
        self._send_btn.clicked.connect(self._send)
        input_row.addWidget(self._send_btn)

        layout.addLayout(input_row)

    def _send(self):
        text = self._input.text().strip()
        if text:
            self._display.append(f"\n<b>You:</b> {text}")
            self._display.append("<i>Thinking...</i>")
            self._input.clear()


def _create_toolbar(window: QMainWindow) -> QToolBar:
    toolbar = QToolBar("Main Toolbar")
    toolbar.setMovable(False)
    window.addToolBar(toolbar)

    toolbar.addAction(QAction("⚙ Settings", window))
    toolbar.addAction(QAction("🔄 Reindex", window))
    toolbar.addAction(QAction("📄 New Thread", window))
    toolbar.addSeparator()
    toolbar.addAction(QAction("🗑 Delete Thread", window))
    return toolbar


def _create_menubar(window: QMainWindow) -> QMenuBar:
    menubar = window.menuBar()
    file_menu = menubar.addMenu("&File")
    file_menu.addAction("New Thread")
    file_menu.addAction("Import Documents...")
    file_menu.addSeparator()
    file_menu.addAction("Quit")

    edit_menu = menubar.addMenu("&Edit")
    edit_menu.addAction("Settings...")

    view_menu = menubar.addMenu("&View")
    view_menu.addAction("Toggle Sidebar")
    return menubar


# --- Variant A: Classic IDE layout ---

class VariantA(QWidget):
    """Left sidebar (threads over docs) | Right chat area. Splitters everywhere."""

    NAME = "A — Classic IDE"

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Main horizontal splitter: sidebar | chat
        h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: vertical splitter (threads | docs)
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        threads = ThreadList()
        threads.setMinimumWidth(200)
        v_splitter.addWidget(threads)
        v_splitter.addWidget(DocumentPanel())
        v_splitter.setSizes([200, 300])
        v_splitter.setMinimumWidth(220)

        h_splitter.addWidget(v_splitter)
        h_splitter.addWidget(ChatArea())
        h_splitter.setSizes([280, 600])
        h_splitter.setStretchFactor(0, 0)
        h_splitter.setStretchFactor(1, 1)

        layout.addWidget(h_splitter)


# --- Variant B: Three-column layout ---

class VariantB(QWidget):
    """Three columns: threads | chat | documents. Each independently resizable."""

    NAME = "B — Three Column"

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left column: threads
        threads = ThreadList()
        threads.setMinimumWidth(150)
        splitter.addWidget(threads)

        # Center: chat
        chat = ChatArea()
        chat.setMinimumWidth(300)
        splitter.addWidget(chat)

        # Right column: documents
        docs = DocumentPanel()
        docs.setMinimumWidth(200)
        splitter.addWidget(docs)

        splitter.setSizes([200, 500, 250])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        layout.addWidget(splitter)


# --- Variant C: Compact drawer layout ---

class VariantC(QWidget):
    """Chat fills window. Left sidebar slides in/out via toggle button."""

    NAME = "C — Compact Drawer"

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Sidebar (starts hidden)
        self._sidebar = QWidget()
        self._sidebar.setFixedWidth(280)
        self._sidebar.setVisible(False)
        sidebar_layout = QVBoxLayout(self._sidebar)
        sidebar_layout.setContentsMargins(4, 4, 4, 4)

        # Threads section
        threads_label = QLabel("<b>Threads</b>")
        sidebar_layout.addWidget(threads_label)
        sidebar_layout.addWidget(ThreadList())

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sidebar_layout.addWidget(sep)

        # Documents section
        docs_label = QLabel("<b>Documents</b>")
        sidebar_layout.addWidget(docs_label)
        sidebar_layout.addWidget(DocumentPanel())

        layout.addWidget(self._sidebar)

        # Main area: toggle button + chat
        main_area = QVBoxLayout()
        main_area.setContentsMargins(0, 0, 0, 0)

        # Toggle row
        toggle_row = QHBoxLayout()
        self._toggle_btn = QPushButton("☰ Sidebar")
        self._toggle_btn.setFixedWidth(100)
        self._toggle_btn.clicked.connect(self._toggle_sidebar)
        toggle_row.addWidget(self._toggle_btn)
        toggle_row.addStretch()
        main_area.addLayout(toggle_row)

        main_area.addWidget(ChatArea())
        layout.addLayout(main_area)

    def _toggle_sidebar(self):
        visible = self._sidebar.isVisible()
        self._sidebar.setVisible(not visible)
        self._toggle_btn.setText("✕ Close" if not visible else "☰ Sidebar")


# --- Main window with variant switcher ---

VARIANTS = [VariantA, VariantB, VariantC]


class PrototypeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG System — Layout Prototype (THROWAWAY)")
        self.setMinimumSize(900, 600)
        self.resize(1100, 700)

        _create_menubar(self)
        _create_toolbar(self)

        # Central area: variant + switcher at bottom
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Variant container
        self._variant_container = QVBoxLayout()
        main_layout.addLayout(self._variant_container, stretch=1)

        # Switcher bar at bottom
        switcher = QHBoxLayout()
        switcher.addStretch()
        lbl = QLabel("Variant:")
        switcher.addWidget(lbl)
        self._combo = QComboBox()
        for v in VARIANTS:
            self._combo.addItem(v.NAME)
        self._combo.currentIndexChanged.connect(self._switch_variant)
        switcher.addWidget(self._combo)

        prev_btn = QPushButton("← Prev")
        prev_btn.clicked.connect(lambda: self._combo.setCurrentIndex(
            (self._combo.currentIndex() - 1) % len(VARIANTS)
        ))
        switcher.addWidget(prev_btn)

        next_btn = QPushButton("Next →")
        next_btn.clicked.connect(lambda: self._combo.setCurrentIndex(
            (self._combo.currentIndex() + 1) % len(VARIANTS)
        ))
        switcher.addWidget(next_btn)
        switcher.addStretch()

        switcher_frame = QFrame()
        switcher_frame.setLayout(switcher)
        switcher_frame.setStyleSheet("background: #2a2a2a; padding: 4px;")
        switcher_frame.setFixedHeight(40)
        main_layout.addWidget(switcher_frame)

        self._current_widget = None
        self._switch_variant(0)

    def _switch_variant(self, index: int):
        if self._current_widget:
            self._variant_container.removeWidget(self._current_widget)
            self._current_widget.deleteLater()

        self._current_widget = VARIANTS[index]()
        self._variant_container.addWidget(self._current_widget)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette for better visual prototype
    from PySide6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base, QColor(40, 40, 40))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(50, 50, 50))
    palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Button, QColor(50, 50, 50))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(70, 130, 180))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = PrototypeWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
