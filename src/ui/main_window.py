import uuid
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from src.ui.async_bridge import AsyncBridge


class ThreadPanel(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(200)


class DocumentPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._drop_zone = QLabel("Drop files here to ingest")
        self._drop_zone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_zone.setMinimumHeight(48)
        self._drop_zone.setStyleSheet(
            "border: 2px dashed #666; border-radius: 6px; color: #888; padding: 8px;"
        )
        layout.addWidget(self._drop_zone)

        self._list = QListWidget()
        layout.addWidget(self._list)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

    def set_documents(self, documents: list[dict], threads: dict[str, str] | None = None):
        """Display all documents with their associated thread names.

        Args:
            documents: list of doc dicts from SQLiteStore.list_documents()
            threads: mapping of thread_id -> thread_title for resolving names
        """
        self._list.clear()
        threads = threads or {}
        for doc in documents:
            status = doc.get("status", "indexed")
            icon = {"indexed": "✓", "processing": "⟳", "failed": "✗"}.get(status, "?")
            name = doc.get("original_name") or Path(doc["source_path"]).name
            text = f"{icon} {name}"
            if status == "indexed":
                chunk_count = doc.get("chunk_count", "?")
                text += f"  [{chunk_count} chunks]"

            # Show which thread/chat this document belongs to
            thread_id = doc.get("thread_id")
            if thread_id and thread_id in threads:
                text += f"\n    ↳ used in: {threads[thread_id]}"
            elif not thread_id:
                text += "\n    ↳ global (all chats)"

            item = QListWidgetItem(text)
            if status == "failed":
                item.setForeground(Qt.GlobalColor.red)
            self._list.addItem(item)

    def show_progress(self, filename: str, percent: int):
        self._progress.setVisible(True)
        self._progress.setValue(percent)
        self._progress.setFormat(f"Ingesting {filename} — %p%")

    def hide_progress(self):
        self._progress.setVisible(False)

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
        self.files_dropped.emit([url.toLocalFile() for url in urls])

    files_dropped = Signal(list)


class ChatArea(QWidget):
    message_sent = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._display = QTextBrowser()
        layout.addWidget(self._display)

        input_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask a question...")
        self._input.returnPressed.connect(self._send)
        input_row.addWidget(self._input)

        self._send_btn = QPushButton("Send")
        self._send_btn.clicked.connect(self._send)
        input_row.addWidget(self._send_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setVisible(False)
        input_row.addWidget(self._stop_btn)

        layout.addLayout(input_row)

    def _send(self):
        text = self._input.text().strip()
        if text:
            self._input.clear()
            self.message_sent.emit(text)

    def append_user_message(self, text: str):
        self._display.append(f"<b>You:</b> {text}")
        self._display.append("")

    def start_assistant_message(self):
        self._display.append("<b>Assistant:</b> ")
        self._send_btn.setEnabled(False)
        self._stop_btn.setVisible(True)

    @Slot(str)
    def append_token(self, token: str):
        cursor = self._display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(token)
        self._display.setTextCursor(cursor)
        self._display.ensureCursorVisible()

    def finish_assistant_message(self):
        self._display.append("")
        self._send_btn.setEnabled(True)
        self._stop_btn.setVisible(False)

    def show_error(self, error: str):
        self._display.append(f"<i style='color: #e55;'>[Error: {error}]</i>")
        self._send_btn.setEnabled(True)
        self._stop_btn.setVisible(False)

    def clear_display(self):
        self._display.clear()

    def load_messages(self, messages: list[dict]):
        self._display.clear()
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "human":
                self._display.append(f"<b>You:</b> {content}")
            else:
                self._display.append(f"<b>Assistant:</b> {content}")
            self._display.append("")


class MainWindow(QMainWindow):
    def __init__(self, bridge: AsyncBridge, engine, sqlite_store, pipeline, watcher, config: dict):
        super().__init__()
        self._bridge = bridge
        self._engine = engine
        self._sqlite = sqlite_store
        self._pipeline = pipeline
        self._watcher = watcher
        self._config = config
        self._current_thread_id: str | None = None

        self.setWindowTitle("RAG System")
        self.setMinimumSize(900, 600)
        self.resize(1100, 700)

        self._setup_menubar()
        self._setup_toolbar()
        self._setup_central()
        self._connect_signals()
        self._load_threads()
        self._refresh_documents()

    def _setup_menubar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction("New Thread", self._new_thread)
        file_menu.addAction("Import Documents...", self._import_documents)
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)

        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction("Settings...")

        view_menu = menubar.addMenu("&View")
        view_menu.addAction("Toggle Sidebar", self._toggle_sidebar)

    def _setup_toolbar(self):
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        toolbar.addAction(QAction("New Thread", self, triggered=self._new_thread))
        toolbar.addAction(QAction("Reindex", self, triggered=self._reindex))
        toolbar.addSeparator()
        toolbar.addAction(QAction("Delete Thread", self, triggered=self._delete_thread))

    def _setup_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left sidebar: threads over documents
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        self._thread_panel = ThreadPanel()
        v_splitter.addWidget(self._thread_panel)
        self._doc_panel = DocumentPanel()
        v_splitter.addWidget(self._doc_panel)
        v_splitter.setSizes([200, 300])
        v_splitter.setMinimumWidth(220)
        self._sidebar_splitter = v_splitter

        h_splitter.addWidget(v_splitter)

        # Chat area
        self._chat = ChatArea()
        h_splitter.addWidget(self._chat)

        h_splitter.setSizes([280, 600])
        h_splitter.setStretchFactor(0, 0)
        h_splitter.setStretchFactor(1, 1)
        self._h_splitter = h_splitter

        layout.addWidget(h_splitter)

    def _connect_signals(self):
        self._chat.message_sent.connect(self._on_message_sent)
        self._chat._stop_btn.clicked.connect(self._bridge.cancel_stream)
        self._bridge.token_received.connect(self._chat.append_token)
        self._bridge.stream_finished.connect(self._on_stream_finished)
        self._bridge.stream_error.connect(self._chat.show_error)
        self._thread_panel.currentRowChanged.connect(self._on_thread_selected)
        self._doc_panel.files_dropped.connect(self._on_files_dropped)

    # --- Thread management ---

    def _load_threads(self):
        threads = self._sqlite.list_threads()
        self._thread_panel.clear()
        for t in threads:
            item = QListWidgetItem(t["title"])
            item.setData(Qt.ItemDataRole.UserRole, t["id"])
            self._thread_panel.addItem(item)
        if not threads:
            self._new_thread()

    def _new_thread(self):
        thread_id = str(uuid.uuid4())
        self._sqlite.create_thread(thread_id, "New chat")
        self._load_threads()
        self._thread_panel.setCurrentRow(0)

    def _delete_thread(self):
        item = self._thread_panel.currentItem()
        if not item:
            return
        thread_id = item.data(Qt.ItemDataRole.UserRole)
        self._sqlite.delete_thread(thread_id)
        self._load_threads()

    @Slot(int)
    def _on_thread_selected(self, row: int):
        if row < 0:
            return
        item = self._thread_panel.item(row)
        if not item:
            return
        thread_id = item.data(Qt.ItemDataRole.UserRole)
        if thread_id == self._current_thread_id:
            return
        self._current_thread_id = thread_id
        self._engine.clear_history()
        messages = self._sqlite.get_thread_messages(thread_id)
        self._chat.load_messages(messages)
        self._refresh_documents()

    # --- Chat ---

    @Slot(str)
    def _on_message_sent(self, text: str):
        if not self._current_thread_id:
            self._new_thread()

        msg_id = str(uuid.uuid4())
        self._sqlite.add_thread_message(msg_id, self._current_thread_id, "human", text)
        self._chat.append_user_message(text)
        self._chat.start_assistant_message()

        self._collected_response = []
        self._bridge.token_received.connect(self._collect_token)

        async def _do_query():
            stream, sources, strategy = await self._engine.aquery_stream(
                text, thread_id=self._current_thread_id
            )
            async for token in stream:
                self._bridge.token_received.emit(token)
            self._bridge.stream_finished.emit()

        self._bridge._current_future = self._bridge.submit(_do_query())

    @Slot(str)
    def _collect_token(self, token: str):
        self._collected_response.append(token)

    @Slot()
    def _on_stream_finished(self):
        self._chat.finish_assistant_message()
        self._bridge.token_received.disconnect(self._collect_token)
        full_response = "".join(self._collected_response)
        if full_response and self._current_thread_id:
            msg_id = str(uuid.uuid4())
            self._sqlite.add_thread_message(msg_id, self._current_thread_id, "ai", full_response)
        self._collected_response = []

    # --- Documents ---

    def _refresh_documents(self):
        docs = self._sqlite.list_documents()
        threads = self._sqlite.list_threads()
        thread_map = {t["id"]: t["title"] for t in threads}
        self._doc_panel.set_documents(docs, thread_map)

    @Slot(list)
    def _on_files_dropped(self, paths: list[str]):
        for path in paths:
            self._bridge.run_blocking(
                self._pipeline.ingest,
                Path(path),
                False,
                Path(path).name,
                self._current_thread_id,
            )
        self._refresh_documents()

    def _import_documents(self):
        from PySide6.QtWidgets import QFileDialog
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import Documents", "", "Documents (*.pdf *.docx *.txt *.md)"
        )
        if paths:
            self._on_files_dropped(paths)

    def _reindex(self):
        docs = self._sqlite.list_documents()
        for doc in docs:
            self._bridge.run_blocking(
                self._pipeline.ingest,
                Path(doc["source_path"]),
                True,
            )
        self._refresh_documents()

    def _toggle_sidebar(self):
        visible = self._sidebar_splitter.isVisible()
        self._sidebar_splitter.setVisible(not visible)

    def closeEvent(self, event):
        if self._watcher:
            self._watcher.stop()
        self._bridge.shutdown()
        self._sqlite.close()
        event.accept()
