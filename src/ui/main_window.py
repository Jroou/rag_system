import threading
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
    QMenu,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from src.ui.async_bridge import AsyncBridge


class _IngestCancelled(Exception):
    pass


class ThreadPanel(QListWidget):
    thread_renamed = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.itemChanged.connect(self._on_item_changed)
        self._editing_item: QListWidgetItem | None = None

    def _show_context_menu(self, pos):
        item = self.itemAt(pos)
        if not item:
            return
        menu = QMenu(self)
        menu.addAction("Rename", lambda: self._start_rename(item))
        menu.exec(self.mapToGlobal(pos))

    def _start_rename(self, item: QListWidgetItem):
        self._editing_item = item
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.setCurrentItem(item)
        self.openPersistentEditor(item)

    def _on_item_changed(self, item: QListWidgetItem):
        if item is not self._editing_item:
            return
        self._editing_item = None
        self.closePersistentEditor(item)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        thread_id = item.data(Qt.ItemDataRole.UserRole)
        if thread_id:
            self.thread_renamed.emit(thread_id, item.text())


class DocumentPanel(QWidget):
    document_delete = Signal(str)
    document_promote = Signal(str)

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
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_doc_menu)
        layout.addWidget(self._list)

        progress_row = QHBoxLayout()
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        progress_row.addWidget(self._progress)
        self._cancel_btn = QPushButton("✕")
        self._cancel_btn.setFixedWidth(28)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.setToolTip("Cancel ingestion")
        progress_row.addWidget(self._cancel_btn)
        layout.addLayout(progress_row)

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

            thread_id = doc.get("thread_id")
            if thread_id and thread_id in threads:
                text += f"\n    ↳ used in: {threads[thread_id]}"
            elif not thread_id:
                text += "\n    ↳ global (all chats)"

            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, doc.get("id"))
            if status in ("failed", "error"):
                item.setForeground(Qt.GlobalColor.red)
                error_msg = doc.get("error_message", "Unknown error")
                item.setToolTip(f"Failed: {error_msg}")
            self._list.addItem(item)

    def get_checked_document_ids(self) -> list[str]:
        """Return document IDs of all checked items."""
        ids = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item and item.checkState() == Qt.CheckState.Checked:
                doc_id = item.data(Qt.ItemDataRole.UserRole)
                if doc_id:
                    ids.append(doc_id)
        return ids

    def _show_doc_menu(self, pos):
        item = self._list.itemAt(pos)
        if not item:
            return
        doc_id = item.data(Qt.ItemDataRole.UserRole)
        if not doc_id:
            return

        menu = QMenu(self)
        menu.addAction("Delete", lambda: self.document_delete.emit(doc_id))
        is_scoped = "↳ used in:" in item.text()
        if is_scoped:
            menu.addAction("Make global", lambda: self.document_promote.emit(doc_id))
        menu.exec(self._list.mapToGlobal(pos))

    def show_progress(self, filename: str, percent: int):
        self._progress.setVisible(True)
        self._progress.setValue(percent)
        self._progress.setFormat(f"Ingesting {filename} — %p%")

    def show_progress_stage(self, filename: str, stage: str, percent: int):
        self._progress.setVisible(True)
        self._cancel_btn.setVisible(True)
        self._progress.setValue(percent)
        short_name = filename if len(filename) <= 20 else filename[:17] + "..."
        if stage.startswith("Failed"):
            self._progress.setFormat(f"✗ {short_name} — {stage}")
            self._progress.setStyleSheet("QProgressBar::chunk { background: #c44; }")
        else:
            self._progress.setFormat(f"{short_name} — {stage}")
            self._progress.setStyleSheet("")

    def hide_progress(self):
        self._progress.setVisible(False)
        self._cancel_btn.setVisible(False)
        self._progress.setStyleSheet("")

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
    title_generated = Signal(str, str)
    _ingest_progress = Signal(str, str, int)

    def __init__(self, bridge: AsyncBridge, engine, sqlite_store, pipeline, watcher, config: dict):
        super().__init__()
        self._bridge = bridge
        self._engine = engine
        self._sqlite = sqlite_store
        self._pipeline = pipeline
        self._watcher = watcher
        self._config = config
        self._current_thread_id: str | None = None
        self._ingest_cancel = threading.Event()

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
        edit_menu.addAction("Settings...", self._open_settings)

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
        self._doc_panel._cancel_btn.clicked.connect(self._cancel_ingestion)
        self._doc_panel.document_delete.connect(self._on_doc_delete)
        self._doc_panel.document_promote.connect(self._on_doc_promote)
        self.title_generated.connect(self._on_title_generated)
        self._thread_panel.thread_renamed.connect(self._on_thread_renamed)
        self._ingest_progress.connect(self._on_ingest_progress)

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

    @Slot(str, str)
    def _on_thread_renamed(self, thread_id: str, new_title: str):
        new_title = new_title.strip()
        if new_title:
            self._sqlite.update_thread_title(thread_id, new_title)

    @Slot(str)
    def _on_doc_delete(self, document_id: str):
        doc = next((d for d in self._sqlite.list_documents() if d["id"] == document_id), None)
        if doc:
            self._pipeline.remove(Path(doc["source_path"]))
            self._refresh_documents()

    @Slot(str)
    def _on_doc_promote(self, document_id: str):
        self._pipeline.promote_to_global(document_id)
        self._refresh_documents()

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

        self._maybe_generate_title(text)

        self._collected_response = []
        self._bridge.token_received.connect(self._collect_token)

        document_ids = self._doc_panel.get_checked_document_ids() or None

        async def _do_query():
            stream, sources, strategy = await self._engine.aquery_stream(
                text, thread_id=self._current_thread_id, document_ids=document_ids
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

    def _maybe_generate_title(self, first_message: str):
        """Generate a chat title from the first message if thread is still unnamed."""
        item = self._thread_panel.currentItem()
        if not item or item.text() != "New chat":
            return
        thread_id = self._current_thread_id

        prompt = (
            "Generate a very short title (3-6 words, no quotes) summarizing this message:\n\n"
            f"{first_message[:300]}"
        )

        def _generate():
            return self._engine.complete(prompt)

        async def _title_task():
            result = await self._bridge._loop.run_in_executor(None, _generate)
            title = result.strip() if result else ""
            if not title or len(title) > 60:
                title = first_message[:40].strip()
            self._sqlite.update_thread_title(thread_id, title)
            self.title_generated.emit(thread_id, title)

        self._bridge.submit(_title_task())

    @Slot(str, str)
    def _on_title_generated(self, thread_id: str, title: str):
        for i in range(self._thread_panel.count()):
            it = self._thread_panel.item(i)
            if it and it.data(Qt.ItemDataRole.UserRole) == thread_id:
                it.setText(title)
                break

    # --- Documents ---

    def _refresh_documents(self):
        docs = self._sqlite.list_documents()
        threads = self._sqlite.list_threads()
        thread_map = {t["id"]: t["title"] for t in threads}
        self._doc_panel.set_documents(docs, thread_map)

    @Slot(list)
    def _on_files_dropped(self, paths: list[str]):
        self._ingest_cancel.clear()

        async def _ingest_with_progress():
            for path in paths:
                if self._ingest_cancel.is_set():
                    break
                filename = Path(path).name

                def _do_ingest(p=path, fn=filename):
                    def _on_progress(stage: str, percent: int):
                        if self._ingest_cancel.is_set():
                            raise _IngestCancelled()
                        self._ingest_progress.emit(fn, stage, percent)

                    self._pipeline.ingest(
                        Path(p),
                        False,
                        fn,
                        self._current_thread_id,
                        on_progress=_on_progress,
                    )

                try:
                    await self._bridge._loop.run_in_executor(None, _do_ingest)
                except _IngestCancelled:
                    break

            self._ingest_progress.emit("", "Done", 100)

        self._bridge.submit(_ingest_with_progress())

    def _cancel_ingestion(self):
        self._ingest_cancel.set()
        self._doc_panel.hide_progress()

    @Slot(str, str, int)
    def _on_ingest_progress(self, filename: str, stage: str, percent: int):
        if stage == "Done" and not filename:
            self._doc_panel.hide_progress()
            self._refresh_documents()
        elif stage.startswith("Failed"):
            self._doc_panel.show_progress_stage(filename, stage, 0)
        else:
            self._doc_panel.show_progress_stage(filename, stage, percent)

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

    def _open_settings(self):
        from src.ui.settings_dialog import SettingsDialog

        dialog = SettingsDialog(self)
        dialog.exec()

    def closeEvent(self, event):
        if self._watcher:
            self._watcher.stop()
        self._bridge.shutdown()
        self._sqlite.close()
        event.accept()
