from pathlib import Path

import yaml
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumSize(550, 450)

        self._config = self._load_config()
        self._fields: dict[str, QWidget] = {}

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        tabs.addTab(self._build_llm_tab(), "LLM")
        tabs.addTab(self._build_retrieval_tab(), "Retrieval")
        tabs.addTab(self._build_chunking_tab(), "Chunking")
        tabs.addTab(self._build_system_prompt_tab(), "System Prompt")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_config(self) -> dict:
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)

    def _save(self):
        self._apply_fields_to_config()
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True, sort_keys=True)
        self.accept()

    def _apply_fields_to_config(self):
        c = self._config

        # LLM
        c["llm"]["active_profile"] = self._fields["active_profile"].currentText()
        for profile_name, widgets in self._profile_widgets.items():
            profile = c["llm"]["profiles"][profile_name]
            profile["model"] = widgets["model"].text()
            profile["temperature"] = float(widgets["temperature"].text() or "0.3")
            api_key = widgets["api_key"].text().strip()
            if api_key:
                profile["api_key_env"] = api_key
            elif "api_key_env" in profile and not api_key:
                pass

        # Retrieval
        c["retrieval"]["top_k"] = self._fields["retrieval_top_k"].value()
        c["retrieval"]["default_strategy"] = self._fields["default_strategy"].currentText()
        c["reranker"]["top_n"] = self._fields["reranker_top_n"].value()
        c["reranker"]["model_name"] = self._fields["reranker_model"].text()

        # Chunking
        c["chunking"]["parent_chunk_size"] = self._fields["parent_chunk_size"].value()
        c["chunking"]["child_chunk_size"] = self._fields["child_chunk_size"].value()
        c["chunking"]["chunk_overlap"] = self._fields["chunk_overlap"].value()

        # System prompt
        c["system_prompt"] = self._fields["system_prompt"].toPlainText()

    def _build_llm_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Active profile selector
        row = QHBoxLayout()
        row.addWidget(QLabel("Active profile:"))
        combo = QComboBox()
        profiles = list(self._config.get("llm", {}).get("profiles", {}).keys())
        combo.addItems(profiles)
        combo.setCurrentText(self._config.get("llm", {}).get("active_profile", ""))
        self._fields["active_profile"] = combo
        row.addWidget(combo)
        layout.addLayout(row)

        layout.addWidget(QLabel(""))

        self._profile_widgets: dict[str, dict[str, QLineEdit]] = {}
        for name, profile in self._config.get("llm", {}).get("profiles", {}).items():
            layout.addWidget(QLabel(f"— {name} ({profile.get('provider', '')}) —"))
            form = QFormLayout()

            model_edit = QLineEdit(profile.get("model", ""))
            form.addRow("Model:", model_edit)

            temp_edit = QLineEdit(str(profile.get("temperature", 0.3)))
            temp_edit.setMaximumWidth(80)
            form.addRow("Temperature:", temp_edit)

            api_key_edit = QLineEdit(profile.get("api_key_env", ""))
            api_key_edit.setPlaceholderText("env var name (e.g. ANTHROPIC_API_KEY)")
            form.addRow("API key env:", api_key_edit)

            self._profile_widgets[name] = {
                "model": model_edit,
                "temperature": temp_edit,
                "api_key": api_key_edit,
            }

            layout.addLayout(form)

        layout.addStretch()
        return widget

    def _build_retrieval_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)

        top_k = QSpinBox()
        top_k.setRange(1, 100)
        top_k.setValue(self._config.get("retrieval", {}).get("top_k", 30))
        self._fields["retrieval_top_k"] = top_k
        form.addRow("Retrieval top_k:", top_k)

        strategy = QComboBox()
        strategy.addItems(["semantic", "hybrid", "hyde", "stepback"])
        strategy.setCurrentText(self._config.get("retrieval", {}).get("default_strategy", "semantic"))
        self._fields["default_strategy"] = strategy
        form.addRow("Default strategy:", strategy)

        reranker_top_n = QSpinBox()
        reranker_top_n.setRange(1, 50)
        reranker_top_n.setValue(self._config.get("reranker", {}).get("top_n", 7))
        self._fields["reranker_top_n"] = reranker_top_n
        form.addRow("Reranker top_n:", reranker_top_n)

        reranker_model = QLineEdit(self._config.get("reranker", {}).get("model_name", ""))
        self._fields["reranker_model"] = reranker_model
        form.addRow("Reranker model:", reranker_model)

        return widget

    def _build_chunking_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)

        parent = QSpinBox()
        parent.setRange(100, 10000)
        parent.setValue(self._config.get("chunking", {}).get("parent_chunk_size", 1000))
        self._fields["parent_chunk_size"] = parent
        form.addRow("Parent chunk size:", parent)

        child = QSpinBox()
        child.setRange(50, 5000)
        child.setValue(self._config.get("chunking", {}).get("child_chunk_size", 200))
        self._fields["child_chunk_size"] = child
        form.addRow("Child chunk size:", child)

        overlap = QSpinBox()
        overlap.setRange(0, 500)
        overlap.setValue(self._config.get("chunking", {}).get("chunk_overlap", 50))
        self._fields["chunk_overlap"] = overlap
        form.addRow("Chunk overlap:", overlap)

        return widget

    def _build_system_prompt_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        prompt_edit = QPlainTextEdit()
        prompt_edit.setPlainText(self._config.get("system_prompt", ""))
        self._fields["system_prompt"] = prompt_edit
        layout.addWidget(prompt_edit)

        return widget
