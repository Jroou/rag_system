"""Tests for CommandDispatcher — routing and handler delegation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


def _make_dispatcher():
    """Construct a CommandDispatcher with minimal mocks."""
    with patch.dict("sys.modules", {"chainlit": MagicMock()}):
        from src.ui.command_dispatcher import CommandDispatcher  # noqa: PLC0415

        engine = MagicMock()
        engine.complete = MagicMock(return_value="summary")
        engine.update_generator = MagicMock()
        engine.update_settings = MagicMock()

        pipeline = MagicMock()
        config = {
            "llm": {
                "profiles": {
                    "default": {"provider": "openai", "model": "gpt-4o", "temperature": 0.3},
                    "fast": {"provider": "openai", "model": "gpt-4o-mini"},
                },
                "active_profile": "default",
            },
            "system_prompt": "You are a helpful assistant.",
            "knowledge_base": {
                "monitored_folder": "/tmp/kb",
                "supported_extensions": [".pdf"],
            },
        }
        sqlite = MagicMock()
        watcher = MagicMock()

        dispatcher = CommandDispatcher(engine, pipeline, config, sqlite, watcher)
        return dispatcher, engine, pipeline, config, sqlite


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Routing — returns True for known commands, False for plain text
# ---------------------------------------------------------------------------

class TestDispatchRouting:
    def test_plain_text_returns_false(self):
        dispatcher, *_ = _make_dispatcher()
        with patch.object(dispatcher, "_handle_profile", new_callable=AsyncMock):
            result = _run(dispatcher.dispatch("What is the capital of France?"))
        assert result is False

    def test_profile_command_returns_true(self):
        dispatcher, *_ = _make_dispatcher()
        with patch.object(dispatcher, "_handle_profile", new_callable=AsyncMock) as mock_h:
            result = _run(dispatcher.dispatch("/profile fast"))
        assert result is True
        mock_h.assert_awaited_once_with("fast")

    def test_findings_returns_true(self):
        dispatcher, *_ = _make_dispatcher()
        with patch.object(dispatcher, "_handle_findings", new_callable=AsyncMock) as mock_h:
            result = _run(dispatcher.dispatch("/findings"))
        assert result is True
        mock_h.assert_awaited_once()

    def test_finding_delete_returns_true(self):
        dispatcher, *_ = _make_dispatcher()
        with patch.object(dispatcher, "_handle_finding_delete", new_callable=AsyncMock) as mock_h:
            result = _run(dispatcher.dispatch("/finding delete abc123"))
        assert result is True
        mock_h.assert_awaited_once_with("abc123")

    def test_settings_returns_true(self):
        dispatcher, *_ = _make_dispatcher()
        with patch.object(dispatcher, "_handle_settings_show", new_callable=AsyncMock) as mock_h:
            result = _run(dispatcher.dispatch("/settings"))
        assert result is True
        mock_h.assert_awaited_once()

    def test_settings_set_returns_true(self):
        dispatcher, *_ = _make_dispatcher()
        with patch.object(dispatcher, "_handle_settings_set", new_callable=AsyncMock) as mock_h:
            result = _run(dispatcher.dispatch("/settings set retrieval.top_k 20"))
        assert result is True
        mock_h.assert_awaited_once_with("retrieval.top_k", "20")

    def test_documents_returns_true(self):
        dispatcher, *_ = _make_dispatcher()
        with patch.object(dispatcher, "_handle_documents", new_callable=AsyncMock) as mock_h:
            result = _run(dispatcher.dispatch("/documents"))
        assert result is True
        mock_h.assert_awaited_once()

    def test_reindex_returns_true(self):
        dispatcher, *_ = _make_dispatcher()
        with patch.object(dispatcher, "_handle_reindex", new_callable=AsyncMock) as mock_h:
            result = _run(dispatcher.dispatch("/reindex report.pdf"))
        assert result is True
        mock_h.assert_awaited_once_with("report.pdf")

    def test_doc_delete_returns_true(self):
        dispatcher, *_ = _make_dispatcher()
        with patch.object(dispatcher, "_handle_doc_delete", new_callable=AsyncMock) as mock_h:
            result = _run(dispatcher.dispatch("/doc delete report.pdf"))
        assert result is True
        mock_h.assert_awaited_once_with("report.pdf")

    def test_history_returns_true(self):
        dispatcher, *_ = _make_dispatcher()
        with patch.object(dispatcher, "_handle_history", new_callable=AsyncMock) as mock_h:
            result = _run(dispatcher.dispatch("/history"))
        assert result is True
        mock_h.assert_awaited_once()

    def test_compress_returns_true(self):
        dispatcher, *_ = _make_dispatcher()
        with patch.object(dispatcher, "_handle_compress", new_callable=AsyncMock) as mock_h:
            result = _run(dispatcher.dispatch("/compress"))
        assert result is True
        mock_h.assert_awaited_once()

    def test_unknown_slash_command_returns_false(self):
        dispatcher, *_ = _make_dispatcher()
        result = _run(dispatcher.dispatch("/unknown_command_xyz"))
        assert result is False


# ---------------------------------------------------------------------------
# Handler behaviour — _handle_profile
# ---------------------------------------------------------------------------

class TestHandleProfile:
    def test_known_profile_calls_update_generator(self):
        fake_llm = MagicMock()
        cl_mock = MagicMock()
        cl_mock.Message.return_value.send = AsyncMock()

        with (
            patch.dict("sys.modules", {"chainlit": cl_mock}),
            patch("src.ui.command_dispatcher.create_llm_from_profile", return_value=fake_llm),
            patch("src.ui.command_dispatcher.cl", cl_mock),
        ):
            from src.ui.command_dispatcher import CommandDispatcher  # noqa: PLC0415
            engine = MagicMock()
            engine.update_generator = MagicMock()
            config = {
                "llm": {
                    "profiles": {
                        "default": {"provider": "openai", "model": "gpt-4o"},
                        "fast": {"provider": "openai", "model": "gpt-4o-mini"},
                    },
                    "active_profile": "default",
                },
                "system_prompt": "You are a helpful assistant.",
            }
            dispatcher = CommandDispatcher(engine, MagicMock(), config, MagicMock())
            _run(dispatcher._handle_profile("fast"))

        engine.update_generator.assert_called_once()

    def test_unknown_profile_sends_error(self):
        cl_mock = MagicMock()
        cl_mock.Message.return_value.send = AsyncMock()

        with (
            patch.dict("sys.modules", {"chainlit": cl_mock}),
            patch("src.ui.command_dispatcher.cl", cl_mock),
        ):
            from src.ui.command_dispatcher import CommandDispatcher  # noqa: PLC0415
            engine = MagicMock()
            config = {
                "llm": {
                    "profiles": {"default": {}, "fast": {}},
                    "active_profile": "default",
                },
                "system_prompt": "You are a helpful assistant.",
            }
            dispatcher = CommandDispatcher(engine, MagicMock(), config, MagicMock())
            _run(dispatcher._handle_profile("nonexistent"))

        engine.update_generator.assert_not_called()
        sent_content = cl_mock.Message.call_args[1]["content"]
        assert "Unknown profile" in sent_content


# ---------------------------------------------------------------------------
# Handler behaviour — _handle_compress uses engine.complete()
# ---------------------------------------------------------------------------

class TestHandleCompress:
    def test_compress_calls_engine_complete(self):
        sqlite = MagicMock()
        sqlite.get_thread_messages.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        sqlite.update_thread_summary = MagicMock()

        cl_mock = MagicMock()
        cl_mock.Message.return_value.send = AsyncMock()
        cl_mock.user_session.get.return_value = "thread-abc"

        with (
            patch.dict("sys.modules", {"chainlit": cl_mock}),
            patch("src.ui.command_dispatcher.cl", cl_mock),
        ):
            from src.ui.command_dispatcher import CommandDispatcher  # noqa: PLC0415
            engine = MagicMock()
            engine.complete = MagicMock(return_value="summary")
            config = {
                "llm": {"profiles": {}, "active_profile": "default"},
                "system_prompt": "You are a helpful assistant.",
            }
            dispatcher = CommandDispatcher(engine, MagicMock(), config, sqlite)
            _run(dispatcher._handle_compress())

        engine.complete.assert_called_once()
        sqlite.update_thread_summary.assert_called_once_with("thread-abc", "summary")
