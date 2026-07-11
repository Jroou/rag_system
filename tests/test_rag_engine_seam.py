"""Tests for RAGEngine.complete() and RAGEngine.update_settings()."""

from unittest.mock import MagicMock

from src.core.rag_engine import RAGEngine
from src.generation.generator import Generator
from src.retrieval.reranker import Reranker
from src.routing.router import Router


def _make_engine(llm=None):
    router = MagicMock(spec=Router)
    reranker = MagicMock(spec=Reranker)
    if llm is None:
        llm = MagicMock()
    generator = Generator(llm=llm, system_prompt="test")
    return RAGEngine(
        router=router,
        reranker=reranker,
        generator=generator,
        top_k=10,
        rerank_top_n=3,
    )


class TestComplete:
    def test_returns_llm_response_content(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="compressed result")
        engine = _make_engine(llm=mock_llm)

        result = engine.complete("summarize this")

        assert result == "compressed result"
        mock_llm.invoke.assert_called_once_with(
            [{"role": "user", "content": "summarize this"}]
        )

    def test_returns_empty_string_on_exception(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionError("LLM unavailable")
        engine = _make_engine(llm=mock_llm)

        result = engine.complete("any prompt")

        assert result == ""

    def test_empty_prompt_still_calls_llm(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="")
        engine = _make_engine(llm=mock_llm)

        result = engine.complete("")

        mock_llm.invoke.assert_called_once()
        assert result == ""


class TestUpdateSettings:
    def test_update_top_k(self):
        engine = _make_engine()
        assert engine._top_k == 10

        engine.update_settings(top_k=25)

        assert engine._top_k == 25

    def test_update_rerank_top_n(self):
        engine = _make_engine()
        assert engine._rerank_top_n == 3

        engine.update_settings(rerank_top_n=8)

        assert engine._rerank_top_n == 8

    def test_update_generator(self):
        engine = _make_engine()
        new_llm = MagicMock()
        new_generator = Generator(llm=new_llm, system_prompt="new prompt")

        engine.update_settings(generator=new_generator)

        assert engine._generator is new_generator

    def test_update_all_at_once(self):
        engine = _make_engine()
        new_generator = Generator(llm=MagicMock(), system_prompt="updated")

        engine.update_settings(top_k=30, rerank_top_n=7, generator=new_generator)

        assert engine._top_k == 30
        assert engine._rerank_top_n == 7
        assert engine._generator is new_generator

    def test_none_params_leave_fields_unchanged(self):
        engine = _make_engine()
        original_generator = engine._generator

        engine.update_settings()  # all None

        assert engine._top_k == 10
        assert engine._rerank_top_n == 3
        assert engine._generator is original_generator

    def test_partial_update_leaves_other_fields_unchanged(self):
        engine = _make_engine()
        original_generator = engine._generator

        engine.update_settings(top_k=15)

        assert engine._top_k == 15
        assert engine._rerank_top_n == 3
        assert engine._generator is original_generator
