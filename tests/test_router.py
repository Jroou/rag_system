import pytest

from src.routing.router import Router, classify_query


class TestClassifyQuery:
    def test_short_factual_is_semantic(self):
        assert classify_query("vector database") == "semantic"
        assert classify_query("Qdrant") == "semantic"

    def test_keyword_heavy_is_hybrid(self):
        assert classify_query("NullPointerException in auth module") == "hybrid"
        assert classify_query("помилка TypeError при виклику функції") == "hybrid"
        assert classify_query("error in config.yaml") == "hybrid"

    def test_conceptual_is_stepback_or_hyde(self):
        result = classify_query("як працює dependency injection?")
        assert result in ("stepback", "hyde")

        result = classify_query("what is the difference between REST and GraphQL?")
        assert result in ("stepback", "hyde")

    def test_narrow_is_stepback(self):
        assert classify_query("конкретно цей рядок коду робить?") == "stepback"
        assert classify_query("specifically this line does what?") == "stepback"

    def test_long_abstract_is_hyde(self):
        long_query = (
            "Я хочу зрозуміти як правильно організувати архітектуру мікросервісів "
            "щоб забезпечити масштабованість та відмовостійкість системи"
        )
        assert classify_query(long_query) == "hyde"

    def test_ukrainian_conceptual(self):
        result = classify_query("поясни як працює цей алгоритм сортування")
        assert result in ("stepback", "hyde")

    def test_english_why_question(self):
        result = classify_query("why does this function return None?")
        assert result in ("hybrid", "stepback")


class TestRouter:
    def test_routes_to_correct_strategy(self):
        mock_strategies = {
            "semantic": "semantic_strategy",
            "hybrid": "hybrid_strategy",
            "hyde": "hyde_strategy",
            "stepback": "stepback_strategy",
        }
        router = Router(strategies=mock_strategies)

        name, strategy = router.route("vector database")
        assert name == "semantic"
        assert strategy == "semantic_strategy"

    def test_falls_back_to_semantic_if_missing(self):
        router = Router(strategies={"semantic": "fallback"})
        name, strategy = router.route("NullPointerException error")
        assert strategy == "fallback"
