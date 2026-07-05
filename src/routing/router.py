import re


CONCEPTUAL_PATTERNS = re.compile(
    r"(як працює|як діє|what is|how does|how do|explain|поясни|що таке|чому|why|в чому різниця|difference between)",
    re.IGNORECASE,
)

NARROW_PATTERNS = re.compile(
    r"(конкретно|specifically|exactly|точно|саме цей|this specific|цей рядок|this line)",
    re.IGNORECASE,
)

KEYWORD_INDICATORS = re.compile(
    r"(error|exception|помилка|трейсбек|traceback|stacktrace|[A-Z][a-z]+[A-Z][a-z]+|__[a-z]+__|0x[0-9a-f]{4,}|\w+\.\w+\.\w+)",
)


def classify_query(query: str) -> str:
    query_stripped = query.strip()

    if NARROW_PATTERNS.search(query_stripped):
        return "stepback"

    if CONCEPTUAL_PATTERNS.search(query_stripped):
        if len(query_stripped) > 80:
            return "hyde"
        return "stepback"

    if KEYWORD_INDICATORS.search(query_stripped):
        return "hybrid"

    if len(query_stripped) < 30:
        return "semantic"

    if len(query_stripped) > 100:
        return "hyde"

    return "semantic"


class Router:
    def __init__(self, strategies: dict):
        self._strategies = strategies

    def route(self, query: str):
        strategy_name = classify_query(query)
        strategy = self._strategies.get(strategy_name)
        if strategy is None:
            strategy = self._strategies.get("semantic")
        return strategy_name, strategy
