"""Score the RAG pipeline: retrieval metrics (no LLM) + generation metrics (LLM-judge).

Outputs JSON to stdout with structure:
{
  "retrieval": {"source_hit_rate": float, "recall_at_10": float},
  "generation": {"faithfulness": float, "completeness": float, "formatting": float},
  "per_query": [...]
}

Exit code 0 on success, 1 on fatal error.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import load_config
from src.ingestion.embedder import Embedder
from src.retrieval.base import RetrievalResult
from src.retrieval.reranker import Reranker
from src.retrieval.semantic import SemanticStrategy
from src.retrieval.hybrid import HybridStrategy
from src.retrieval.hyde import HyDEStrategy
from src.retrieval.stepback import StepBackStrategy
from src.routing.router import Router
from src.storage.qdrant_store import QdrantStore

EVAL_DIR = Path(__file__).resolve().parent
QDRANT_PATH = str(EVAL_DIR / "qdrant_eval_data")
COLLECTION_NAME = "eval_knowledge_base"
QUERIES_PATH = EVAL_DIR / "golden_queries.json"


def build_eval_engine():
    config = load_config()
    qdrant = QdrantStore(path=QDRANT_PATH, collection_name=COLLECTION_NAME)
    embedder = Embedder(device="cpu")

    semantic = SemanticStrategy(qdrant, embedder)
    strategies = {"semantic": semantic}

    try:
        hybrid = HybridStrategy(qdrant, embedder)
        strategies["hybrid"] = hybrid
    except Exception:
        strategies["hybrid"] = semantic

    try:
        from src.core.llm_factory import create_llm
        llm = create_llm(config)
        hyde = HyDEStrategy(qdrant, embedder, llm)
        stepback = StepBackStrategy(qdrant, embedder, llm)
        strategies["hyde"] = hyde
        strategies["stepback"] = stepback
    except Exception:
        strategies["hyde"] = semantic
        strategies["stepback"] = semantic

    router = Router(strategies)

    reranker_cfg = config.get("reranker", {})
    reranker = Reranker(
        model_name=reranker_cfg.get("model_name", "BAAI/bge-reranker-v2-m3"),
        device=reranker_cfg.get("device", "cpu"),
    )

    rerank_top_n = reranker_cfg.get("top_n", 5)

    return router, reranker, rerank_top_n, qdrant, config


def score_retrieval(queries, router, reranker, top_k=20, rerank_top_n=5):
    per_query = []
    for q in queries:
        strategy_name, strategy = router.route(q["query"])
        results = strategy.retrieve(q["query"], top_k=top_k)

        if results:
            results = reranker.rerank(q["query"], results, top_n=rerank_top_n)

        retrieved_sources = []
        for r in results:
            short = r.source_path.split("rag-golden-ds/")[-1] if "rag-golden-ds" in r.source_path else r.source_path
            retrieved_sources.append(short)

        source_hit = any(
            any(exp in src for exp in q["expected_sources"])
            for src in retrieved_sources
        )

        found_expected = sum(
            1 for exp in q["expected_sources"]
            if any(exp in src for src in retrieved_sources[:10])
        )
        max_possible = len(q["expected_sources"])
        recall = found_expected / max_possible if max_possible > 0 else 0.0

        per_query.append({
            "id": q["id"],
            "query": q["query"],
            "strategy": strategy_name,
            "source_hit": source_hit,
            "recall_at_10": recall,
            "retrieved_sources": retrieved_sources[:10],
            "expected_sources": q["expected_sources"],
            "top_scores": [r.score for r in results[:5]],
        })

    hit_rate = sum(1 for pq in per_query if pq["source_hit"]) / len(per_query)
    avg_recall = sum(pq["recall_at_10"] for pq in per_query) / len(per_query)

    return {"source_hit_rate": round(hit_rate, 4), "recall_at_10": round(avg_recall, 4)}, per_query


def score_generation(queries, router, reranker, config, top_k=20, rerank_top_n=5):
    """Score generation quality using LLM-as-judge. Returns None if LLM unavailable."""
    try:
        from src.core.llm_factory import create_llm
        from src.generation.generator import Generator
        llm = create_llm(config)
    except Exception:
        return None, []

    generator = Generator(llm, config.get("system_prompt", ""))

    judge_prompt = """You are an evaluation judge. Score the following response on three dimensions (0.0 to 1.0):

1. **Faithfulness**: Does the response only contain information supported by the provided context? (1.0 = fully faithful, 0.0 = hallucinated)
2. **Completeness**: Does the response cover the key facts expected? (1.0 = all key facts covered, 0.0 = none)
3. **Formatting**: Is the response well-structured, concise, and using appropriate headings/lists? (1.0 = excellent, 0.0 = poor)

Context provided to the system:
{context}

Question: {query}

System response: {response}

Expected key facts: {key_facts}

Respond ONLY with JSON: {{"faithfulness": float, "completeness": float, "formatting": float}}"""

    per_query = []
    scores_sum = {"faithfulness": 0.0, "completeness": 0.0, "formatting": 0.0}
    count = 0

    for q in queries:
        strategy_name, strategy = router.route(q["query"])
        results = strategy.retrieve(q["query"], top_k=top_k)
        if results:
            results = reranker.rerank(q["query"], results, top_n=rerank_top_n)

        if not results:
            per_query.append({"id": q["id"], "scores": None, "error": "no_results"})
            continue

        response = generator.generate(q["query"], results)

        context_text = "\n".join(
            f"[{i+1}] {r.source_path}: {(r.parent_text or r.text)[:300]}"
            for i, r in enumerate(results[:5])
        )

        judge_input = judge_prompt.format(
            context=context_text,
            query=q["query"],
            response=response,
            key_facts=", ".join(q["key_facts"]),
        )

        try:
            judge_response = llm.invoke([
                {"role": "system", "content": "You are a precise evaluation judge. Output only valid JSON."},
                {"role": "user", "content": judge_input},
            ])
            content = judge_response.content
            # Extract JSON from response
            if "{" in content:
                json_str = content[content.index("{"):content.rindex("}") + 1]
                scores = json.loads(json_str)
            else:
                per_query.append({"id": q["id"], "scores": None, "error": "parse_error"})
                continue

            per_query.append({"id": q["id"], "scores": scores, "response_preview": response[:200]})
            for k in scores_sum:
                scores_sum[k] += scores.get(k, 0.0)
            count += 1
        except Exception as e:
            per_query.append({"id": q["id"], "scores": None, "error": str(e)[:100]})

    if count == 0:
        return None, per_query

    avg_scores = {k: round(v / count, 4) for k, v in scores_sum.items()}
    return avg_scores, per_query


def main():
    print("Loading eval infrastructure...", file=sys.stderr, flush=True)
    router, reranker, rerank_top_n, qdrant, config = build_eval_engine()

    with open(QUERIES_PATH) as f:
        queries = json.load(f)

    print(f"Scoring {len(queries)} queries...", file=sys.stderr, flush=True)

    top_k = config.get("retrieval", {}).get("top_k", 20)

    retrieval_scores, retrieval_per_query = score_retrieval(
        queries, router, reranker, top_k=top_k, rerank_top_n=rerank_top_n
    )

    print(f"Retrieval: hit_rate={retrieval_scores['source_hit_rate']}, recall@10={retrieval_scores['recall_at_10']}", file=sys.stderr, flush=True)

    gen_scores, gen_per_query = score_generation(
        queries, router, reranker, config, top_k=top_k, rerank_top_n=rerank_top_n
    )

    if gen_scores:
        print(f"Generation: faith={gen_scores['faithfulness']}, complete={gen_scores['completeness']}, format={gen_scores['formatting']}", file=sys.stderr, flush=True)
    else:
        print("Generation scoring skipped (LLM unavailable or all queries failed)", file=sys.stderr, flush=True)

    combined_per_query = []
    for rq in retrieval_per_query:
        entry = {**rq}
        gq = next((g for g in gen_per_query if g["id"] == rq["id"]), None) if gen_per_query else None
        if gq:
            entry["generation"] = gq.get("scores")
            entry["response_preview"] = gq.get("response_preview")
        combined_per_query.append(entry)

    output = {
        "retrieval": retrieval_scores,
        "generation": gen_scores or {"faithfulness": None, "completeness": None, "formatting": None},
        "per_query": combined_per_query,
    }

    qdrant.close()
    json.dump(output, sys.stdout, indent=2)
    print(file=sys.stdout)


if __name__ == "__main__":
    main()
