"""Run all golden queries through the RAG pipeline and produce raw results."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.embedder import Embedder
from src.retrieval.semantic import SemanticRetriever
from src.storage.qdrant_store import QdrantStore

EVAL_DIR = Path(__file__).resolve().parent
QDRANT_PATH = str(EVAL_DIR / "qdrant_eval_data")
COLLECTION_NAME = "eval_knowledge_base"
QUERIES_PATH = EVAL_DIR / "golden_queries.json"
RESULTS_PATH = EVAL_DIR / "eval_results.json"


def main():
    print("Loading eval infrastructure...", flush=True)
    qdrant = QdrantStore(path=QDRANT_PATH, collection_name=COLLECTION_NAME)
    embedder = Embedder(device="cpu")

    with open(QUERIES_PATH) as f:
        queries = json.load(f)

    print(f"Running {len(queries)} eval queries...", flush=True)
    results = []

    for i, q in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] {q['query'][:60]}...", flush=True)

        query_embedding = embedder.embed_query(q["query"])
        hits = qdrant.search(
            query_embedding=query_embedding,
            top_k=10,
            filter_conditions={"chunk_type": "child"},
        )

        retrieved_sources = []
        retrieved_texts = []
        for hit in hits:
            source = hit["metadata"].get("source_path", "")
            short_source = source.split("rag-golden-ds/")[-1] if "rag-golden-ds" in source else source
            retrieved_sources.append(short_source)
            retrieved_texts.append(hit["metadata"].get("text", "")[:200])

        result = {
            "id": q["id"],
            "query": q["query"],
            "expected_sources": q["expected_sources"],
            "key_facts": q["key_facts"],
            "retrieved_sources": retrieved_sources,
            "retrieved_texts": retrieved_texts,
            "scores": [hit["score"] for hit in hits],
            "source_hit": any(
                any(exp in src for exp in q["expected_sources"])
                for src in retrieved_sources
            ),
        }
        results.append(result)

    hit_count = sum(1 for r in results if r["source_hit"])
    print(f"\nRetrieval source hit rate: {hit_count}/{len(results)} "
          f"({hit_count/len(results)*100:.1f}%)", flush=True)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULTS_PATH}", flush=True)

    qdrant.close()


if __name__ == "__main__":
    main()
