import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from src.core.config import load_config
from src.core.llm_factory import create_llm
from src.core.rag_engine import RAGEngine
from src.generation.generator import Generator
from src.ingestion.embedder import Embedder
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.watcher import FileWatcher
from src.retrieval.hybrid import HybridStrategy
from src.retrieval.hyde import HyDEStrategy
from src.retrieval.reranker import Reranker
from src.retrieval.semantic import SemanticStrategy
from src.retrieval.stepback import StepBackStrategy
from src.routing.router import Router
from src.storage.qdrant_store import QdrantStore
from src.storage.sqlite_store import SQLiteStore
from src.ui.async_bridge import AsyncBridge
from src.ui.main_window import MainWindow


def _build_strategies(qdrant: QdrantStore, embedder: Embedder, llm: Any) -> dict[str, Any]:
    return {
        "semantic": SemanticStrategy(qdrant_store=qdrant, embedder=embedder),
        "hybrid": HybridStrategy(qdrant_store=qdrant, embedder=embedder),
        "hyde": HyDEStrategy(qdrant_store=qdrant, embedder=embedder, llm=llm),
        "stepback": StepBackStrategy(qdrant_store=qdrant, embedder=embedder, llm=llm),
    }


def _initialize(config: dict) -> tuple[RAGEngine, SQLiteStore, IngestionPipeline, FileWatcher]:
    storage_cfg = config["storage"]

    embedder = Embedder(
        model_name=f"intfloat/{config['embedding']['model_name']}",
        device=config["embedding"]["device"],
    )
    qdrant = QdrantStore(
        path=storage_cfg["qdrant_path"],
        collection_name=storage_cfg["collection_name"],
        vector_size=embedder.vector_size,
    )
    sqlite = SQLiteStore(db_path=storage_cfg["sqlite_path"])

    chunking_cfg = config.get("chunking", {})
    pipeline = IngestionPipeline(
        qdrant_store=qdrant,
        sqlite_store=sqlite,
        embedder=embedder,
        parent_chunk_size=chunking_cfg.get("parent_chunk_size", 1000),
        child_chunk_size=chunking_cfg.get("child_chunk_size", 200),
        chunk_overlap=chunking_cfg.get("chunk_overlap", 50),
    )

    llm = create_llm(config)
    system_prompt = config.get("system_prompt", "You are a helpful assistant.")
    generator = Generator(llm=llm, system_prompt=system_prompt)
    strategies = _build_strategies(qdrant, embedder, llm)
    router = Router(strategies=strategies)

    reranker_cfg = config.get("reranker", {})
    reranker = Reranker(
        model_name=reranker_cfg.get("model_name", "BAAI/bge-reranker-v2-m3"),
        device=reranker_cfg.get("device", "cpu"),
    )

    engine = RAGEngine(
        router=router,
        reranker=reranker,
        generator=generator,
        top_k=config["retrieval"]["top_k"],
        rerank_top_n=reranker_cfg.get("top_n", 5),
        qdrant_store=qdrant,
    )

    kb_cfg = config["knowledge_base"]
    watcher = FileWatcher(
        folder=kb_cfg["monitored_folder"],
        supported_extensions=kb_cfg["supported_extensions"],
        on_ingest=pipeline.ingest,
        on_delete=pipeline.remove,
    )
    watcher.start()

    return engine, sqlite, pipeline, watcher


def _apply_dark_palette(app: QApplication):
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


def main():
    qt_app = QApplication(sys.argv)
    qt_app.setStyle("Fusion")
    _apply_dark_palette(qt_app)

    config = load_config()
    engine, sqlite, pipeline, watcher = _initialize(config)
    bridge = AsyncBridge()

    window = MainWindow(
        bridge=bridge,
        engine=engine,
        sqlite_store=sqlite,
        pipeline=pipeline,
        watcher=watcher,
        config=config,
    )
    window.show()

    exit_code = qt_app.exec()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
