import subprocess
import sys
from pathlib import Path

from src.core.config import load_config


def main() -> None:
    config = load_config()
    print("RAG System v0.1.0")
    print(f"Monitored folder: {config['knowledge_base']['monitored_folder']}")
    print(f"Embedding model: {config['embedding']['model_name']}")
    print(f"Active LLM profile: {config['llm']['active_profile']}")
    print(f"Vector store path: {config['storage']['qdrant_path']}")
    print()

    app_path = Path(__file__).parent / "ui" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "chainlit", "run", str(app_path), "--host", "0.0.0.0", "--port", "8000"],
        check=True,
    )


if __name__ == "__main__":
    main()
