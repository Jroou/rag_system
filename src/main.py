import sys

from src.core.config import load_config


def main() -> None:
    config = load_config()
    print(f"RAG System v0.1.0")
    print(f"Monitored folder: {config['knowledge_base']['monitored_folder']}")
    print(f"Embedding model: {config['embedding']['model_name']}")
    print(f"Active LLM profile: {config['llm']['active_profile']}")
    print(f"Vector store path: {config['storage']['qdrant_path']}")
    print()
    print("System ready. Chainlit UI will be available here in future versions.")
    sys.exit(0)


if __name__ == "__main__":
    main()
