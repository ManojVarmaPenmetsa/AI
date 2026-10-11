from pathlib import Path


ROOT = Path(__file__).resolve().parent

FOLDERS = (
    "app/api",
    "app/core",
    "app/rag",
    "app/services",
    "data/sample_kb",
    "templates",
    "static/css",
    "static/js",
    "uploads",
    "tests",
)

FILES = {
    "app/__init__.py": "",
    "app/api/__init__.py": "",
    "app/api/routes.py": "",
    "app/core/__init__.py": "",
    "app/core/config.py": "",
    "app/core/logging.py": "",
    "app/rag/__init__.py": "",
    "app/rag/state.py": "",
    "app/rag/vectorstore.py": "",
    "app/rag/workflow.py": "",
    "app/services/__init__.py": "",
    "app/services/audit.py": "",
    "app/services/ingestion.py": "",
    "app/main.py": "",
    "ingest_sample_kb.py": "",
    "run.py": "",
    "templates/index.html": "",
    "static/css/style.css": "",
    "static/js/app.js": "",
    "tests/test_ingestion.py": "",
    "data/sample_kb/.gitkeep": "",
    "uploads/.gitkeep": "",
    ".env.example": (
        "# Copy this file to .env and add your own API keys.\n"
        "OPENROUTER_API_KEY=\n"
        "PINECONE_API_KEY=\n"
        "TAVILY_API_KEY=\n"
    ),
}


def main() -> None:
    created_folders: list[str] = []
    created_files: list[str] = []
    existing_files: list[str] = []

    for relative_path in FOLDERS:
        path = ROOT / relative_path
        if path.exists() and not path.is_dir():
            raise NotADirectoryError(f"Expected a directory but found a file: {path}")
        if not path.exists():
            path.mkdir(parents=True)
            created_folders.append(relative_path)

    for relative_path, content in FILES.items():
        path = ROOT / relative_path
        if path.exists():
            if not path.is_file():
                raise IsADirectoryError(f"Expected a file but found a directory: {path}")
            existing_files.append(relative_path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as file:
            file.write(content)
        created_files.append(relative_path)

    print("Project scaffold is ready.")
    print(f"Created {len(created_folders)} folders and {len(created_files)} files.")
    if existing_files:
        print(f"Preserved {len(existing_files)} existing files.")
    print("No .env file or API keys were created.")
    print("These files are placeholders; implement them before running the web app.")


if __name__ == "__main__":
    main()