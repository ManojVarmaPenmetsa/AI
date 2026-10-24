from pathlib import Path
from app.core.config import get_settings
from app.services.ingestion import SUPPORTED, stable_document_id
from app.services.document_registry import DocumentRegistry
from app.services.document_replacement import replace_document

settings = get_settings()

folder = Path(settings.sample_kb_dir)
files = [
    path
    for path in folder.iterdir()
    if path.is_file() and path.suffix.lower() in SUPPORTED
]
if not files:
    raise FileNotFoundError(
        f"No supported sample documents found in {folder}. "
        f"Supported extensions: {', '.join(sorted(SUPPORTED))}"
    )

warnings = []
registry = DocumentRegistry(settings.document_registry_db_path)
chunk_count = 0
vector_count = 0

for path in files:
    document_id = registry.resolve_document(
        path.name, legacy_id=stable_document_id(path.name)
    )
    outcome = replace_document(registry, path, path.name, path.read_bytes(), document_id)
    chunk_count += len(outcome["chunks"])
    vector_count += len(outcome["ids"])
    warnings.extend(f"{path.name}: {warning}" for warning in outcome["warnings"])
print(f"Indexed {len(files)} files -> {chunk_count} chunks -> {vector_count} Pinecone vectors")
for warning in warnings:
    print(f"WARNING: {warning}")
