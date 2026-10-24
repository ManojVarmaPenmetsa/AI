import time
from app.services.document_registry import DocumentRegistry
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.core.config import get_settings

settings = get_settings()


_embeddings = None
_vectorstore = None


EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-minilm-l6-v2": 384,
}



def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or settings.embedding_model or "").strip()
    if not name:
        raise RuntimeError("Embedding model is not configured")

    normalized = name.lower()
    if normalized in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[normalized]
    if "text-embedding-3-small" in normalized:
        return 1536
    if "text-embedding-3-large" in normalized:
        return 3072
    if "text-embedding-ada-002" in normalized:
        return 1536
    if "all-minilm" in normalized:
        return 384

    raise ValueError(
        f"Unsupported embedding model '{model_name or settings.embedding_model}' for Pinecone. "
        "Add the matching dimension to EMBEDDING_DIMENSIONS."
    )



def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings




def ensure_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    desired_dimension = get_embedding_dimension()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    names = [x["name"] for x in pc.list_indexes()]

    if settings.pinecone_index_name in names:
        index_info = pc.describe_index(settings.pinecone_index_name)
        current_dimension = getattr(index_info, "dimension", None)
        if current_dimension is None and isinstance(index_info, dict):
            current_dimension = index_info.get("dimension")

        if current_dimension is not None and current_dimension != desired_dimension:
            raise RuntimeError(
                f"Pinecone index '{settings.pinecone_index_name}' has dimension "
                f"{current_dimension}, but embedding model '{settings.embedding_model}' "
                f"produces {desired_dimension} dimensions. Choose a new index name or "
                "migrate the index; this app will not delete existing index data."
            )

    if settings.pinecone_index_name not in [x["name"] for x in pc.list_indexes()]:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
            time.sleep(1)

    return pc.Index(settings.pinecone_index_name)





def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        index = ensure_index()
        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )
    return _vectorstore



def get_retriever():
    registry = DocumentRegistry(settings.document_registry_db_path)
    return get_vectorstore().as_retriever(
        search_kwargs={"k": settings.top_k, "filter": registry.active_generation_filter()}
    )



def add_documents(chunks):
    store = get_vectorstore()
    ids = [
        str(doc.metadata.get("chunk_id") or _fallback_chunk_id(doc))
        for doc in chunks
    ]
    return store.add_documents(chunks, ids=ids)


def upsert_generation_documents(chunks) -> list[str]:
    """Upsert and fetch-verify every expected ID before a generation can activate."""
    expected = [str(doc.metadata["chunk_id"]) for doc in chunks]
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("Generation has no chunks or contains duplicate chunk IDs")
    returned = add_documents(chunks)
    if returned is None or set(map(str, returned)) != set(expected) or len(returned) != len(expected):
        raise RuntimeError("Pinecone did not confirm every expected chunk ID")

    index = get_vectorstore().index
    for offset in range(0, len(expected), 100):
        batch = expected[offset:offset + 100]
        fetched = index.fetch(ids=batch, namespace=settings.pinecone_namespace)
        vectors = fetched.get("vectors", {}) if isinstance(fetched, dict) else getattr(fetched, "vectors", {})
        missing = set(batch) - set(vectors or {})
        if missing:
            raise RuntimeError(f"Pinecone fetch verification missed {len(missing)} chunk IDs")
    return expected


def delete_inactive_document_generations(document_id: str, active_generation_id: str) -> None:
    """Delete only inactive/legacy vectors belonging to this registered document."""
    index = get_vectorstore().index
    index.delete(
        namespace=settings.pinecone_namespace,
        filter={"document_id": document_id, "generation_id": {"$ne": active_generation_id}},
    )


def delete_vector_ids(ids: list[str]) -> None:
    if not ids:
        return
    get_vectorstore().index.delete(ids=ids, namespace=settings.pinecone_namespace)


def _fallback_chunk_id(document):
    import hashlib

    source = str(document.metadata.get("source") or "")
    page = str(document.metadata.get("page") or "")
    content = document.page_content
    return hashlib.sha256(f"{source}|{page}|{content}".encode("utf-8")).hexdigest()


def delete_document_vectors(document_id: str) -> None:
    store = get_vectorstore()
    store.index.delete(
        namespace=settings.pinecone_namespace,
        filter={"document_id": document_id},
    )


def delete_legacy_source_vectors(source: str) -> None:
    """Clean up vectors created before document IDs and versions were added."""
    store = get_vectorstore()
    store.index.delete(
        namespace=settings.pinecone_namespace,
        filter={"source": {"$eq": source}},
    )


def delete_obsolete_document_versions(document_id: str, current_version: str) -> None:
    store = get_vectorstore()
    store.index.delete(
        namespace=settings.pinecone_namespace,
        filter={
            "document_id": document_id,
            "document_version": {"$ne": current_version},
        },
    )
