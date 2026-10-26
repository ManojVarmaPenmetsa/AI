from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Agentic RAG"
    app_env: str = "development"

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "nvidia/nemotron-3.5-lightning:free"
    vision_provider: str = "openai_compatible"
    vision_model: str = "nvidia/nemotron-3.5-super-vl:free"
    vision_api_key: str = ""
    vision_base_url: str = ""
    vision_timeout_seconds: int = 30
    pdf_vision_max_pages: int = 8
    pdf_vision_min_image_area_ratio: float = 0.05
    pdf_vision_min_drawings: int = 80
    pdf_render_dpi: int = 150
    pdf_vision_max_pixels: int = 20_000_000
    pdf_vision_max_image_bytes: int = 12_582_912
    pdf_min_text_words: int = 25
    ocr_provider: str = "ocr_space"
    ocr_space_api_key: str = ""
    ocr_space_api_url: str = "https://api.ocr.space/parse/image"
    ocr_space_engine: str = "2"
    ocr_space_max_file_bytes: int = 1_000_000
    ocr_timeout_seconds: int = 20
    tavily_api_key: str = ""
    pinecone_api_key: str = ""
    admin_api_key: str = ""
    pinecone_index_name: str = "fde-it-support-rag"
    pinecone_namespace: str = "company-it-kb"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    top_k: int = 4
    retrieval_candidate_k: int = 12
    max_retries: int = 1

    audit_db_path: str = str(BASE_DIR / "data" / "audit.db")
    conversation_db_path: str = str(BASE_DIR / "data" / "conversations.db")
    document_registry_db_path: str = str(BASE_DIR / "data" / "document_registry.db")
    upload_dir: str = str(BASE_DIR / "uploads")
    sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb")

    model_config = SettingsConfigDict(
        env_file=(
            str(BASE_DIR / ".conda_env" / ".env"),
            str(BASE_DIR / ".env"),
        ),
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
