from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Enterprise Research Assistant"
    app_version: str = "0.1.0"
    environment: Literal["development", "test", "staging", "production"] = "development"
    debug: bool = False
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:3000"]
    trusted_hosts: list[str] = ["localhost", "127.0.0.1", "testserver"]
    chroma_persist_directory: str = "data/chroma"
    chroma_collection_name: str = "research_chunks"
    lexical_index_path: str = "data/lexical.db"
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"
    ingestion_max_file_size_bytes: int = 25 * 1024 * 1024
    ingestion_chunk_size: int = 512
    ingestion_chunk_overlap: int = 64
    retrieval_candidate_count: int = 20
    retrieval_final_count: int = 8
    retrieval_rrf_k: int = 60
    cohere_api_key: SecretStr | None = None
    cohere_rerank_model: str = "rerank-v4.0-pro"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash"
    answer_max_completion_tokens: int = 4096
    agent_recursion_limit: int = 4
    agent_plan_max_subquestions: int = 4
    agent_plan_max_completion_tokens: int = 8192
    evaluation_results_directory: str = "data/evaluations/results"
    evaluation_throttle_seconds: float = 20.0
    deepseek_api_key: SecretStr | None = None
    deepseek_model: str = "deepseek-v4-flash"
    evaluation_deepseek_model: str = "deepseek-v4-flash"

    @field_validator("cors_origins", "trusted_hosts", mode="before")
    @classmethod
    def parse_csv_list(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.upper()


@lru_cache
def get_settings() -> Settings:
    return Settings()
