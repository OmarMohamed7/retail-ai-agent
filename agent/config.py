"""
Configuration management using Pydantic.
"""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentConfig(BaseSettings):
    """Agent configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(
        default="Retail AI Agent", description="Name of the application"
    )

    # Paths
    docs_dir: Path = Field(
        default=Path("docs/"), description="Directory containing documentation files"
    )
    db_path: Path = Field(
        default=Path("data/northwind.sqlite"), description="Path to the SQLite database"
    )

    # Model settings
    ollama_host: str = Field(
        default="http://localhost:11434", description="Ollama server host URL"
    )
    model_name: str = Field(default="phi3.5:mini", description="Model name to use")
    temperature: float = Field(
        default=0.1, description="Temperature for model generation"
    )
    max_tokens: int = Field(
        default=2000, description="Maximum tokens for model generation"
    )

    # Retrieval settings
    chunk_size: int = Field(default=300, description="Chunk size in words")
    top_k: int = Field(default=3, description="Number of top results to retrieve")

    # Agent settings
    max_repairs: int = Field(default=2, description="Maximum number of repair attempts")

    # Cost approximation
    cost_multiplier: float = Field(
        default=0.7, description="Cost multiplier for approximation"
    )

    @field_validator("docs_dir", "db_path", mode="before")
    @classmethod
    def validate_paths(cls, v):
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("temperature", "cost_multiplier", mode="before")
    @classmethod
    def validate_float(cls, v):
        """Ensure float values are properly converted."""
        if isinstance(v, str):
            return float(v)
        return v

    @field_validator("max_tokens", "chunk_size", "top_k", "max_repairs", mode="before")
    @classmethod
    def validate_int(cls, v):
        """Ensure int values are properly converted."""
        if isinstance(v, str):
            return int(v)
        return v
