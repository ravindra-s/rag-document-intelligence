from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Paths
    project_root: Path = Path(__file__).resolve().parents[1]
    data_raw_dir: Path = project_root / "data" / "raw"
    data_processed_dir: Path = project_root / "data" / "processed"

    # Logging
    log_level: str = "INFO"

    # Models
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    #llm_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    #llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
