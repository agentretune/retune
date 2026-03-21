"""Server configuration."""

from pydantic_settings import BaseSettings


class ServerSettings(BaseSettings):
    model_config = {"env_prefix": "AUTOMLAGENT_", "env_file": ".env", "extra": "ignore"}

    host: str = "0.0.0.0"
    port: int = 8000
    storage_path: str = "automlagent.db"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]


server_settings = ServerSettings()
