import os
from typing import Annotated, Any

from dotenv import find_dotenv
from pydantic import BeforeValidator, HttpUrl, SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.core import (
    AllModelEnum,
    AnthropicModelName,
    FakeModelName,
    GroqModelName,
    HuggingFaceModelName,
    OllamaModelName,
    OpenAIModelName,
    Provider,
)
from backend.utils.url import ensure_https

ANTHROPIC_MODEL_ALIASES = {
    "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
}

HttpUrlAnnotated = Annotated[HttpUrl | None, BeforeValidator(ensure_https)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        validate_default=False,
    )
    MODE: str | None = None

    WORKING_DIRECTORY: str | None = None

    # API service title and version
    TITLE: str = "dft_agent"
    VERSION: str = "0.0.1"

    HOST: str = "0.0.0.0"
    PORT: int = 8083

    AUTH_SECRET: SecretStr | None = None

    # CORS configuration - comma-separated list of allowed origins
    # Example: "http://localhost:3000,http://localhost:8501,https://yourdomain.com"
    ALLOWED_ORIGINS: str = "http://localhost:8501,http://localhost:3000"

    # Security: Enable Python REPL tool (DANGEROUS - allows arbitrary code execution)
    # Only enable in trusted, isolated environments
    ENABLE_PYTHON_REPL: bool = False

    OPENAI_API_KEY: SecretStr | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None
    CEREBRAS_API_KEY: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None
    HF_API_KEY: SecretStr | None = None
    OLLAMA_MODEL: str | None = None
    OLLAMA_BASE_URL: str | None = None
    USE_FAKE_MODEL: bool = False

    # Anthropic feature flags (comma-separated, e.g., "tools-2024-10-22")
    ANTHROPIC_BETA_TAGS: str | None = None

    # If DEFAULT_MODEL is None, it will be set in model_post_init
    DEFAULT_MODEL: AllModelEnum | None = OpenAIModelName.GPT_4O  # type: ignore[assignment]
    AVAILABLE_MODELS: set[AllModelEnum] = set()  # type: ignore[assignment]

    MP_API_KEY: SecretStr | None = None
    ASTA_KEY: SecretStr | None = None

    # Database settings
    DATABASE_URL: str = ""

    def model_post_init(self, __context: Any) -> None:
        api_keys = {
            Provider.ANTHROPIC: self.ANTHROPIC_API_KEY,
            Provider.OPENAI: self.OPENAI_API_KEY,
            Provider.GROQ: self.GROQ_API_KEY,
            Provider.HUGGINGFACE: self.HF_API_KEY,
            Provider.OLLAMA: self.OLLAMA_MODEL,
            Provider.FAKE: self.USE_FAKE_MODEL,
        }
        active_keys = [k for k, v in api_keys.items() if v]
        if not active_keys:
            raise ValueError("At least one LLM API key must be provided.")

        for provider in active_keys:
            match provider:
                case Provider.ANTHROPIC:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AnthropicModelName.CLAUDE_SONNET_4_5
                    self.AVAILABLE_MODELS.update(set(AnthropicModelName))
                case Provider.OPENAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenAIModelName.GPT_4O
                    self.AVAILABLE_MODELS.update(set(OpenAIModelName))
                case Provider.GROQ:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = GroqModelName.LLAMA_31_8B
                    self.AVAILABLE_MODELS.update(set(GroqModelName))
                case Provider.HUGGINGFACE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = HuggingFaceModelName.DEEPSEEK_R1
                    self.AVAILABLE_MODELS.update(set(HuggingFaceModelName))
                case Provider.OLLAMA:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OllamaModelName.OLLAMA_GENERIC
                    self.AVAILABLE_MODELS.update(set(OllamaModelName))
                case Provider.FAKE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = FakeModelName.FAKE
                    self.AVAILABLE_MODELS.update(set(FakeModelName))
                case _:
                    raise ValueError(f"Unknown provider: {provider}")

    @computed_field
    @property
    def BASE_URL(self) -> str:
        return f"http://{self.HOST}:{self.PORT}"

    @computed_field
    @property
    def ROOT_PATH(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def is_dev(self) -> bool:
        return self.MODE == "dev"

    @property
    def anthropic_betas(self) -> tuple[str, ...]:
        if not self.ANTHROPIC_BETA_TAGS:
            return ()
        return tuple(
            tag.strip()
            for tag in self.ANTHROPIC_BETA_TAGS.split(",")
            if tag.strip()
        )

    @field_validator("DEFAULT_MODEL", mode="before")
    @classmethod
    def _upgrade_default_model(cls, value: Any) -> Any:
        if isinstance(value, str):
            return ANTHROPIC_MODEL_ALIASES.get(value, value)
        return value

    def resolve_anthropic_model_name(self, model_name: str | None) -> str | None:
        if isinstance(model_name, str):
            return ANTHROPIC_MODEL_ALIASES.get(model_name, model_name)
        return model_name


settings = Settings()
