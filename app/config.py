from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str
    PORT: int
    HOST: str
    GEMINI_API_KEY: str
    OPENAI_API_KEY: str
    GPT_VERIFICATION_MODEL: str
    GPT_GENERATIVE_MODEL: str
    TOKEN: str
    REDIS_URL: str
    REDIS_PORT: int
    BROKER: str
    BACKEND: str
    RELOAD: bool
    FLASH_TPM_LIMIT: int
    FLASH_RPM_LIMIT: int
    FLASH_LIGHT_TPM_LIMIT: int
    FLASH_LIGHT_RPM_LIMIT: int

    model_config = SettingsConfigDict(env_file=".env")


ENV_PROJECT = Settings()
