from typing import Final

GROQ_MIXTRAL_MODEL_NAME: Final[str] = "mixtral-8x7b-32768"
GROQ_LLAMA2_MODEL_NAME: Final[str] = "llama2-70b-4096"

OPENAI_GPT35_MODEL_NAME: Final[str] = "gpt-3.5-turbo"
OPENAI_GPT4_MODEL_NAME: Final[str] = "gpt-4"

CREATIVE_MODEL_TEMP: Final[float] = 0.8
FACTUAL_MODEL_TEMP: Final[float] = 0

HUB_TOOLS_PROMPT: Final[str] = "hwchase17/openai-tools-agent"

LOG_FORMAT: Final[str] = "%(asctime)s - %(process)d - %(levelname)s - %(message)s"
