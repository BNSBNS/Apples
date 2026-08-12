"""One client, many providers.

Ollama, OpenAI and DeepSeek all speak the OpenAI wire format, so the *same*
client class and the *same* request shape work against all three. Switching
provider is a base_url and a model name — not a code path, not an abstraction
layer, not a framework. That is why the list below is a table and not a class
hierarchy: nothing about these providers differs in *behaviour*, only in values.

That is also the data-residency pattern banks need: route sensitive customer
data to the local model, everything else to a frontier model, with one flag.

The client is **async**. The agent loop runs inside an MCP session, which is an
async context; a synchronous `create()` there blocks the event loop. Harmless
under stdio with one request in flight, not harmless under streamable-http with
concurrent sessions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import AsyncOpenAI

import tls_trust

load_dotenv()

# The OpenAI SDK uses httpx, which trusts certifi rather than the OS store.
# On a TLS-inspecting network that means every call fails. See tls_trust.
_TRUSTSTORE_OK = tls_trust.enable()


@dataclass(frozen=True)
class ProviderConfig:
    """Everything that differs between providers. There is nothing else."""

    base_url: str | None  # None -> the SDK's OpenAI default
    key_env: str | None  # None -> no key required (Ollama)
    model_env: str
    default_model: str | None  # None -> must be set in .env


PROVIDERS: dict[str, ProviderConfig] = {
    "local": ProviderConfig("http://localhost:11434/v1", None, "OLLAMA_MODEL", "llama3.2:latest"),
    "openai": ProviderConfig(None, "OPENAI_API_KEY", "OPENAI_MODEL", None),
    "deepseek": ProviderConfig(
        "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "deepseek-chat"
    ),
}

# `cloud` predates the table and is baked into every CLI's choices=[...] and into
# the labels on files already in evals/results/. Keeping it as an alias costs one
# line; renaming it would break comparability with runs already recorded.
ALIASES = {"cloud": "openai"}

KINDS = sorted(PROVIDERS) + sorted(ALIASES)


@dataclass(frozen=True)
class Provider:
    name: str
    client: AsyncOpenAI
    model: str

    def __str__(self) -> str:
        return f"{self.name} ({self.model})"


def get_provider(kind: str = "local", model_override: str | None = None) -> Provider:
    """Build a configured client.

    local     -> Ollama, free and offline. No key required.
    openai    -> OpenAI. Requires OPENAI_API_KEY and OPENAI_MODEL.
    deepseek  -> DeepSeek. Requires DEEPSEEK_API_KEY.
    cloud     -> alias for openai.

    `model_override` lets the eval harness and the training data generator sweep
    models without touching .env.
    """
    resolved = ALIASES.get(kind, kind)
    config = PROVIDERS.get(resolved)
    if config is None:
        raise ValueError(f"unknown provider {kind!r}; expected one of {KINDS}")

    api_key = "not-needed"
    if config.key_env:
        api_key = os.getenv(config.key_env) or ""
        if not api_key:
            raise SystemExit(
                f"{config.key_env} not set. Copy .env.example to .env and fill it in, "
                "or run with --provider local (free, offline)."
            )
        if not _TRUSTSTORE_OK:
            print(
                "[providers] warning: truststore unavailable. On a TLS-inspecting network "
                "expect CERTIFICATE_VERIFY_FAILED. Install it: uv pip install truststore",
                flush=True,
            )

    model = model_override or os.getenv(config.model_env) or config.default_model
    if not model:
        raise SystemExit(
            f"{config.model_env} not set in .env (e.g. {config.model_env}=gpt-4o-mini)."
        )

    # Ollama ignores the key but the SDK requires a non-empty string.
    client = (
        AsyncOpenAI(base_url=config.base_url, api_key=api_key)
        if config.base_url
        else AsyncOpenAI(api_key=api_key)
    )

    # If Langfuse is configured, wrap the client so every LLM call is traced
    # regardless of which interface (dashboard, CLI, evals) made it.
    if os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse.openai import AsyncOpenAI as LangfuseAsyncOpenAI

            client = (
                LangfuseAsyncOpenAI(base_url=config.base_url, api_key=api_key)
                if config.base_url
                else LangfuseAsyncOpenAI(api_key=api_key)
            )
        except ImportError:
            pass

    return Provider(resolved, client, model)
