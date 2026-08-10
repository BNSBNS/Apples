"""Make Python's TLS trust the OS certificate store.

Cross-cutting infrastructure, deliberately kept out of both `server/` and
`agent/` because it belongs to neither.

The problem, which shows up on most corporate networks: a TLS-inspecting proxy
presents certificates signed by a private root CA. That CA is installed in the
OS certificate store. Python libraries that bundle their own trust roots —
certifi, and therefore `httpx`, `requests` and `huggingface_hub` — have never
heard of it, and every HTTPS call dies with:

    CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate

`truststore` redirects Python's `ssl` module at the OS store, where the CA
already is. **Verification stays fully on.** This is the opposite of
`verify=False`, which would disable the check that is working correctly.

This project hit the same root cause four separate times, each with a different
symptom and a different fix:

    conda     failed to reach repo.anaconda.com   -> `conda create --offline`
    uv        "invalid peer certificate"          -> `uv pip --system-certs`
    openai    APIConnectionError                  -> truststore (here)
    HF hub    "Can't load the configuration of"   -> truststore (here)

The HuggingFace one is the instructive one: a bare `urllib` probe to
huggingface.co *succeeded*, which made the network look fine. `huggingface_hub`
uses its own HTTP client with its own trust bundle, so reachability tested
through the wrong library proves nothing. Test through the library that is
actually failing.

Import this module before importing anything that makes HTTPS calls.
"""

from __future__ import annotations

import os

_INJECTED: bool | None = None

# Environment variables that name a CA bundle. Every HTTP client in the stack
# reads at least one of them.
_CA_ENV_VARS = ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")


def drop_broken_ca_env() -> list[str]:
    """Unset CA-bundle env vars that point at paths which do not exist.

    A dangling CA path is strictly worse than no CA path: clients pass it
    straight to `ssl.SSLContext.load_verify_locations`, which raises
    `FileNotFoundError` at *client construction* — before any request, and
    regardless of whether the connection uses TLS at all.

    The symptom is bewildering: creating an OpenAI client pointed at
    `http://localhost:11434` (plain HTTP, no TLS anywhere) dies with
    `FileNotFoundError: [Errno 2] No such file or directory` from deep inside
    httpx's SSL setup.

    How it happens here: `conda create --offline` builds an environment without
    the `ca-certificates` package, but conda's activation script still exports
    `SSL_CERT_FILE=<env>/ssl/cacert.pem`. The variable is set; the file was
    never installed. Only shells that ran `conda activate` are affected, which
    is why it reproduces for a user and not for tooling that calls the env's
    python directly.

    Dropping the variable is safe: clients fall back to certifi, and
    `inject_into_ssl()` has already routed trust decisions to the OS store.
    """
    dropped = []
    for var in _CA_ENV_VARS:
        path = os.environ.get(var)
        if path and not os.path.exists(path):
            del os.environ[var]
            dropped.append(f"{var}={path}")
    return dropped


def enable() -> bool:
    """Point Python's TLS at the OS trust store. Idempotent; safe if absent."""
    global _INJECTED
    if _INJECTED is not None:
        return _INJECTED

    for entry in drop_broken_ca_env():
        print(f"[tls] ignoring broken CA bundle path: {entry}", flush=True)

    try:
        import truststore

        truststore.inject_into_ssl()
        _INJECTED = True
    except Exception:
        _INJECTED = False
    return _INJECTED


def warn_if_unavailable(what: str) -> None:
    if not enable():
        print(
            f"[tls] truststore unavailable — {what} may fail with "
            "CERTIFICATE_VERIFY_FAILED on a TLS-inspecting network. "
            "Install it: uv pip install truststore",
            flush=True,
        )
