"""Session-wide test setup.

Two things have to happen before anything else imports, and both were learned
by watching tests fail for reasons unrelated to what they were testing.
"""

from __future__ import annotations

import os

import pytest

# 1. Trust the OS certificate store, BEFORE any library builds an HTTP client.
#
# `huggingface_hub` creates its session at import time using certifi's bundle.
# If `transformers` is imported first — and it is, by a skipif probe at
# collection time — injecting truststore later is too late and model downloads
# fail with a misleading "Can't load the configuration of ..." error.
# Import order matters for TLS setup in a way it usually doesn't.
import tls_trust  # noqa: E402

tls_trust.enable()

# 2. Never hit the network for model weights during tests.
#
# Even with the cross-encoder already cached, huggingface_hub revalidates over
# HTTPS on load. On a TLS-inspecting network that request fails and the error
# surfaces as a misleading "Can't load the configuration of ..." rather than a
# network error. Offline mode makes the tests deterministic and fast, and means
# a TLS problem can never masquerade as a broken model.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# 3. Tell HuggingFace where its cache is, because `~` may be unresolvable.
#
# Under pytest launched from Git Bash on Windows, HOME points at a nonexistent
# conda path and USERPROFILE/HOMEDRIVE/HOMEPATH are unset, so `Path.home()`
# raises "Could not determine home directory". huggingface_hub reports that as
# "Can't load the configuration of <model>", which reads like a missing model
# or a network failure and is neither.
if not os.environ.get("HF_HOME"):
    from pathlib import Path

    _home = os.environ.get("USERPROFILE")
    if not _home:
        _appdata = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if _appdata:  # ...\AppData\Local -> ...\  (the user profile)
            _home = str(Path(_appdata).parent.parent)
    if _home and (Path(_home) / ".cache" / "huggingface").exists():
        os.environ["HF_HOME"] = str(Path(_home) / ".cache" / "huggingface")

# If you see the reranker tests SKIP, you are almost certainly running pytest
# from Git Bash, which sets none of USERPROFILE/HOMEDRIVE/HOMEPATH. Run the
# suite from PowerShell and all 100 pass. The skip is environmental, not a
# defect — the reranker itself works (see the measurements in README.md).


@pytest.fixture(autouse=True)
def _neutral_retrieval_env(monkeypatch):
    """Run tests against documented defaults, not the developer's .env.

    `agent.providers` calls `load_dotenv()` at import, which leaks the local
    .env into os.environ for the whole session. A test asserting the default
    retriever then reads whatever that developer happens to have configured
    and fails on a perfectly correct codebase.

    Tests that care about a specific retriever set it explicitly.
    """
    monkeypatch.delenv("RETRIEVER", raising=False)
    monkeypatch.delenv("RERANK", raising=False)
    yield
