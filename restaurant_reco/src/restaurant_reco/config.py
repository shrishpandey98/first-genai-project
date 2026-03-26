from __future__ import annotations

from pydantic import BaseModel


class LLMConfig(BaseModel):
    """
    Phase 0 convention only.

    Phase 0/1 does not call Grok; later phases will consume these env vars.
    """

    xai_api_key: str | None = None
    grok_model: str | None = None

