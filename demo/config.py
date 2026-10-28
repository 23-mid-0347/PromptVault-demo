"""Configuration loading for the demo."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    xai_api_key: str
    grok_model: str


def get_settings() -> Settings:
    """Load local environment configuration.

    Streamlit secrets are checked first when available, followed by environment variables.
    """
    load_dotenv()

    secret_key = ""
    secret_model = ""
    try:
        import streamlit as st

        secret_key = str(st.secrets.get("XAI_API_KEY", ""))
        secret_model = str(st.secrets.get("GROK_MODEL", ""))
    except Exception:
        # Streamlit secrets are optional outside a Streamlit runtime.
        pass

    return Settings(
        xai_api_key=secret_key or os.getenv("XAI_API_KEY", ""),
        grok_model=secret_model or os.getenv("GROK_MODEL", "grok-4.7"),
    )
