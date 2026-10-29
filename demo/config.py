"""Configuration loading for the demo."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


@dataclass(frozen=True, slots=True)
class Settings:
    openrouter_api_key: str
    openrouter_model: str


def get_settings() -> Settings:
    """Load local environment configuration.

    Streamlit secrets are checked first when available, followed by environment variables.
    """
    load_dotenv()

    secret_key = ""
    secret_model = ""
    try:
        import streamlit as st

        secret_key = str(st.secrets.get("OPENROUTER_API_KEY", ""))
        secret_model = str(st.secrets.get("OPENROUTER_MODEL", ""))
    except Exception:
        # Streamlit secrets are optional outside a Streamlit runtime.
        pass

    return Settings(
        openrouter_api_key=secret_key or os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_model=(
            secret_model or os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
        ),
    )
