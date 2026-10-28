"""Configuration loading for the demo."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    gemini_api_key: str
    gemini_model: str


def get_settings() -> Settings:
    """Load local environment configuration.

    Streamlit secrets are checked first when available, followed by environment variables.
    """
    load_dotenv()

    secret_key = ""
    secret_model = ""
    try:
        import streamlit as st

        secret_key = str(st.secrets.get("GEMINI_API_KEY", ""))
        secret_model = str(st.secrets.get("GEMINI_MODEL", ""))
    except Exception:
        # Streamlit secrets are optional outside a Streamlit runtime.
        pass

    return Settings(
        gemini_api_key=secret_key or os.getenv("GEMINI_API_KEY", ""),
        gemini_model=secret_model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    )
