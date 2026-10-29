"""PromptVault Streamlit demo entry point."""

from __future__ import annotations

import streamlit as st
from demo.config import get_settings
from demo.providers.openrouter import OpenRouterProvider

from demo.chat_service import prepare_history
from demo.strategies import StrategyName

st.set_page_config(page_title="PromptVault Demo", page_icon="🧠", layout="wide")

st.title("PromptVault Demo")
st.caption(
    "Explore the trade-off between conversation-history size and reliable recall. "
    "Use synthetic data only with free hosted model endpoints."
)

settings = get_settings()

with st.sidebar:
    st.header("Run settings")
    mode = st.radio(
        "History strategy",
        options=[StrategyName.FULL_HISTORY.value, StrategyName.PROMPTVAULT.value],
        format_func=lambda value: "Full history" if value == "full_history" else "PromptVault",
    )
    model_id = st.text_input("OpenRouter model ID", value=settings.openrouter_model)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    max_output_tokens = st.number_input(
        "Maximum output tokens", min_value=32, max_value=2048, value=256, step=32
    )
    token_budget = st.number_input(
        "PromptVault token budget", min_value=128, max_value=16000, value=2000, step=128
    )
    preserve_recent_n = st.number_input(
        "Preserve recent exchanges", min_value=0, max_value=10, value=2, step=1
    )
    st.divider()
    st.caption("API keys are read from `.env` or Streamlit secrets; they are not shown here.")
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.diagnostics = None
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "diagnostics" not in st.session_state:
    st.session_state.diagnostics = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask something, or plant a fact for a later recall question")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    api_key = settings.openrouter_api_key
    if not api_key:
        with st.chat_message("assistant"):
            st.error(
                "OpenRouter API key not found. Add `OPENROUTER_API_KEY` to `.env` or "
                "`.streamlit/secrets.toml`, then restart the app. Get a key at "
                "https://openrouter.ai/keys."
            )
    else:
        try:
            prepared = prepare_history(
                history=st.session_state.messages,
                strategy=StrategyName(mode),
                token_budget=int(token_budget),
                preserve_recent_n=int(preserve_recent_n),
            )
            provider = OpenRouterProvider(api_key=api_key, model_id=model_id.strip())
            result = provider.generate(
                prepared.messages,
                temperature=float(temperature),
                max_output_tokens=int(max_output_tokens),
            )
            st.session_state.messages.append({"role": "assistant", "content": result.text})
            st.session_state.diagnostics = {
                "strategy": mode,
                "provider": result.provider,
                "model_id": result.model_id,
                "estimated_prepared_tokens": prepared.estimated_tokens,
                "compression_seconds": prepared.compression_seconds,
                "request_latency_seconds": result.request_latency_seconds,
                "provider_input_tokens": result.input_tokens,
                "provider_output_tokens": result.output_tokens,
                "finish_reason": result.finish_reason,
                "original_tokens": prepared.original_tokens,
                "compressed_tokens": prepared.compressed_tokens,
                "compression_ratio": prepared.compression_ratio,
                "notes": prepared.notes,
            }
            with st.chat_message("assistant"):
                st.markdown(result.text)
        except Exception as exc:  # Surface an actionable error without exposing credentials.
            with st.chat_message("assistant"):
                status_code = getattr(exc, "status_code", None)
                if status_code == 401:
                    st.error("OpenRouter rejected the API key. Check OPENROUTER_API_KEY.")
                elif status_code == 402:
                    st.error(
                        "OpenRouter reports insufficient credits or access for this request. "
                        "Check your account balance and ensure you selected a free model variant."
                    )
                elif status_code == 429:
                    st.error(
                        "OpenRouter or the upstream model rate limit was reached. Wait before "
                        "retrying; failed attempts may still count toward free-tier limits."
                    )
                else:
                    st.error(f"The OpenRouter request failed: {exc}")

if st.session_state.diagnostics:
    with st.expander("Latest request diagnostics", expanded=False):
        st.json(st.session_state.diagnostics)

st.divider()
st.info(
    "Local token counts use a rough character-based estimate, not the provider tokenizer. "
    "Provider-reported usage is shown separately when available. The free Nemotron endpoint "
    "states that requests may be logged and used to improve NVIDIA products; use synthetic "
    "test conversations only. PromptVault integration fields still need verification against "
    "your checked-out core library."
)
