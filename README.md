# PromptVault Demo

A Streamlit chat and evaluation demo for measuring the trade-off between conversation-history compression, token efficiency, and answer quality.

> **Status:** scaffold / planning implementation. The integration is not considered complete until the PromptVault API has been verified against the local checkout, the app has been exercised with a real provider, and tests pass.

## Goals

- Demonstrate the same multi-turn chat using **Full history** or **PromptVault**.
- Compare Full history, Last-N turns, Sliding window, and PromptVault on the same synthetic cases.
- Keep provider-reported token usage distinct from local estimates.
- Track compression time, API latency, factual accuracy, and evidence retention.
- Export experiment records to CSV and JSONL.

## Repository boundary

This is a separate demo repository: [`23-mid-0347/PromptVault-demo`](https://github.com/23-mid-0347/PromptVault-demo). It is not the core PromptVault library repository. The demo imports the installed/local `promptvault` package; it does not copy or rebuild the existing LangGraph adapter.

## Requirements

- Python 3.11 or newer (use the Python version supported by your PromptVault library checkout).
- Access to the PromptVault package, either installed in the environment or installed from a local checkout.
- Optional Gemini API key for real chat requests.
- Synthetic test conversations only for the initial demo. Review provider data-use terms before submitting anything sensitive.

## Quick start

From this repository folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add `GEMINI_API_KEY` if you plan to use Gemini. Never commit `.env`.

Install PromptVault in the same environment using the project's current documented method. For a local sibling checkout, adapt the path:

```bash
pip install -e ../PromptValt
```

The path above is only an example; make sure it points to your actual core-library checkout and that its public API matches `demo/strategies.py`.

Run the app:

```bash
streamlit run app.py
```

Run offline tests:

```bash
pytest -q
ruff check .
```

Offline tests do not require an API key. Provider integration is only available after valid credentials and dependencies are configured.

## First-run checklist

1. [ ] Verify the PromptVault package import and public constructor/result fields against the actual core repository.
2. [ ] Configure the Gemini API key in `.env` or Streamlit secrets.
3. [ ] Start with the small included synthetic dataset.
4. [ ] Run the test case in Full history and PromptVault modes with the exact same model and settings.
5. [ ] Inspect the diagnostics and confirm local token estimates are clearly labelled as estimates.
6. [ ] Check API limits/pricing and data-use terms before scaling up the benchmark.

## Layout

```text
PromptVault-demo/
├── app.py
├── demo/
│   ├── config.py
│   ├── providers/
│   │   ├── base.py
│   │   └── gemini.py
│   ├── strategies.py
│   ├── chat_service.py
│   ├── metrics.py
│   ├── evaluation.py
│   └── datasets/
│       └── synthetic_qa.json
├── tests/
├── outputs/                 # ignored local exports
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── PromptVault_demo.md
```

## Provider and privacy note

Gemini is the first planned API adapter. Free-tier eligibility, model IDs, context limits, rate limits, pricing, and data-use terms can change. Check the provider's current official documentation before running experiments. The Google AI Studio free tier may use submitted content to improve Google's products; use only synthetic prompts for this demo unless the applicable terms have been reviewed and data owners have approved.

- Gemini setup: https://ai.google.dev/gemini-api/docs/get-started
- Gemini pricing and free-tier conditions: https://ai.google.dev/gemini-api/docs/pricing
- Gemini generate-content API: https://ai.google.dev/api/generate-content
- Streamlit chat elements: https://docs.streamlit.io/develop/api-reference/chat

## What is intentionally not implemented yet

- Additional provider adapters (OpenRouter, Groq, xAI/Grok).
- RAG/vector database or persistent multi-user storage.
- Authentication or production deployment.
- Unverified support for system, tool, image, audio, or other multimodal PromptVault messages.
- Claims that compression always improves answer quality.

See [`PromptVault_demo.md`](PromptVault_demo.md) for the phased implementation plan and evaluation rules.
