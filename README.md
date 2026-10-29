# PromptVault Demo

A Streamlit chat and evaluation demo for measuring the trade-off between conversation-history compression, token efficiency, and answer quality.

> **Status:** early scaffold. Do not treat the integration as complete until OpenRouter is tested with a real API call, the PromptVault API is verified against the local core-library checkout, and the tests pass.

## Goals

- Demonstrate the same multi-turn chat using **Full history** or **PromptVault**.
- Compare Full history, Last-N turns, Sliding window, and PromptVault on the same synthetic cases.
- Keep provider-reported token usage distinct from local estimates.
- Track compression time, API latency, factual accuracy, and evidence retention.
- Export experiment records to CSV and JSONL as the evaluation phase is implemented.

## Repository boundary

This is a separate demo repository: [`23-mid-0347/PromptVault-demo`](https://github.com/23-mid-0347/PromptVault-demo). It is not the core PromptVault library repository. The demo imports the installed/local `promptvault` package; it does not copy or rebuild the existing LangGraph adapter.

## Requirements

- Python 3.11 or newer (use the Python version supported by the core PromptVault checkout).
- Access to the PromptVault package, installed in this environment or installed from a local checkout.
- OpenRouter API key from [OpenRouter Keys](https://openrouter.ai/keys).
- Synthetic test conversations only for the initial demo; review the data-use terms of the selected model/provider before sharing prompts.

## Quick start

From this repository folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `OPENROUTER_API_KEY` with your key. Keep `.env` local and never commit it. The starting model is configured as:

```dotenv
OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
```

You can change the model ID in `.env` or in the Streamlit sidebar. For fair benchmark runs, record and keep the exact same model ID for every strategy in that run. Avoid `openrouter/free` auto-routing for comparisons that require a fixed model, because auto-routing can select different models across requests.

Install PromptVault in the same environment using the core project's current documented method. For a local sibling checkout, adapt the path:

```bash
python -m pip install -e ../PromptValt
```

The path above is only an example; confirm that it points to the actual core-library checkout and that the public API matches `demo/strategies.py`.

Run the app with the active virtual environment:

```bash
python -m streamlit run app.py
```

Run offline tests:

```bash
python -m pytest -q
ruff check .
```

Offline tests do not require an API key. A real provider request requires valid credentials and an available selected model.

## First-run checklist

1. [ ] Confirm `OPENROUTER_API_KEY` is set in `.env` (or Streamlit secrets).
2. [ ] Verify the selected model ID on its [OpenRouter model page](https://openrouter.ai/models?variant=free).
3. [ ] Verify the PromptVault package import, public constructor, and compression-result fields against the actual core repository.
4. [ ] Start with a short synthetic recall conversation.
5. [ ] Test the same conversation in Full history and PromptVault modes with the exact same model and generation parameters.
6. [ ] Inspect diagnostics and confirm provider usage is separated from local token estimates.
7. [ ] Check limits, provider availability, and data-use terms before scaling up the benchmark.

## Layout

```text
PromptVault-demo/
├── app.py
├── demo/
│   ├── config.py
│   ├── providers/
│   │   ├── base.py
│   │   └── openrouter.py
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

## Provider, rate-limit, and privacy notes

OpenRouter's free-model variants are rate-limited. Its current free-model page documents a limit of 20 requests per minute and 50 requests per day for `:free` variants; buying at least $10 in credits may raise the daily free-model allowance. Check the live page because limits can change. Model availability can also vary by provider/capacity, so the demo surfaces request errors rather than silently switching model IDs.

The selected [NVIDIA Nemotron 3 Ultra free endpoint](https://openrouter.ai/nvidia/nemotron-3-ultra-550b-a55b:free) advertises up to a 1M context window, but that is not a guarantee that every request at that size will succeed. The endpoint page states that requests may be logged for security and used to improve NVIDIA products. Use synthetic conversations only; do not send private histories, secrets, credentials, or proprietary information.

- OpenRouter quickstart: https://openrouter.ai/docs/quickstart
- OpenRouter free models: https://openrouter.ai/collections/free-models
- OpenRouter pricing/limits: https://openrouter.ai/pricing
- Selected model card: https://openrouter.ai/nvidia/nemotron-3-ultra-550b-a55b:free
- Streamlit chat elements: https://docs.streamlit.io/develop/api-reference/chat

## What is intentionally not implemented yet

- Fully validated PromptVault integration with the core package's actual API.
- Full automated fair-comparison UI/export workflow.
- Additional provider adapters (Gemini, Groq, xAI/Grok).
- RAG/vector database or persistent multi-user storage.
- Authentication or production deployment.
- Unverified support for system, tool, image, audio, or other multimodal PromptVault messages.
- Claims that compression always improves answer quality.

See [`PromptVault_demo.md`](PromptVault_demo.md) for the implementation plan and evaluation rules.
