# PromptVault Demo — missing-code patch

This patch adds the modules missing from the current `23-mid-0347/PromptVault-demo` repository:

- `demo/providers/base.py`
- `demo/strategies.py`
- `demo/chat_service.py`
- `demo/metrics.py`
- `demo/evaluation.py`
- `demo/datasets/synthetic_qa.json`
- package marker files and an additional offline test file

## Apply safely

1. Open your local `promptvault-demo` repository folder.
2. Run `git status --short` first. Save or commit unrelated local changes before applying the patch.
3. Extract this ZIP into the repository root. Use the option that does **not** overwrite existing files, since the repo already has tests and may contain local edits.
4. Run the checks below from the activated project virtual environment.

```bash
python -m pytest -q
ruff check .
python -m streamlit run app.py
```

The first two commands should not need a network connection or an OpenRouter API key. The app needs `OPENROUTER_API_KEY` in the local ignored `.env` file for real model calls.

## Important PromptVault integration limitation

The adapter uses the public API described in `PromptVault_demo.md` V2:

```python
from promptvault import PromptVault

vault = PromptVault(
    token_budget=2000,
    semantic_weight=0.6,
    lexical_weight=0.4,
    preserve_recent_n=2,
)
result = vault.compress(history, query)
```

The core PromptVault checkout was not present in this demo repository, so confirm the constructor, message format and `result` attributes against the actual core package. If import or compression fails, this patch raises an actionable error instead of pretending to compress or silently sending full history.

The OpenRouter integration has not been live-smoke-tested by this patch. After the offline tests pass, verify Full history and PromptVault with the same short, fictional recall test and fixed model ID. Do not commit `.env` or real API keys.
