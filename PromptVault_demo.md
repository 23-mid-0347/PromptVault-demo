# PromptVault Demo — Implementation Plan

**Document version:** V1  
**Created:** 2026-10-28  
**Project repository:** https://github.com/23-mid-0347/PromptVault-demo  
**Python package/import:** `promptvault`  
**App framework:** Streamlit  
**Status:** Planning — do not treat the demo or evaluation as implemented until the code is committed and the tests are observed passing.

---

## 1. Purpose

Build a small, usable Streamlit chat application that demonstrates PromptVault compressing older conversation history before it is sent to a cloud LLM API. Then grow the same application into a diagnostic and benchmarking dashboard that compares PromptVault against ordinary conversation-history strategies.

The demo must answer two questions with measured evidence:

1. **Efficiency:** How many input tokens does PromptVault save compared with sending the full conversation?
2. **Quality:** Can the model still correctly answer questions that depend on facts from older messages after the history has been compressed?

The demo must not assume PromptVault is better. It should expose cases where compression helps, has little effect, or loses relevant information.

## 2. Current project context

The existing PromptVault library already has its core compression pipeline and a LangGraph adapter. The latest handoff records the following checks as passing:

```text
ruff check src/promptvault/integrations/langgraph.py tests/test_langgraph_integration.py
# All checks passed!

pytest -q -m "not integration"
# 92 passed, 11 deselected

pytest -q tests/test_langgraph_integration.py -m integration
# 1 passed, 9 deselected
```

The successful GitHub Actions run recorded in the handoff is [run 40439143067](https://github.com/23-mid-0347/PromptVault-demo/actions/runs/40439143067). Re-check the repository before implementation because the local and remote state may have changed since this document was written.

The main public API planned for the demo is:

```python
from promptvault import PromptVault

vault = PromptVault(
    token_budget=2000,
    semantic_weight=0.6,
    lexical_weight=0.4,
    preserve_recent_n=2,
)

result = vault.compress(history, query)
# result.messages
# result.original_tokens
# result.compressed_tokens
# result.compression_ratio
# result.tier_assignment
# result.dropped_turn_ids
# result.merged_groups
```

Confirm the actual names and result-field semantics against the checked-out implementation before relying on them. Do not rebuild the LangGraph adapter as part of this task.

## 3. Scope and non-goals

### In scope

- A working Streamlit chat interface.
- A configurable cloud LLM provider, starting with the Gemini API free tier where eligible.
- A provider abstraction so a second API can be added without rewriting chat logic.
- A switch between full conversation history and PromptVault-compressed history.
- An automated comparison runner for multiple history strategies.
- Token, latency, and answer-quality measurements.
- Exportable run records (CSV or JSONL) and visible diagnostics.
- Repeatable synthetic test conversations with known answers.
- Documentation for setup, API keys, limitations, and result interpretation.

### Not in the first version

- A production multi-user service, authentication, or database-backed user accounts.
- A vector database or a separate RAG knowledge base.
- Fine-tuning models.
- A claim that PromptVault improves every model or every conversation.
- Automatically routing every comparison to a different model. A valid comparison must hold the model constant within a run.
- A dependency on local LLM inference. Local models may be useful later for offline experiments, but they are not required for this demo.
- Support for all LangGraph message types through the current adapter. Its documented first scope is text-only user/assistant content; system, tool, and multimodal message preservation needs separate work.

## 4. LLM provider strategy

### Recommended starting point: Gemini API

Use the official Gemini API as the first provider **if a suitable model is available under the account's free-tier limits**. Select a model with a context window large enough for the test being run, and store the exact model ID in every experiment record. Do not assume that all Gemini models share the same context window or free quota.

Official references:

- Gemini API pricing and free-tier conditions: https://ai.google.dev/gemini-api/docs/pricing
- Gemini API setup: https://ai.google.dev/gemini-api/docs/get-started

Important privacy note: Google states that content submitted under the free tier may be used to improve its products. Use **synthetic conversations only** in this project unless the applicable terms have been checked and the data owner has approved the use.

### Alternative providers

| Provider | Role in this plan | Notes |
|---|---|---|
| **Gemini API** | First integration | Has a free tier for eligible models, subject to model- and account-specific limits. Best first option for a low-cost prototype. |
| **OpenRouter free models** | Optional second integration | Offers API access to free models, with limits on free requests. Model context length, availability, and provider behavior vary, so log the exact model and provider for every run. See https://openrouter.ai/pricing. |
| **Groq API** | Optional later integration | Rate limits vary by model/account and can change. Check the account's current limits before using it for a benchmark. See https://console.groq.com/docs/rate-limits. |
| **Grok / xAI API** | Optional paid comparison | The Grok chatbot's free access does not mean API calls are free. The xAI API is usage-billed; see https://docs.x.ai/developers/pricing and https://x.ai/api. |

**Do not hard-code an API key or assume a free API is unlimited.** The selected model and its current pricing/rate limits must be checked before running a large benchmark. The demo must show a readable message when quota, rate limit, or billing blocks a request.

### Provider abstraction

Keep provider-specific code behind a small interface, for example:

```python
class ChatProvider:
    def generate(self, messages: list[dict], *, temperature: float,
                 max_output_tokens: int) -> "GenerationResult":
        ...
```

`GenerationResult` should standardize at least:

- response text;
- provider name;
- exact model ID;
- input-token usage reported by the provider, if returned;
- output-token usage reported by the provider, if returned;
- request latency;
- finish/stop reason, if available;
- a safe, user-displayable error when a request fails.

Use the provider's actual token usage as the authoritative API-usage measurement when available. Keep PromptVault's token estimate separate. Do not silently treat estimates as actual provider usage.

## 5. User experience and screens

Use Streamlit because the first deliverable is a single-user chat and diagnostics interface, not a complex frontend. Streamlit supplies chat elements such as `st.chat_message` and `st.chat_input`; see https://docs.streamlit.io/develop/api-reference/chat.

### Screen A — Chat Demo (first visible milestone)

The user can:

1. Choose the provider and model.
2. Select **Full history** or **PromptVault** mode.
3. Chat naturally over multiple turns.
4. Ask a follow-up that relies on an earlier fact.
5. Expand a diagnostics panel to inspect context size, token usage, compression time, request latency, and whether PromptVault was enabled.
6. Reset the conversation and start a clean run.

The visible answer should remain the main focus. Put diagnostic details in an expandable panel so the chatbot is not overwhelmed by charts at first.

### Screen B — Compare Strategies

Run the same prepared conversation and question against these strategies:

1. **Full history (control):** send all prior conversation messages.
2. **Last-N turns:** send only the most recent N turns/exchanges.
3. **Sliding window / token-bounded history:** keep recent messages within a specified context budget.
4. **PromptVault:** use `PromptVault.compress(history, query)` and send the returned messages.

All strategies in a comparison must use the same provider, exact model ID, system prompt, question, output limit, and temperature. Run strategies independently so a prior answer cannot leak into the next strategy's conversation state.

### Screen C — Diagnostics Dashboard (later phase, built into the same app)

Display aggregate run results:

- input tokens before and after history preparation;
- actual API input tokens (when returned by the provider);
- token savings and percentage savings;
- PromptVault's internal compression ratio;
- answer accuracy / fact-recall rate;
- compression time, API latency, total latency;
- estimated API cost where pricing data is known;
- results by token budget and conversation length;
- failures and skipped runs;
- export buttons for CSV and JSONL.

The dashboard must clearly label **estimated** versus **provider-reported** values.

## 6. History and test data design

Create a small version-controlled synthetic dataset with gold facts and questions. Avoid private, personal, or proprietary transcripts.

Each test case should contain:

- stable `case_id`;
- conversation messages (`role`, `content`);
- the follow-up question;
- expected answer(s) or accepted answer variants;
- which earlier message(s) contain the supporting fact;
- a category such as `direct_recall`, `paraphrase`, `distractor`, `updated_fact`, or `unanswerable`;
- optional notes for human review.

### Required test categories

1. **Direct recall:** e.g. “My preferred circuit is Silverstone.” Later ask which circuit was mentioned.
2. **Paraphrased recall:** express a fact differently in the later question.
3. **Distractors:** add irrelevant weather, food, hobby, and travel messages between the important fact and the question.
4. **Middle-of-history fact:** place the required fact far from the newest turns.
5. **Updated fact:** a user changes a previously stated preference. The system should use the latest valid statement.
6. **Unanswerable question:** the conversation does not contain the answer. The expected behavior is to admit that the information was not provided rather than invent it.
7. **Long conversation:** lengthen the history gradually while preserving known gold facts.

Start with perhaps 10–20 small, hand-checkable cases. Add longer synthetic contexts only after the basic test is repeatable. Make dataset size and length configurable instead of assuming that every free model accepts very large prompts.

### Context-length sweep

Use controlled synthetic histories that gradually increase in size, for example approximately 2k, 8k, 16k, and 32k estimated tokens where the selected model supports them. These are **experiment targets, not guaranteed supported limits**. Check model context limits, rate limits, and token quota before running each size. If a size cannot run, record it as skipped with the reason rather than silently shortening it.

## 7. Parameters and recommended defaults

The first UI should expose only settings that affect the experiment meaningfully. Advanced PromptVault weights can go in an expandable section.

| Parameter | Initial default / candidates | Reason and validation |
|---|---|---|
| `provider` | Gemini | Start with one provider to reduce engineering complexity; keep it replaceable. |
| `model_id` | Selected from currently available models | Model IDs and free-tier eligibility change. Log the exact ID; don't silently switch models in the middle of a run. |
| `history_strategy` | `promptvault` in chat screen; all strategies in compare screen | Makes compression behavior easy to demonstrate while retaining a full-history control. |
| `token_budget` | 2,000 for the first demo; benchmark sweep 512, 1,024, 2,000, 4,000 | Shows how the quality/size trade-off changes. Check the actual PromptVault implementation and effective budget before finalizing. |
| `preserve_recent_n` | 2 | Preserves recent context as a safety net; test whether it changes answer quality and size. |
| `semantic_weight` | 0.6 | Existing design default; keep fixed for initial comparisons. |
| `lexical_weight` | 0.4 | Existing design default; keep fixed for initial comparisons. |
| `temperature` | 0 when supported | Reduces response variability for factual evaluation; does not guarantee deterministic provider output. Keep identical across strategies. |
| `max_output_tokens` | 256 for the factual QA benchmark | Keeps answers short and costs controlled; increase for free-form chat only if required. |
| `system_prompt` | One constant, shared by all strategies | Avoids introducing prompt differences into the experiment. Example: answer from the supplied conversation, and say when the answer is absent. |
| `repeat_count` | 1 initially; 3+ for selected cases later | One run is enough for smoke tests; repeated runs help estimate output variability. |
| `dataset_id` | Versioned synthetic dataset | Supports reproducibility and comparisons between code changes. |

Do not let the UI quietly change multiple parameters at once. Every run record must include the resolved configuration.

## 8. Benchmark metrics and definitions

### A. Token efficiency

Measure both the prepared history and the complete request context where possible. Keep system prompts and current questions accounted for consistently.

Let:

- `T_full` = provider-reported input tokens for the full-history control, when available;
- `T_strategy` = provider-reported input tokens for the selected strategy, when available.

Then:

```text
input_tokens_saved = T_full - T_strategy
input_token_savings_pct = 100 * (T_full - T_strategy) / T_full
```

Handle `T_full == 0` safely and mark missing provider usage as unavailable rather than inventing a value. For the initial comparison dashboard, also report estimated tokens using the same local counter for every strategy. Clearly distinguish local estimates from provider-reported usage.

Report PromptVault's own `original_tokens`, `compressed_tokens`, and `compression_ratio` separately. Verify the exact semantics of these fields from the implementation before labelling the UI.

### B. Answer quality

Use deterministic, hand-checkable gold answers for the initial dataset:

- **Exact/normalized match:** suitable for short answers such as circuit names or a preference.
- **Accepted-answer match:** allows declared aliases or variants.
- **Gold fact accuracy:** proportion of questions answered correctly against the dataset's expected answer.
- **Unanswerable abstention rate:** proportion of unanswerable cases where the model does not invent an answer.
- **Evidence retention / fact recall:** whether the required supporting fact remains available in the prepared PromptVault messages. Measure this separately from whether the model happens to answer correctly.

A model can answer incorrectly even when the fact is retained, or answer correctly by chance even when it is absent. Therefore, **context retention and generated-answer accuracy must be separate metrics**.

Do not use an LLM judge as the sole quality metric in the first version. A judge may be added later as a secondary metric, with judge model, prompt, and uncertainty documented.

### C. Latency

Record separate timings for:

- PromptVault compression time;
- provider request latency;
- total wall-clock time for the strategy.

This allows us to check whether token savings translate to an acceptable end-to-end latency trade-off.

### D. Cost

Where current model pricing is known, estimate request cost from actual input/output tokens and the relevant published rates. Store the pricing source and date used. Do not label the estimate as an invoice or guarantee. For a free-tier call, report monetary API cost as `$0 for the currently eligible quota` only if the account and model are actually on the free tier; also show that rate limits and terms apply.

### E. Rate-distortion analysis

Repeat the same dataset at several PromptVault token budgets, then plot:

- X-axis: compression ratio or percentage of input tokens saved;
- Y-axis: gold-answer accuracy and/or evidence-retention rate.

This is the central evaluation artifact: it shows the savings/quality trade-off as the budget tightens. Show token savings and correctness together; a high compression ratio alone does not prove the system is useful.

## 9. Experiment rules for fair comparisons

1. Use the same provider and exact model ID within a comparison run.
2. Use the same synthetic transcript, current question, system prompt, output limit, and temperature.
3. Keep each strategy's history independent; do not let one strategy see another strategy's answer.
4. Do not switch provider or model automatically after an error without marking the run as a different configuration.
5. Save provider-reported usage, when returned, along with local estimates. Never substitute one for the other without a label.
6. Record API errors, quota errors, timeouts, and unsupported-context failures. Do not silently drop failed cases from summary statistics.
7. If repeated runs are used, report the number of successful and failed attempts and the mean/median where appropriate.
8. Save configuration and dataset version with every run so results can be reproduced.
9. Compare answer quality both on recent facts and old facts; the tool's specific value proposition relates to useful older context, not only the newest message.
10. Start at modest context sizes. Scale up only after the basic pipeline works and the selected provider's current limits have been checked.

## 10. Proposed project layout

Confirm the repository's current conventions before adding files. A possible layout is:

```text
PromptVault-demo/
├── app.py                         # Streamlit entry point
├── demo/
│   ├── __init__.py
│   ├── config.py                   # resolved settings / secret lookup
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py                 # common provider interface/result
│   │   └── gemini.py               # first API implementation
│   ├── strategies.py               # full, last-N, sliding-window, PromptVault
│   ├── chat_service.py             # prepare messages and call provider
│   ├── metrics.py                  # token, quality, latency, cost calculations
│   ├── evaluation.py               # run dataset cases and gather records
│   └── datasets/
│       └── synthetic_qa.json
├── tests/
│   ├── test_demo_metrics.py
│   ├── test_demo_strategies.py
│   └── test_demo_evaluation.py
├── outputs/                        # local CSV/JSONL results; do not commit runs by default
├── .env.example                    # variable names only, no real secrets
├── .gitignore                      # ignore .env and local experiment outputs
└── PromptVault_demo.md              # this plan
```

Keep the UI, provider request, context strategy, and metric functions separate. Pure metric and strategy logic should be testable without network access. Only the provider integration tests should need an API key and network access.

## 11. Configuration and key management

- Read keys from environment variables or Streamlit secrets.
- For local development, use an ignored `.env` file and a loader already accepted by the repository.
- `.env.example` contains placeholders only, such as `GEMINI_API_KEY=`.
- Never put an API key in source code, screenshots, test fixtures, error messages, or committed output files.
- Never log raw provider request headers or secrets.
- Use only synthetic prompts in the initial demo because provider data-use terms differ by tier.
- Keep model selection, provider selection, and API key fields separate. A missing key should create a clear setup message rather than crash the entire app.

## 12. Implementation phases and checklist

### Phase 0 — Confirm baseline (before coding)

- [ ] Pull the latest `main` and inspect current `pyproject.toml`, public `PromptVault` API, and README.
- [ ] Re-run the documented unit, integration, and Ruff commands.
- [ ] Confirm how `CompressionResult` token fields are counted and whether `compress()` expects the current question separately from history.
- [ ] Decide which Gemini model is available on the account and fits the first test set. Confirm its current context and rate limits.

**Exit condition:** clean baseline tests and one selected model/provider documented. Do not proceed on an assumed API shape.

### Phase 1 — Minimum working chat

- [ ] Create a small Streamlit app using `st.chat_message` and `st.chat_input`.
- [ ] Implement the first provider adapter.
- [ ] Allow a user to choose Full history or PromptVault mode.
- [ ] Feed only supported text user/assistant roles to PromptVault.
- [ ] Preserve the newest user question in the actual model request.
- [ ] Add reset-chat behavior and readable API/rate-limit error handling.
- [ ] Show prepared-context estimates and compression latency in an expander.
- [ ] Add an end-to-end synthetic test: plant a fact, add distractors, then ask about the fact.

**Exit condition:** the app can complete a multi-turn API conversation in both modes, and the smoke test demonstrates the fact-recall question end to end. This is the first real demo milestone.

### Phase 2 — Fair comparison engine

- [ ] Implement Full history, Last-N, Sliding window, and PromptVault strategies.
- [ ] Add the small gold-answer synthetic dataset.
- [ ] Call the exact same model with the same settings per strategy/case.
- [ ] Capture provider usage, local estimates, output, correctness, latency, and failures.
- [ ] Write a small pure test suite for the formulas and strategy message construction.
- [ ] Export results to CSV and JSONL.

**Exit condition:** a repeatable command/UI action creates a run file and makes the control vs. PromptVault comparison inspectable.

### Phase 3 — Diagnostics dashboard

- [ ] Add summary metrics and per-case comparison table.
- [ ] Add token savings and quality charts.
- [ ] Add context-length and token-budget filters.
- [ ] Add the rate-distortion plot.
- [ ] Separate compression time from model request time.
- [ ] Mark failed/skipped experiments visibly.
- [ ] Add a note that small samples are exploratory, not a general benchmark claim.

**Exit condition:** a reviewer can see where tokens were saved, whether the gold answer survived, and how that changes with the budget.

### Phase 4 — Hardening and docs

- [ ] Test missing API key, rate limit, timeout, malformed response, and unsupported message content.
- [ ] Verify secrets are not shown or saved.
- [ ] Expand README with installation, setup, usage, known limitations, and screenshots.
- [ ] Add dependency instructions for the demo separate from the core library if practical.
- [ ] Run Ruff and the existing test suite plus demo unit tests.
- [ ] Confirm no generated credentials or personal conversation data are tracked.
- [ ] Record current limitations and exact benchmark setup.

**Exit condition:** a new developer can install and run the demo by following the README, and automated checks pass.

### Phase 5 — Optional follow-up work

- [ ] Add OpenRouter or Groq as a second provider.
- [ ] Compare additional models as separate experiments.
- [ ] Increase dataset size and repeat selected cases.
- [ ] Add a secondary LLM judge only after gold-answer scoring is stable.
- [ ] Compare more context lengths and publish a small benchmark report.
- [ ] Consider public hosting only after key/secrets and free-tier data-use implications are reviewed.

## 13. Minimum acceptance criteria

The demo is not ready to call complete until all of these are true:

- [ ] The Streamlit app runs from the documented command.
- [ ] A real API-backed multi-turn conversation works in Full history mode.
- [ ] The same conversation works through PromptVault mode.
- [ ] An old gold fact can be tested with distractor messages in the middle.
- [ ] At least Full history and PromptVault can be compared on the same case and model.
- [ ] Results show input tokens or explicitly indicate when exact usage is unavailable.
- [ ] Token savings and correctness are calculated by tested functions.
- [ ] PromptVault compression metadata is shown separately from actual provider usage.
- [ ] API keys are not committed or written to result logs.
- [ ] Failures and unsupported configurations are visible, not silently hidden.
- [ ] Existing PromptVault and new demo tests pass.

A successful demo is evidence that the integration works end to end. It is **not** by itself evidence that compression improves overall performance. That claim requires the benchmark results.

## 14. Known risks and how to handle them

| Risk | Handling |
|---|---|
| Free quota or model availability changes | Choose the model at setup time, record the ID, check current provider limits, and surface quota errors. |
| Provider token usage does not match local token estimates | Keep actual usage and estimated counts in separate fields. Use actual usage for savings when the provider returns it. |
| PromptVault saves tokens but loses the gold fact | Measure evidence retention and answer accuracy; test several token budgets and show the trade-off. |
| Model outputs vary between runs | Start at temperature 0 where available, use short gold-answer questions, and repeat selected cases later. |
| Large context tests exceed API limits or free quotas | Start small and ramp up; mark unsupported or failed sizes rather than shortening silently. |
| Compression takes longer than expected | Record compression, API, and total latency separately. |
| Current LangGraph integration supports only text roles | The first Streamlit implementation should use plain text `user` and `assistant` message dictionaries and the direct `compress()` API. Do not pass system/tool/multimodal messages through the current adapter without explicit support. |
| Benchmarks accidentally compare different prompts or models | Freeze and log all non-strategy parameters within each comparison. |
| Free provider terms allow submitted content to be used for product improvement | Use synthetic test data; document provider and tier used in the run. |

## 15. Definition of the first deliverable

The first deliverable is deliberately small:

> **A Streamlit chatbot with a Full history / PromptVault switch, using one configured cloud API, plus a visible token/latency diagnostic panel and one repeatable old-fact recall test.**

Do not build all benchmark charts before this works. Once the chat and fact-retention smoke test work, add the comparison runner; then add the broader diagnostics dashboard in the same app.

## 16. Official references checked for this plan

- Gemini API pricing/free-tier terms: https://ai.google.dev/gemini-api/docs/pricing
- Gemini API setup: https://ai.google.dev/gemini-api/docs/get-started
- OpenRouter pricing and free-model limits: https://openrouter.ai/pricing
- Groq rate limits: https://console.groq.com/docs/rate-limits
- xAI API information: https://x.ai/api
- xAI API pricing: https://docs.x.ai/developers/pricing
- Streamlit chat elements: https://docs.streamlit.io/develop/api-reference/chat

API prices, model IDs, context windows, free-tier eligibility, and rate limits can change. Re-check the provider's official page immediately before implementation and before publishing benchmark results.
