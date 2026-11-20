# TrainLeak: project report

**Goal:** a small, trustworthy tool that tells ML students and researchers, in one line, whether their test score is
inflated by data leakage, and exactly how to fix it.

## Why this tool
- Our own projects (ScamGuard, CommerceGuard, DataPipeline) each hit a leak that inflated results; our most-watched
  YouTube video is about one (SMOTE before the split). Students search for this ("research paper reject", "data leakage").
- Research (Nov 2026, `research/UX_RESEARCH.md`): established tools (deepchecks, cleanlab) are broad and heavy; seven small
  2026 packages cover parts of it; **no pip-installable tool also checks the notebook code**. The name `leakcheck` was taken
  on PyPI, so the tool ships as `trainleak`.

## What it does
Data checks (TL1xx–TL3xx) on a DataFrame, arrays or files; code checks (TL4xx) on `.ipynb`/`.py`, read as text, never run.
Every finding: stable code, level (leak / risk / info), What (real numbers), Why, Fix (code), examples. Output: notebook
card, terminal, Markdown report, JSON (`schema_version` 1). See `README.md`.

## Validation on real data (no tuning)
| Data | Known (manual) | TrainLeak |
|---|---|---|
| Bengali SMS smishing, train+val vs test | 81 template copies | 93 (incl. 79 of the 81 + 14 missed by hand), 48 near-duplicates, ~2 s |
| NYC Airbnb, target price | `price_quote_price_per_night` = price | R² 1.000 leak, ~2 s |
| 78,130 BD e-commerce reviews | 13,983 repeated texts | 13,983; plus `Emotion` → `Sentiment` 1.000 (new), ~3 s |
| 5 leak-safe notebooks | clean | no findings |

Validation found 6 bugs before release (pandas-3 missing values made rows look like copies; pandas-3 text dtype; Bangla
vowel signs dropped by `\w`; punctuated numbers; product names flagged as duplicate text; crash on tiny datasets). Each has
a regression test.

## Engineering
- `src/` layout, hatchling, single-source version, `py.typed`, SPDX licence, extras `[parquet]`, `[excel]` (+defusedxml).
- Gates: ruff (lint + format), mypy, pytest 37 tests, branch coverage ≥ 85 % (89 %).
- Compatibility verified locally: Python 3.10 / pandas 2.0 / numpy 1.26 / sklearn 1.2 up to Python 3.14 / pandas 3.1.
- CI: lint, matrix (3 OS × Python 3.10/3.12/3.14 × pandas 2/3 + oldest stack), build + clean-install smoke test.
- Release: tag-triggered, tests again, version/tag check, PyPI trusted publishing (OIDC, no tokens) behind a protected
  environment, automatic attestations. Actions pinned to SHAs; `zizmor` (pedantic) and `actionlint` clean; Dependabot.
- Security: no code execution, no network (tested), refuses pickle-like formats, size caps, escaped HTML/Markdown output.

## Limits
Static code reading (line order, no cross-file analysis); domain-knowledge leakage and model selection on the test set are
not detected yet. A clean report is not a proof.

## Next
P1 from the research: AST-based code checks + "model chosen on the test set" rule, `check_cv()`, docs site per finding
code, in-browser web demo (files never leave the browser), pre-commit hook published, Kaggle notebook.
