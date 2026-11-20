# Contributing

Thanks for helping. Bug reports with a small example (a few rows of data, or a short notebook cell) are the most useful thing.

## Set up

```bash
git clone https://github.com/reazulzishan/TrainLeak && cd TrainLeak
uv sync                      # Python 3.10+; installs dev tools from uv.lock
uv run pytest                # tests + coverage gate (85 %)
uv run ruff check src tests && uv run ruff format src tests && uv run mypy
./scripts/matrix.sh          # optional: the Python × pandas matrix CI runs
```

## Rules for a new check

1. Every check has a **leaky example and a clean twin** in the tests. The clean twin must produce no finding: false alarms cost users' trust.
2. Give it a new **stable code** in `report.py` (`CODES`). Never renumber or reuse a code.
3. The message has **What** (real numbers), **Why** (for a student, in plain words), **Fix** (code they can paste).
4. "leak" only when the pattern leaks in every normal use; otherwise "risk".
5. Never execute user code, never touch the network.

## Releases (maintainer)

Update `__version__` in `src/trainleak/__init__.py` and `CHANGELOG.md`, merge, then tag `vX.Y.Z`. The Release workflow
tests, builds and publishes to PyPI after approval of the `pypi` environment. Versions follow SemVer; finding codes and the
JSON `schema_version` are part of the public API.
