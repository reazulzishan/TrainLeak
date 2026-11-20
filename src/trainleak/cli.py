"""trainleak command line.

  trainleak data.csv                      check one dataset before splitting
  trainleak train.csv test.csv            check a split
  trainleak notebook.ipynb  script.py     check code
Options: --target COL --group COL --time COL --text COL [COL...]  --report report.md  --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__, check, check_code

CODE = {".ipynb", ".py"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="trainleak",
        description="Find data leakage in a dataset, a train/test split, or a notebook.",
        epilog="Examples:\n  trainleak data.csv\n  trainleak train.csv test.csv --target label\n  trainleak analysis.ipynb",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("files", nargs="+", help="a dataset, a train + test pair, or .ipynb/.py files")
    p.add_argument("--target", help="the column you predict (guessed from names like label/target/y if not given)")
    p.add_argument(
        "--group", help="column that identifies a person/user/product/patient (rows of one group must stay on one side)"
    )
    p.add_argument("--time", help="date/time column (guessed if not given)")
    p.add_argument("--text", nargs="+", help="text columns (guessed if not given)")
    p.add_argument("--no-near", action="store_true", help="skip the near-duplicate text search (faster)")
    p.add_argument(
        "--report", metavar="FILE.md", help="also save a Markdown report you can attach to a thesis or paper"
    )
    p.add_argument("--json", action="store_true", help="print JSON (schema_version 1) instead of text")
    p.add_argument(
        "--ignore", metavar="CODES", default="", help="comma-separated finding codes to hide, e.g. TL304,TL303"
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="print only the one-line verdict (exit code still tells leak / no leak)",
    )
    p.add_argument("--no-color", action="store_true", help="plain text output (also: NO_COLOR=1)")
    p.add_argument("--version", action="version", version=f"trainleak {__version__}")
    p.epilog = (p.epilog or "") + "\n\nExit codes: 0 no leak found · 1 leak found · 2 error (bad input)"
    a = p.parse_args(argv)
    ignore = [c for c in a.ignore.split(",") if c.strip()]

    files = [Path(x) for x in a.files]
    missing = [str(x) for x in files if not x.exists()]
    if missing:
        p.error("file not found: " + ", ".join(missing))
    code = [x for x in files if x.suffix in CODE]
    data = [x for x in files if x.suffix not in CODE]
    if len(data) > 2:
        p.error("give one dataset, or a train and a test file (2 at most)")

    try:
        reports = [check_code(x, ignore=ignore) for x in code]
        if data:
            reports.append(
                check(
                    data[0],
                    data[1] if len(data) == 2 else None,
                    target=a.target,
                    group=a.group,
                    time=a.time,
                    text=a.text,
                    near=not a.no_near,
                    ignore=ignore,
                )
            )
    except ImportError as e:
        mod = getattr(e, "name", "") or str(e)
        hint = {
            "pyarrow": "pip install pyarrow   (needed for .parquet files)",
            "openpyxl": "pip install openpyxl   (needed for .xlsx files)",
        }.get(mod, f"pip install {mod}")
        print(f"trainleak: a package is missing to read this file. Run:\n  {hint}", file=sys.stderr)
        return 2
    except (ValueError, KeyError, UnicodeDecodeError, json.JSONDecodeError, OSError) as e:
        print(
            f"trainleak: could not read the input: {e}\nTip: check the file is a valid CSV / Parquet / Excel / .ipynb, and the column names you passed exist.",
            file=sys.stderr,
        )
        return 2
    color = sys.stdout.isatty() and not (
        a.json or a.no_color or os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb"
    )
    for r in reports:
        print(r.json() if a.json else (r.summary() if a.quiet else r.text(color=color)))
    if a.report:
        Path(a.report).write_text("\n".join(r.markdown() for r in reports), encoding="utf-8")
        print(f"Report saved: {a.report}", file=sys.stderr)
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
