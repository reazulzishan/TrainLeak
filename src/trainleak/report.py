"""Findings and the report a user reads: plain words, why it matters, how to fix it."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

LEVELS = {"leak": 0, "risk": 1, "info": 2}  # leak = certain problem, risk = likely problem, info = good to know
ICON = {"leak": "✖", "risk": "!", "info": "i"}
COLOR = {"leak": "\033[31m", "risk": "\033[33m", "info": "\033[36m"}


SCHEMA_VERSION = 1
# Stable codes: users can --ignore them and docs can explain each one. Never reuse or renumber a code.
CODES = [
    (r"^Test rows that are also in the training set", "TL101"),
    (r"^Same row with only the numbers changed", "TL102"),
    (r"^Same '.*' text in train and test", "TL103"),
    (r"^Near-duplicate texts across train and test", "TL104"),
    (r"^Same group '", "TL105"),
    (r"^Same ID '", "TL106"),
    (r"^Test data is not later than training data", "TL107"),
    (r"^Same '.*' values in train and test", "TL108"),
    (r"^Column '.*' alone predicts", "TL201"),
    (r"^Copies inside your data", "TL301"),
    (r"^The same '.*' text appears more than once", "TL302"),
    (r"^'.*' repeats", "TL303"),
    (r"^Many rows share the same", "TL304"),
    (r"runs before the train/test split$", "TL401"),
    (r"applied to the test set$", "TL402"),
    (r"is fitted on test data$|fitted on test data$", "TL403"),
    (r"learns from all data before the split$|fitted before the split$", "TL404"),
    (r"^Features chosen using the target on all data", "TL405"),
    (r"^Resampling \+ cross-validation without an imblearn Pipeline", "TL406"),
]


def _code(title: str) -> str:
    return next((c for rx, c in CODES if re.search(rx, title)), "TL000")


def _md(s) -> str:
    """Make user data safe inside Markdown: no raw HTML, no breaking out of code spans/blocks."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("`", "'").replace("\n", " ")


@dataclass
class Finding:
    level: str  # "leak" | "risk" | "info"
    title: str  # one line, plain words
    what: str  # what we found, with the real numbers
    why: str  # why it matters for the result / a reviewer
    fix: str  # what to do, with code when it helps
    examples: list = field(default_factory=list)  # a few real rows / lines as proof
    code: str = ""

    def __post_init__(self):
        if not self.code:
            self.code = _code(self.title)


@dataclass
class Report:
    findings: list
    notes: list = field(default_factory=list)  # what TrainLeak assumed (guessed columns, sampling)
    source: str = ""
    checks: list = field(
        default_factory=list
    )  # what was actually checked (shown so a clean report isn't read as a proof)

    def ignore(self, codes) -> Report:
        codes = {c.strip().upper() for c in (codes or []) if c.strip()}
        return Report(
            [f for f in self.findings if f.code not in codes],
            self.notes + ([f"ignored: {', '.join(sorted(codes))}"] if codes else []),
            self.source,
            self.checks,
        )

    @property
    def verdict(self) -> str:
        n = sum(f.level == "leak" for f in self.findings)
        r = sum(f.level == "risk" for f in self.findings)
        if n:
            return f"{n} leak{'s' * (n > 1)} found: your test score is probably too optimistic."
        if r:
            return f"No certain leak, but {r} thing{'s' * (r > 1)} to check."
        return "No leaks found by these checks."

    @property
    def ok(self) -> bool:
        return not any(f.level == "leak" for f in self.findings)

    def sorted(self):
        return sorted(self.findings, key=lambda f: LEVELS[f.level])

    def text(self, color: bool = True) -> str:
        c = (lambda lv, s: f"{COLOR[lv]}{s}\033[0m") if color else (lambda lv, s: s)
        b = (lambda s: f"\033[1m{s}\033[0m") if color else (lambda s: s)
        head = (
            "leak"
            if not self.ok
            else ("risk" if self.findings and any(f.level == "risk" for f in self.findings) else "info")
        )
        out = [
            b("TrainLeak" + (f" · {self.source}" if self.source else "")),
            c(head, self.verdict) if self.findings else c("info", "✔ " + self.verdict),
            "",
        ]
        for i, f in enumerate(self.sorted(), 1):
            out += [
                c(f.level, f"{ICON[f.level]} {i}. [{f.level.upper()} {f.code}] {f.title}"),
                f"   What:  {f.what}",
                f"   Why:   {f.why}",
                "   Fix:   " + f.fix.replace("\n", "\n          "),
            ]
            for e in f.examples[:3]:
                out.append(f"   e.g.   {e}")
            out.append("")
        if self.notes:
            out += ["Assumed: " + n for n in self.notes]
        if self.checks:
            out.append(
                "Checked: "
                + "; ".join(self.checks)
                + ". A clean result means these checks found nothing, not that no leak exists."
            )
        return "\n".join(out).rstrip() + "\n"

    def summary(self) -> str:
        return ("LEAK: " if not self.ok else "OK: ") + self.verdict + (f" ({self.source})" if self.source else "")

    def markdown(self) -> str:
        md = [f"# TrainLeak report{' · ' + self.source if self.source else ''}", "", f"**{self.verdict}**", ""]
        for i, f in enumerate(self.sorted(), 1):
            md += [
                f"## {i}. [{f.level.upper()} {f.code}] {_md(f.title)}",
                "",
                f"- **What:** {_md(f.what)}",
                f"- **Why it matters:** {_md(f.why)}",
                "",
                "**Fix:**",
                "",
                "````",
                f.fix,
                "````",
            ]
            if f.examples:
                md += ["", "Examples:", ""] + [f"- `{_md(str(e)[:200])}`" for e in f.examples[:5]]
            md.append("")
        if self.notes:
            md += ["---", ""] + [f"- Assumed: {_md(n)}" for n in self.notes]
        if self.checks:
            md += [
                "",
                "**Checks performed:** " + "; ".join(self.checks) + ".",
                "",
                "_A clean result means these checks found nothing. It is not a proof that no leak exists._",
            ]
        return "\n".join(md) + "\n"

    def json(self) -> str:
        return json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "source": self.source,
                "verdict": self.verdict,
                "ok": self.ok,
                "notes": self.notes,
                "checks": self.checks,
                "findings": [asdict(f) for f in self.sorted()],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    THEMES = {  # noqa: RUF012
        # WCAG contrast, dark on #0B1220/#131C2E: text 15.9/14.5, cyan 10.4/9.4, red 6.8/6.2, amber 11.2/10.2, muted 8.5/7.7, green 9.7/8.9
        "dark": {
            "bg": "#0B1220",
            "card": "#131C2E",
            "code": "#0B1220",
            "ink": "#E6EDF7",
            "muted": "#9FB0C8",
            "leak": "#F87171",
            "risk": "#FBBF24",
            "info": "#22D3EE",
            "ok": "#34D399",
        },
        # light on #FFFFFF/#F1F5F9: ink 17.9/16.3, muted 7.6/6.9, cyan 5.4/4.9, red 6.5/5.9, amber 5.0/4.6, green 5.0/4.6 (all >= 4.5 AA)
        "light": {
            "bg": "#FFFFFF",
            "card": "#F1F5F9",
            "code": "#FFFFFF",
            "ink": "#0F172A",
            "muted": "#475569",
            "leak": "#B91C1C",
            "risk": "#B45309",
            "info": "#0E7490",
            "ok": "#15803D",
        },
    }

    def html(self, theme: str = "dark") -> str:
        """The report as one self-contained HTML card. theme: "dark" (default, reads the same in light and dark notebooks) or "light"."""
        from html import escape

        p = self.THEMES[theme]
        head = p["leak"] if not self.ok else (p["risk"] if any(f.level == "risk" for f in self.findings) else p["ok"])

        def item(f):
            ex = "".join(
                f"<div style='color:{p['muted']};font-family:ui-monospace,monospace;font-size:12px'>{escape(str(e)[:200])}</div>"
                for e in f.examples[:3]
            )
            return (
                f'<div style="border-left:4px solid {p[f.level]};padding:8px 12px;margin:10px 0;background:{p["card"]};border-radius:6px">'
                f'<div><span style="color:{p[f.level]};font-weight:700">{ICON[f.level]} {f.level.upper()}</span> '
                f'<span style="color:{p["muted"]};font-family:ui-monospace,monospace">{f.code}</span> <b>{escape(f.title)}</b></div>'
                f'<div style="margin-top:4px"><b>What:</b> {escape(f.what)}</div><div><b>Why:</b> {escape(f.why)}</div>'
                f'<div><b>Fix:</b></div><pre style="margin:4px 0;padding:8px;background:{p["code"]};color:{p["ink"]};border-radius:4px;'
                f'white-space:pre-wrap;border:1px solid {p["card"]}">{escape(f.fix)}</pre>{ex}</div>'
            )

        notes = "".join(
            f"<div style='color:{p['muted']};font-size:12px'>Assumed: {escape(n)}</div>" for n in self.notes
        )
        checks = (
            f"<div style='color:{p['muted']};font-size:12px;margin-top:6px'>Checked: {escape('; '.join(self.checks))}. "
            "A clean result means these checks found nothing, not that no leak exists.</div>"
            if self.checks
            else ""
        )
        return (
            f'<div style="font-family:system-ui,sans-serif;background:{p["bg"]};color:{p["ink"]};padding:14px 16px;border-radius:10px;line-height:1.45">'
            f'<div style="font-weight:800">TrainLeak<span style="color:{p["muted"]};font-weight:400">{(" · " + escape(self.source)) if self.source else ""}</span></div>'
            f'<div style="color:{head};font-weight:700;margin:4px 0 6px">{escape(self.verdict)}</div>'
            + "".join(item(f) for f in self.sorted())
            + notes
            + checks
            + "</div>"
        )

    def _repr_html_(self):
        """Jupyter/Colab display: the dark card has its own background, so it reads the same in light and dark notebook themes."""
        return self.html("dark")

    def __repr__(self):
        return self.text(color=False)
