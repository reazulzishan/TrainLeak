"""Data checks: is any of the test set already in the training set, or does a column give the answer away?"""

from __future__ import annotations

import re
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .report import Finding, Report

TARGET_NAMES = ("target", "label", "labels", "y", "class", "outcome", "sentiment", "category", "is_fraud", "is_scam")
ENTITY = r"product|item|user|customer|client|patient|subject|participant|host|seller|shop|store|author|speaker|session|device|account|household|school|hospital"
MAX_ROWS = 20_000  # target-leak check samples this many training rows (speed)


# ------------------------------------------------------------------ inputs
MAX_FILE_BYTES = 2 * 1024**3  # 2 GB: larger files need the Python API with a sampled DataFrame
UNSAFE = {".pkl", ".pickle", ".joblib", ".npy", ".npz"}  # can run code when loaded: never opened


def _frame(x, name):
    if x is None or isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, (str, Path)):
        p = Path(x)
        s = p.suffix.lower()
        if s in UNSAFE:
            raise ValueError(
                f"{p.name}: {s} files can run code when opened, so TrainLeak won't open them. Save the data as CSV or Parquet."
            )
        if p.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(
                f"{p.name} is larger than 2 GB. Load a sample in Python and pass the DataFrame: trainleak.check(df.sample(500_000))"
            )
        if s == ".parquet":
            try:
                return pd.read_parquet(p)
            except ImportError:
                raise
            except (
                Exception
            ):  # pandas 3.0 + pyarrow 22 in Pyodide fail inside pandas' arrow patch; pyarrow itself reads fine
                import pyarrow.parquet as pq

                return pq.read_table(p).to_pandas()
        if s in (".xlsx", ".xls"):
            try:
                import defusedxml  # noqa: F401  (openpyxl uses it, when installed, to block XML attacks in .xlsx files)
            except ImportError as e:
                raise ImportError("Excel files need: pip install 'trainleak[excel]'", name="openpyxl") from e
            return pd.read_excel(p)
        if s == ".json":
            return pd.read_json(p)
        return pd.read_csv(p, sep=None, engine="python") if s in (".tsv", ".txt") else pd.read_csv(p)
    if isinstance(x, pd.Series):
        return x.to_frame(x.name or name)
    a = np.asarray(x)
    return pd.DataFrame(a.reshape(len(a), -1)).rename(columns=lambda c: f"col_{c}")


def _guess(df, target, text, ids, time, notes):
    cols = list(df.columns)
    if target is None:
        target = next((c for c in cols if str(c).lower() in TARGET_NAMES), None)
        if target is not None:
            notes.append(f"target column = '{target}' (guessed from its name; pass target=... to change)")
    obj = [c for c in cols if c != target and _is_text(df[c])]
    if text is None:
        text = [
            c
            for c in obj
            if df[c].dropna().astype(str).str.len().mean() > 25 and df[c].nunique() > 0.5 * df[c].notna().sum()
        ]
        if text:
            notes.append(f"text columns = {text}")
    if ids is None:
        ids = [
            c
            for c in cols
            if c != target and re.search(r"(^|_)(id|uuid|key)$|^id", str(c).lower()) and df[c].nunique() > 0.9 * len(df)
        ]
        if ids:
            notes.append(f"ID columns = {ids} (left out of the duplicate check; IDs shared across splits are checked)")
    if time is None:
        for c in cols:
            if pd.api.types.is_datetime64_any_dtype(df[c]) or (
                re.search(r"date|time|timestamp", str(c).lower()) and _is_text(df[c])
            ):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        parsed = pd.to_datetime(df[c].dropna().head(200), errors="coerce")
                    if parsed.notna().mean() > 0.9:
                        time = c
                        notes.append(f"time column = '{c}'")
                        break
                except (ValueError, TypeError, OverflowError):
                    continue  # not a date column after all
    return target, list(text or []), list(ids or []), time


def _is_text(s: pd.Series) -> bool:
    """object (pandas < 3) or the new str dtype (pandas 3)."""
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def _squash(t: str) -> str:
    # keep letters AND combining marks (Bangla া ি ্ …); drop punctuation, symbols, spaces, control chars; every digit run -> 0
    out, prev_digit = [], False
    for ch in unicodedata.normalize(
        "NFKC", t
    ).lower():  # NFKC: one form for letters typed in two ways (Bangla য় = য + ়)
        if ch.isdigit():
            if not prev_digit:
                out.append("0")
            prev_digit = True
            continue
        if unicodedata.category(ch)[0] not in "PSZC":
            out.append(ch)
            prev_digit = False
        # punctuation/space inside a number (50,000 · 0171-234 · +880 17) keeps it one number
    return "".join(out)


def _norm_text(s: pd.Series) -> pd.Series:
    """Same text once case, numbers, punctuation and spacing are ignored: 'Call 017.. now!' == 'call 018..now'."""
    return s.fillna("").astype(str).map(_squash)


def _keys(df, cols, text, normalise):
    """One hash per row over `cols` (missing values hash as missing, never as 'equal to everything')."""
    if not cols:
        return pd.Series(np.zeros(len(df), dtype="uint64"), index=df.index)
    d = {}
    for c in cols:
        v = df[c]
        if normalise and c in text:
            v = _norm_text(v)
        elif normalise and pd.api.types.is_float_dtype(v):
            v = v.round(4)
        d[c] = v.astype(object).where(v.notna(), "\x00<NA>").astype(str)
    return pd.util.hash_pandas_object(pd.DataFrame(d, index=df.index), index=False)


def _show(row, cols, n=90):
    return " | ".join(f"{c}={str(row[c])[:n]}" for c in cols[:4])


# ------------------------------------------------------------------ checks
def _overlap(train, test, cols, text, f):
    if not cols:
        return
    ktr, kte = _keys(train, cols, text, False), _keys(test, cols, text, False)
    exact = kte.isin(set(ktr))
    n = int(exact.sum())
    if n:
        f.append(
            Finding(
                "leak",
                "Test rows that are also in the training set",
                f"{n:,} of {len(test):,} test rows ({n / len(test):.1%}) are exact copies of training rows.",
                "The model has already seen these answers, so its test score is inflated. Reviewers call this data leakage.",
                "Remove the copies before splitting, then split:\n  df = df.drop_duplicates(subset=FEATURES)\n  train, test = train_test_split(df, test_size=0.2, random_state=42)",
                [_show(r, cols) for _, r in test[exact].head(3).iterrows()],
            )
        )
    if text or any(pd.api.types.is_float_dtype(train[c]) for c in cols):
        ntr, nte = _keys(train, cols, text, True), _keys(test, cols, text, True)
        near = nte.isin(set(ntr)) & ~exact
        m = int(near.sum())
        if m:
            f.append(
                Finding(
                    "leak",
                    "Same row with only the numbers changed",
                    f"{m:,} more test rows ({m / len(test):.1%}) match a training row once numbers, case and punctuation are ignored "
                    "(e.g. the same SMS template with a different phone number or amount).",
                    "The model memorises the template, not the pattern, and scores well on 'new' data that isn't new.",
                    "Group the look-alikes and split by group, so a template sits on one side only:\n"
                    "  from sklearn.model_selection import GroupShuffleSplit\n  groups = trainleak.template_groups(df, text_cols)\n"
                    "  tr, te = next(GroupShuffleSplit(test_size=0.2, random_state=42).split(df, groups=groups))",
                    [_show(r, cols) for _, r in test[near].head(3).iterrows()],
                )
            )


def _text_overlap(train, test, cols, text, f):
    if not text or set(text) == set(cols):
        return
    row_hit = _keys(test, cols, text, True).isin(set(_keys(train, cols, text, True)))
    for c in text:
        hit = _keys(test, [c], text, True).isin(set(_keys(train, [c], text, True))) & ~row_hit
        n = int(hit.sum())
        if n:
            f.append(
                Finding(
                    "leak",
                    f"Same '{c}' text in train and test",
                    f"{n:,} more test rows ({n / len(test):.1%}) have a '{c}' text that is already in training (other columns differ).",
                    "In text tasks the model learns from the words, so these test rows are not new to it.",
                    f"Remove repeated texts before splitting, or split by text group: trainleak.template_groups(df, ['{c}'])",
                    [_show(r, [c]) for _, r in test[hit].head(3).iterrows()],
                )
            )


def _near_text(train, test, text, f, threshold=0.9, limit=30_000):
    if not text:
        return
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        return
    a, b = train[text].astype(str).agg(" ".join, axis=1), test[text].astype(str).agg(" ".join, axis=1)
    if len(a) > limit:
        a = a.sample(limit, random_state=0)
    if len(b) > limit:
        b = b.sample(limit, random_state=0)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True).fit(pd.concat([a, b]))
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(vec.transform(a))
    d, idx = nn.kneighbors(vec.transform(b))
    sim = 1 - d[:, 0]
    na, nb = _norm_text(a).values, _norm_text(b).values
    hit = (sim >= threshold) & (nb != na[idx[:, 0]])  # not already counted as the same template
    n = int(hit.sum())
    if n:
        ex = [
            f"test: {b.iloc[i][:80]!r}  ~  train: {a.iloc[idx[i, 0]][:80]!r} (similarity {sim[i]:.2f})"
            for i in np.where(hit)[0][:3]
        ]
        f.append(
            Finding(
                "risk",
                "Near-duplicate texts across train and test",
                f"{n:,} test texts ({n / len(b):.1%}{', sampled' if len(test) > limit else ''}) are ≥{threshold:.0%} similar to a training text.",
                "Very similar texts on both sides make the test easier than real life.",
                "Look at the examples. If they are the same message, group them and split by group (see fix above).",
                ex,
            )
        )


def _shared_ids(train, test, cols, f, what):
    for c in cols:
        if c not in test:
            continue
        shared = set(train[c].dropna()) & set(test[c].dropna())
        if shared:
            n = int(test[c].isin(shared).sum())
            f.append(
                Finding(
                    "leak",
                    f"Same {what} '{c}' in train and test",
                    f"{len(shared):,} {what} values appear on both sides ({n:,} test rows, {n / len(test):.1%}).",
                    f"Rows from one {what} (a patient, user, product, host…) share hidden traits; the model recognises the {what} instead of learning the task.",
                    f"Split by {what}:\n  from sklearn.model_selection import GroupShuffleSplit\n"
                    f"  tr, te = next(GroupShuffleSplit(test_size=0.2, random_state=42).split(df, groups=df['{c}']))",
                    [str(v) for v in list(shared)[:3]],
                )
            )


def _entities(df, target, skip):
    out = []
    for c in df.columns:
        name = str(c).lower()
        if (
            c == target
            or c in skip
            or not re.search(ENTITY, name)
            or re.search(r"categor|type|kind|class|status|level|segment", name)
        ):
            continue
        u = df[c].nunique()
        if 1 < u < 0.9 * len(df):
            out.append(c)
    return out


def _entity_hint(train, test, cols, f):
    for c in cols:
        if test is None:
            per = len(train) / max(train[c].nunique(), 1)
            f.append(
                Finding(
                    "info",
                    f"Many rows share the same '{c}'",
                    f"{train[c].nunique():,} different values for {len(train):,} rows (about {per:.1f} rows each).",
                    f"If the same {c} lands in train and test, the model can recognise it instead of learning the task, and the test score looks better than it will on new {c}s.",
                    f"If you will predict for NEW {c}s, split by it:\n  from sklearn.model_selection import GroupShuffleSplit\n"
                    f"  tr, te = next(GroupShuffleSplit(test_size=0.2, random_state=42).split(df, groups=df['{c}']))",
                    [],
                )
            )
        elif c in test:
            shared = set(train[c].dropna()) & set(test[c].dropna())
            if shared:
                n = int(test[c].isin(shared).sum())
                f.append(
                    Finding(
                        "risk",
                        f"Same '{c}' values in train and test",
                        f"{n:,} test rows ({n / len(test):.1%}) have a '{c}' that also appears in training ({len(shared):,} shared values).",
                        f"Fine if you will only predict for {c}s you already know. If you'll predict for NEW ones, this score is too optimistic.",
                        f"Split by '{c}' (GroupShuffleSplit / GroupKFold with groups=df['{c}']), or pass group='{c}' to make this a hard check.",
                        [str(v)[:80] for v in list(shared)[:3]],
                    )
                )


def _target_leak(df, target, skip, f, notes):
    if target is None or target not in df:
        return
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    d = df.dropna(subset=[target])
    if len(d) > MAX_ROWS:
        d = d.sample(MAX_ROWS, random_state=0)
    y = d[target]
    reg = pd.api.types.is_numeric_dtype(y) and y.nunique() > 20
    if not reg:
        y = pd.Series(pd.factorize(y)[0], index=y.index)
    if not reg and y.value_counts(normalize=True).iloc[0] > 0.98:
        return
    if len(d) < 30 or (not reg and y.value_counts().min() < 3):
        notes.append(
            "too few rows (or a class with < 3 rows) to test single columns against the target: that check was skipped"
        )
        return
    hits = []
    for c in d.columns:
        if c == target or c in skip:
            continue
        x = d[c]
        if x.nunique(dropna=True) <= 1:
            continue
        if pd.api.types.is_numeric_dtype(x):
            xv = x.fillna(x.median()).to_numpy(float)
        elif pd.api.types.is_datetime64_any_dtype(x) or x.nunique() > 0.5 * len(x):
            continue
        else:
            xv = pd.factorize(x)[0].astype(float)
        X = xv.reshape(-1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if reg:
                s = cross_val_score(DecisionTreeRegressor(max_depth=6, random_state=0), X, y, cv=3, scoring="r2").mean()
                rho = pd.Series(xv).corr(pd.Series(y.to_numpy(float)), method="spearman")
                score, label = max(s, (rho or 0) ** 2), "R²"
            else:
                score, label = (
                    cross_val_score(
                        DecisionTreeClassifier(max_depth=6, random_state=0), X, y, cv=3, scoring="balanced_accuracy"
                    ).mean(),
                    "balanced accuracy",
                )
        if score >= 0.9:
            hits.append((score, c, label))
    for score, c, label in sorted(hits, reverse=True)[:5]:
        lvl = "leak" if score >= 0.98 else "risk"
        f.append(
            Finding(
                lvl,
                f"Column '{c}' alone predicts '{target}'",
                f"Using only '{c}', a small tree reaches {label} {score:.3f} on held-out rows.",
                "One column that (almost) is the answer usually means it was recorded after the outcome, or computed from it. "
                "Real-world predictions won't have it."
                if lvl == "leak"
                else "That is suspiciously strong for a single column. Check it was known before the outcome happened.",
                f"Ask: would I know '{c}' at prediction time? If not, drop it:\n  X = df.drop(columns=['{target}', '{c}'])",
                [],
            )
        )


def _time(train, test, time, f):
    if time is None or time not in test:
        return
    tr, te = pd.to_datetime(train[time], errors="coerce"), pd.to_datetime(test[time], errors="coerce")
    if tr.notna().sum() == 0 or te.notna().sum() == 0:
        return
    early = (te < tr.max()).mean()
    if early > 0.05:
        f.append(
            Finding(
                "risk",
                "Test data is not later than training data",
                f"{early:.0%} of test rows are dated before the last training row ({tr.max().date()}).",
                "If you will predict the future, training on later data and testing on earlier data shows a score you won't get in real use.",
                f"Split by time:\n  df = df.sort_values('{time}')\n  cut = int(len(df) * 0.8)\n  train, test = df.iloc[:cut], df.iloc[cut:]",
                [],
            )
        )


def _within(df, cols, text, f, ids):
    n_ex = int(_keys(df, cols, text, False).duplicated().sum())
    n_nm = int(_keys(df, cols, text, True).duplicated().sum()) if text else n_ex
    if n_nm:
        f.append(
            Finding(
                "risk",
                "Copies inside your data",
                f"{n_ex:,} rows are exact copies"
                + (f", {n_nm:,} once numbers/case are ignored" if n_nm != n_ex else "")
                + f" (of {len(df):,}).",
                "A random split will put copies on both sides, so train and test overlap.",
                "Drop exact copies, and split look-alikes by group:\n  df = df.drop_duplicates(subset=FEATURES)\n"
                "  groups = trainleak.template_groups(df, text_cols)  # then GroupShuffleSplit / GroupKFold",
                [],
            )
        )
    for c in text if set(text) != set(cols) else []:
        t_ex = int(_keys(df, [c], text, False).duplicated().sum())
        t_nm = int(_keys(df, [c], text, True).duplicated().sum())
        if t_nm > n_nm:
            f.append(
                Finding(
                    "risk",
                    f"The same '{c}' text appears more than once",
                    f"{t_ex:,} rows repeat a '{c}' text already in the data"
                    + (f" ({t_nm:,} once numbers/case/punctuation are ignored)" if t_nm != t_ex else "")
                    + ", even where other columns differ.",
                    "In text tasks the model learns from the words. The same text on both sides of a split is a leak even if the other columns differ.",
                    f"Keep one copy per text, or split by text group:\n  df = df.drop_duplicates(subset=['{c}'])\n"
                    f"  groups = trainleak.template_groups(df, ['{c}'])  # then GroupShuffleSplit / GroupKFold",
                    [],
                )
            )
    for c in ids:
        rep = df[c].duplicated().sum()
        if rep:
            f.append(
                Finding(
                    "info",
                    f"'{c}' repeats",
                    f"{rep:,} rows share a '{c}' with another row.",
                    "Rows from the same entity should stay on one side of the split.",
                    f"Use GroupKFold / GroupShuffleSplit with groups=df['{c}'].",
                    [],
                )
            )


# ------------------------------------------------------------------ public
def template_groups(df: pd.DataFrame, text_cols) -> np.ndarray:
    """Group id per row: rows that are the same once numbers/case/punctuation are ignored share a group."""
    k = _norm_text(df[list(text_cols)].astype(str).agg(" ".join, axis=1))
    return pd.factorize(k)[0]


def check(
    train,
    test=None,
    y_train=None,
    y_test=None,
    *,
    target=None,
    group=None,
    time=None,
    text=None,
    ids=None,
    near=True,
    ignore=None,
) -> Report:
    """Check a dataset (or a train/test split) for leakage.

    trainleak.check(df)                                   # before splitting
    trainleak.check(train_df, test_df, target="label")    # after splitting
    trainleak.check(X_train, X_test, y_train, y_test)     # sklearn style
    """
    notes: list[str] = []
    f: list[Finding] = []
    src = " vs ".join(str(Path(x).name) for x in (train, test) if isinstance(x, (str, Path)))
    tr, te = _frame(train, "train"), _frame(test, "test")
    if y_train is not None:
        target = target or (getattr(y_train, "name", None) or "target")
        tr = tr.assign(**{target: np.asarray(y_train)})
        if te is not None and y_test is not None:
            te = te.assign(**{target: np.asarray(y_test)})
    for name, v in (("target", target), ("group", group), ("time", time)):
        if v is not None and v not in tr.columns:
            raise KeyError(f"{name} column '{v}' is not in the data. Columns: {list(tr.columns)[:15]}")
    if te is not None:
        only_tr, only_te = set(tr.columns) - set(te.columns) - {target}, set(te.columns) - set(tr.columns)
        if only_tr or only_te:
            notes.append(
                f"train and test have different columns (only in train: {sorted(map(str, only_tr))[:5]}, only in test: {sorted(map(str, only_te))[:5]}); compared the shared ones"
            )
    target, text, ids, time = _guess(tr, target, text, ids, time, notes)
    if target is None:
        notes.append("no target column found: the 'column gives the answer away' check was skipped (pass target=...)")
    groups = [group] if group else []
    feats = [
        c for c in tr.columns if c != target and c not in ids and c not in groups and (te is None or c in te.columns)
    ]
    ents = [c for c in _entities(tr, target, set(ids) | set(text) | set(groups)) if not group]
    if te is None:
        _within(tr, feats, text, f, ids + groups)
        _entity_hint(tr, None, ents, f)
    else:
        _overlap(tr, te, feats, text, f)
        _text_overlap(tr, te, feats, text, f)
        if near:
            _near_text(tr, te, text, f)
        _shared_ids(tr, te, groups, f, "group")
        _shared_ids(tr, te, ids, f, "ID")
        _time(tr, te, time, f)
        _entity_hint(tr, te, ents, f)
    _target_leak(tr, target, set(ids) | set(text) | set(groups) | ({time} if time else set()), f, notes)
    if te is None:
        checks = [
            "copies inside the data",
            "repeated texts" if text else "",
            "repeated IDs/entities",
            "single columns that predict the target" if target else "",
        ]
    else:
        checks = [
            "exact copies across train/test",
            "same row with numbers changed",
            "same text across train/test" if text else "",
            "near-duplicate text" if text and near else "",
            "shared groups/IDs/entities",
            "time order" if time else "",
            "single columns that predict the target" if target else "",
        ]
    return Report(f, notes, src, [c for c in checks if c]).ignore(ignore)
