"""Code checks for .ipynb / .py: steps that learn from data before (or on) the test set.

Static reading of the code, in the order it runs top to bottom. It can't see runtime values, so a "risk" is a
pattern worth a second look, and a "leak" is a pattern that leaks in every normal use.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .report import Finding, Report

RESAMPLERS = r"SMOTE\w*|ADASYN|BorderlineSMOTE|SVMSMOTE|KMeansSMOTE|RandomOverSampler|RandomUnderSampler|NearMiss|TomekLinks|SMOTEENN|SMOTETomek"
LEARNERS = (
    r"StandardScaler|MinMaxScaler|RobustScaler|MaxAbsScaler|Normalizer|QuantileTransformer|PowerTransformer|SimpleImputer|KNNImputer|"
    r"IterativeImputer|PCA|TruncatedSVD|SelectKBest|SelectPercentile|SelectFromModel|RFE|RFECV|VarianceThreshold|TfidfVectorizer|"
    r"CountVectorizer|OneHotEncoder|OrdinalEncoder|TargetEncoder|KBinsDiscretizer|PolynomialFeatures"
)
LEARNS = {
    "StandardScaler": "the mean and spread",
    "MinMaxScaler": "the min and max",
    "RobustScaler": "the median and quartiles",
    "MaxAbsScaler": "the max",
    "QuantileTransformer": "the quantiles",
    "PowerTransformer": "the shape",
    "SimpleImputer": "the fill values (mean/median)",
    "KNNImputer": "neighbour values",
    "IterativeImputer": "a model of the other columns",
    "PCA": "the main directions",
    "TruncatedSVD": "the main directions",
    "TfidfVectorizer": "the vocabulary and word weights",
    "CountVectorizer": "the vocabulary",
    "OneHotEncoder": "the categories",
    "OrdinalEncoder": "the categories",
    "KBinsDiscretizer": "the bin edges",
    "VarianceThreshold": "which columns vary",
    "PolynomialFeatures": "nothing from the values (safe)",
    "TargetEncoder": "the average TARGET per category",
    "SelectKBest": "which columns relate to the TARGET",
    "SelectPercentile": "which columns relate to the TARGET",
    "SelectFromModel": "which columns relate to the TARGET",
    "RFE": "which columns relate to the TARGET",
    "RFECV": "which columns relate to the TARGET",
}
USES_TARGET = {"TargetEncoder", "SelectKBest", "SelectPercentile", "SelectFromModel", "RFE", "RFECV"}
SPLIT = re.compile(
    r"train_test_split\s*\(|\b\w*(KFold|ShuffleSplit|LeaveOneOut|TimeSeriesSplit)\s*\(|\.split\s*\(\s*\w|cross_val_score\s*\(|cross_validate\s*\(|GridSearchCV\s*\(|RandomizedSearchCV\s*\("
)


def _cells(path: Path):
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        for i, c in enumerate(nb.get("cells", []), 1):
            if c.get("cell_type") == "code":
                src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
                yield f"cell {i}", src
    else:
        yield "", path.read_text(encoding="utf-8")


def _lines(path: Path):
    for where, src in _cells(path):
        for j, line in enumerate(src.splitlines(), 1):
            s = line.split("#", 1)[0].rstrip()  # ignore comments
            if s.strip() and not s.lstrip().startswith(("!", "%")):
                yield (f"{where}, line {j}" if where else f"line {j}"), s


def _testish(arg):
    return re.search(r"test|holdout|valid|_te\b|_val\b", arg, re.I)


def _trainish(arg):
    return re.search(r"train|_tr\b", arg, re.I)


MAX_CODE_BYTES = 50 * 1024**2


def check_code(path, ignore=None) -> Report:
    """Read a notebook or script as TEXT and flag leaky patterns. The code is never run or imported."""
    path = Path(path)
    if path.suffix not in (".ipynb", ".py"):
        raise ValueError(f"{path.name}: give a .ipynb or .py file")
    if path.stat().st_size > MAX_CODE_BYTES:
        raise ValueError(
            f"{path.name} is larger than 50 MB: is it a notebook with embedded outputs? Clear outputs and retry."
        )
    lines = list(_lines(path))
    f, notes = [], []
    names = {}  # variable -> class it was built from
    split_at = next((i for i, (_, s) in enumerate(lines) if SPLIT.search(s)), None)
    if split_at is None:
        notes.append(
            "no train/test split or cross-validation found in this file, so 'before the split' checks were skipped"
        )

    for i, (where, s) in enumerate(lines):
        for var, cls in re.findall(rf"(\w+)\s*=\s*(?:\w+\.)*({RESAMPLERS}|{LEARNERS})\s*\(", s):
            names[var] = cls
        before = split_at is not None and i < split_at
        split_where = lines[split_at][0] if split_at is not None else ""

        # resampling (SMOTE & co.)
        m = re.search(rf"(?:(\w+)|(?:{RESAMPLERS})\s*\([^)]*\))\s*\.fit_resample\s*\(([^)]*)\)", s)
        if m:
            inline = re.search(RESAMPLERS, s)
            cls = (
                names.get(m.group(1), "the resampler")
                if m.group(1)
                else (inline.group(0) if inline else "the resampler")
            )
            arg = m.group(2)
            if before and not _testish(arg):
                f.append(
                    Finding(
                        "leak",
                        f"{cls} runs before the train/test split",
                        f"{where}: `{s.strip()[:110]}` comes before the split ({split_where}).",
                        f"{cls} creates new rows from neighbours. Done before the split, test rows are built from training rows (and the other way round), "
                        "so the test score is too high. This is a common reason ML papers get rejected.",
                        "Split first, then resample ONLY the training part:\n  X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=42)\n"
                        f"  X_tr, y_tr = {cls}(random_state=42).fit_resample(X_tr, y_tr)\n"
                        "With cross-validation, put it inside an imblearn Pipeline:\n  from imblearn.pipeline import make_pipeline\n"
                        f"  model = make_pipeline({cls}(random_state=42), YourModel())",
                        [],
                    )
                )
            elif _testish(arg):
                f.append(
                    Finding(
                        "leak",
                        f"{cls} applied to the test set",
                        f"{where}: `{s.strip()[:110]}`.",
                        "The test set must stay real. Synthetic test rows make the score meaningless.",
                        "Never resample test data. Resample the training part only.",
                        [],
                    )
                )

        # preprocessing that learns (scalers, imputers, PCA, vectorisers, encoders, selectors)
        for var, _meth, arg in re.findall(r"(\w+)\s*\.\s*(fit_transform|fit)\s*\(([^)]*)\)", s):
            cls = names.get(var)
            if not cls or re.fullmatch(RESAMPLERS, cls):
                continue
            if _testish(arg):
                f.append(
                    Finding(
                        "leak",
                        f"{cls} is fitted on test data",
                        f"{where}: `{s.strip()[:110]}`.",
                        "Fitting on the test set lets its statistics (mean, vocabulary, components…) shape the model.",
                        f"Fit on training data, then only transform the test data:\n  {var}.fit(X_train)\n  X_test = {var}.transform(X_test)",
                        [],
                    )
                )
            elif before and not _trainish(arg) and cls != "PolynomialFeatures":
                tgt = cls in USES_TARGET
                f.append(
                    Finding(
                        "leak" if tgt else "risk",
                        f"{cls} learns from all data before the split",
                        f"{where}: `{s.strip()[:110]}` runs before the split ({split_where}).",
                        f"{cls} learns {LEARNS.get(cls, 'statistics')} from every row, including the rows that later become the test set. "
                        + (
                            "Because it uses the target, the test labels leak straight into your features: scores can jump a lot."
                            if tgt
                            else "The effect is usually small, but it is a leak and reviewers flag it."
                        ),
                        f"Fit it after the split on the training part, or wrap it with the model:\n  from sklearn.pipeline import make_pipeline\n"
                        f"  model = make_pipeline({cls}(), YourModel())\n  model.fit(X_train, y_train)",
                        [],
                    )
                )
        for cls in re.findall(rf"\b({LEARNERS})\s*\([^)]*\)\s*\.\s*fit_transform\s*\(([^)]*)\)", s):
            name, arg = cls
            if (split_at is not None and i < split_at and not _trainish(arg)) or _testish(arg):
                f.append(
                    Finding(
                        "risk" if not _testish(arg) else "leak",
                        f"{name} fitted {'on test data' if _testish(arg) else 'before the split'}",
                        f"{where}: `{s.strip()[:110]}`.",
                        "The preprocessing learns from rows that should be unseen.",
                        f"Fit on the training part only, or use make_pipeline({name}(), YourModel()).",
                        [],
                    )
                )

        # feature selection with the target on all data
        if (
            before
            and re.search(r"\.corr\(\)\s*\[\s*['\"]\w+['\"]\s*\]|corrwith\(", s)
            and re.search(r"drop|select|>|<|abs", s)
        ):
            f.append(
                Finding(
                    "risk",
                    "Features chosen using the target on all data",
                    f"{where}: `{s.strip()[:110]}`.",
                    "Choosing features by their link to the target before splitting lets the test labels guide the model.",
                    "Do the selection on the training part only (or inside a Pipeline).",
                    [],
                )
            )

    if (
        any(re.search(r"\.fit_resample\s*\(", s) for _, s in lines)
        and any(re.search(r"cross_val_score|cross_validate|GridSearchCV|RandomizedSearchCV", s) for _, s in lines)
        and not any(re.search(r"imblearn\.pipeline|from imblearn.pipeline", s) for _, s in lines)
    ):
        f.append(
            Finding(
                "risk",
                "Resampling + cross-validation without an imblearn Pipeline",
                "The file resamples data and also cross-validates, but never uses imblearn's Pipeline.",
                "If the resampled data goes into cross-validation, synthetic copies of a row land in both the training and validation folds.",
                "from imblearn.pipeline import make_pipeline\nmodel = make_pipeline(SMOTE(random_state=42), YourModel())\ncross_val_score(model, X_train, y_train, cv=5)",
                [],
            )
        )
    # one finding per pattern+place is enough
    seen, out = set(), []
    for x in f:
        k = (x.title, x.what)
        if k not in seen:
            seen.add(k)
            out.append(x)
    checks = [
        "resampling (SMOTE etc.) before the split or on test data",
        "preprocessing fitted before the split or on test data",
        "target-based feature selection on all data",
        "resampling inside cross-validation without an imblearn Pipeline",
    ]
    return Report(out, notes, path.name, checks).ignore(ignore)
