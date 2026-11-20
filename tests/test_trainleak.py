import json

import numpy as np
import pandas as pd
import pytest

import trainleak
from trainleak.cli import main

rng = np.random.default_rng(0)


def titles(r):
    return [f.title for f in r.findings]


def levels(r):
    return {f.title: f.level for f in r.findings}


def clean_split(n=600):
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.integers(0, 50, n), "c": rng.choice(list("xyz"), n)})
    df["label"] = (df.a + rng.normal(scale=1.5, size=n) > 0).astype(int)
    return df.iloc[: n * 4 // 5].copy(), df.iloc[n * 4 // 5 :].copy()


def test_clean_split_has_no_leak():
    tr, te = clean_split()
    r = trainleak.check(tr, te)
    assert r.ok, r.text(color=False)
    assert "target column = 'label'" in " ".join(r.notes)


def test_exact_copies_across_split():
    tr, te = clean_split()
    te = pd.concat([te, tr.head(7)])
    r = trainleak.check(tr, te, target="label")
    f = next(x for x in r.findings if x.title.startswith("Test rows that are also"))
    assert f.level == "leak" and f.what.startswith("7 of")


def test_same_template_different_number():
    msgs = [f"Your account will be blocked. Call {rng.integers(10**9, 10**10)} now" for _ in range(5)]
    tr = pd.DataFrame({"text": msgs[:3] + ["hello how are you doing today my friend"] * 1, "label": [1, 1, 1, 0]})
    te = pd.DataFrame({"text": [*msgs[3:], "the meeting moved to friday afternoon please confirm"], "label": [1, 1, 0]})
    r = trainleak.check(tr, te, near=False)
    f = next(x for x in r.findings if x.title == "Same row with only the numbers changed")
    assert f.level == "leak" and f.what.startswith("2 more")


def test_column_that_is_the_answer():
    n = 800
    df = pd.DataFrame({"size": rng.normal(50, 10, n), "rooms": rng.integers(1, 5, n)})
    df["price"] = df["size"] * 3 + rng.normal(scale=40, size=n) + 100
    df["price_per_night_quote"] = df["price"]  # the DataPipeline trap: a copy of the target
    r = trainleak.check(df.iloc[:600], df.iloc[600:], target="price")
    lv = levels(r)
    assert lv.get("Column 'price_per_night_quote' alone predicts 'price'") == "leak"
    assert "Column 'size' alone predicts 'price'" not in lv  # an honest strong feature is not a leak


def test_shared_group():
    tr, te = clean_split()
    tr["patient"] = rng.integers(0, 60, len(tr))
    te["patient"] = rng.integers(50, 100, len(te))
    r = trainleak.check(tr, te, group="patient")
    assert levels(r)["Same group 'patient' in train and test"] == "leak"


def test_time_order():
    tr, te = clean_split()
    tr["date"] = pd.date_range("2024-01-01", periods=len(tr), freq="D")
    te["date"] = pd.date_range("2023-06-01", periods=len(te), freq="D")
    r = trainleak.check(tr, te)
    assert "Test data is not later than training data" in titles(r)


def test_single_dataset_copies():
    tr, _ = clean_split()
    df = pd.concat([tr, tr.head(10)])
    r = trainleak.check(df)
    f = next(x for x in r.findings if x.title == "Copies inside your data")
    assert f.what.startswith("10 rows")


def test_sklearn_style_arrays():
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] > 0).astype(int)
    r = trainleak.check(X[:240], np.vstack([X[240:], X[:5]]), y[:240], np.concatenate([y[240:], y[:5]]))
    assert any(f.title.startswith("Test rows that are also") for f in r.findings)


def test_template_groups():
    df = pd.DataFrame({"t": ["Call 0171 now", "call 0189 NOW!", "hello"]})
    g = trainleak.template_groups(df, ["t"])
    assert g[0] == g[1] != g[2]


# ------------------------------------------------------------------ code checks
def nb(tmp_path, *cells):
    p = tmp_path / "n.ipynb"
    p.write_text(
        json.dumps(
            {
                "cells": [{"cell_type": "code", "source": c} for c in cells],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        )
    )
    return p


def test_smote_before_split(tmp_path):
    p = nb(
        tmp_path,
        "from imblearn.over_sampling import SMOTE\nsm = SMOTE(random_state=42)\nX_res, y_res = sm.fit_resample(X, y)",
        "X_train, X_test, y_train, y_test = train_test_split(X_res, y_res)",
    )
    r = trainleak.check_code(p)
    f = r.findings[0]
    assert f.level == "leak" and f.title == "SMOTE runs before the train/test split" and "cell 1, line 3" in f.what


def test_smote_after_split_is_fine(tmp_path):
    p = nb(
        tmp_path,
        "X_train, X_test, y_train, y_test = train_test_split(X, y)",
        "X_train, y_train = SMOTE().fit_resample(X_train, y_train)",
    )
    assert trainleak.check_code(p).ok and not trainleak.check_code(p).findings


def test_scaler_before_split_and_on_test(tmp_path):
    p = nb(
        tmp_path,
        "sc = StandardScaler()\nX = sc.fit_transform(X)",
        "X_train, X_test = train_test_split(X)",
        "X_test = sc.fit_transform(X_test)",
    )
    lv = levels(trainleak.check_code(p))
    assert lv["StandardScaler learns from all data before the split"] == "risk"
    assert lv["StandardScaler is fitted on test data"] == "leak"


def test_comments_are_ignored(tmp_path):
    p = nb(tmp_path, "# X_res, y_res = SMOTE().fit_resample(X, y)\nX_train, X_test = train_test_split(X)")
    assert not trainleak.check_code(p).findings


# ------------------------------------------------------------------ CLI + report
def test_cli_exit_code_and_report(tmp_path, capsys):
    tr, te = clean_split()
    te = pd.concat([te, tr.head(3)])
    a, b, md = tmp_path / "train.csv", tmp_path / "test.csv", tmp_path / "r.md"
    tr.to_csv(a, index=False)
    te.to_csv(b, index=False)
    assert main([str(a), str(b), "--report", str(md)]) == 1
    out = capsys.readouterr().out
    assert "leak" in out.lower() and "Fix:" in out
    assert md.read_text().startswith("# TrainLeak report")
    tr.to_csv(b, index=False)
    te2 = clean_split()[1]
    te2.to_csv(b, index=False)
    assert main([str(a), str(b)]) == 0


def test_cli_missing_file():
    with pytest.raises(SystemExit):
        main(["nope.csv"])


def test_missing_values_are_not_copies():
    # regression: with pandas 3, one NaN made every row's key missing, so unrelated rows looked like copies
    df = pd.DataFrame(
        {"a": rng.normal(size=200), "b": [np.nan] * 200, "c": [None if i % 2 else "x" for i in range(200)]}
    )
    r = trainleak.check(df)
    assert not any(f.title == "Copies inside your data" for f in r.findings), r.text(color=False)


def test_same_text_different_other_columns():
    tr = pd.DataFrame(
        {
            "review": ["great phone, fast delivery and good box", "bad battery, it died after two days"] * 3,
            "shop": list("abcdef"),
            "label": [1, 0] * 3,
        }
    )
    r = trainleak.check(tr, text=["review"])
    assert any(f.title == "The same 'review' text appears more than once" for f in r.findings)
    te = pd.DataFrame({"review": ["great phone, fast delivery and good box"], "shop": ["zzz"], "label": [1]})
    r = trainleak.check(tr.iloc[:2], te, text=["review"], near=False)
    assert any(f.code == "TL103" for f in r.findings)


def test_entity_columns_get_a_hint_not_a_false_alarm():
    n = 300
    df = pd.DataFrame(
        {
            "review": [
                " ".join(
                    rng.choice(
                        [
                            "good",
                            "bad",
                            "fast",
                            "slow",
                            "box",
                            "phone",
                            "battery",
                            "screen",
                            "price",
                            "seller",
                            "late",
                            "nice",
                        ],
                        8,
                    )
                )
                for _ in range(n)
            ],
            "product_name": rng.choice(
                [f"Phone model {i} with a long descriptive product title" for i in range(20)], n
            ),
            "label": rng.integers(0, 2, n),
        }
    )
    r = trainleak.check(df)
    lv = levels(r)
    assert lv.get("Many rows share the same 'product_name'") == "info"
    assert not any("text appears more than once" in t for t in lv)
    r2 = trainleak.check(df.iloc[:200], df.iloc[200:], near=False)
    assert levels(r2).get("Same 'product_name' values in train and test") == "risk"


def test_friendly_errors(tmp_path, capsys):
    a = tmp_path / "d.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(a, index=False)
    assert main([str(a), "--target", "nope"]) == 2
    assert "not in the data" in capsys.readouterr().err


def test_target_based_steps_before_split_are_leaks(tmp_path):
    p = nb(
        tmp_path,
        "sel = SelectKBest(k=10)\nX = sel.fit_transform(X, y)",
        "X_train, X_test, y_train, y_test = train_test_split(X, y)",
    )
    f = trainleak.check_code(p).findings[0]
    assert f.level == "leak" and "TARGET" in f.why


def test_bangla_marks_are_kept_and_numbers_merge():
    from trainleak.data import _norm_text

    out = _norm_text(
        pd.Series(["বাংলা ভাষা", "বংল ভষ", "Earn 30,000 taka!", "earn 5 taka", "Call 0171-234 now", "call 0189234 NOW!"])
    ).tolist()
    assert out[0] != out[1]  # vowel signs are letters, not punctuation
    assert out[2] == out[3] and out[4] == out[5]


# ------------------------------------------------------------------ release guarantees
def test_codes_are_unique_and_every_finding_has_one(tmp_path):
    from trainleak.report import CODES

    codes = [c for _, c in CODES]
    assert len(codes) == len(set(codes))
    p = nb(
        tmp_path,
        "sc = StandardScaler()\nX = sc.fit_transform(X)\nX_res, y_res = SMOTE().fit_resample(X, y)",
        "X_train, X_test = train_test_split(X)",
        "X_test = sc.fit_transform(X_test)",
    )
    tr, te = clean_split()
    te = pd.concat([te, tr.head(3)])
    for r in (trainleak.check_code(p), trainleak.check(tr, te)):
        assert r.findings and all(f.code != "TL000" for f in r.findings)


def test_ignore_by_code():
    tr, te = clean_split()
    te = pd.concat([te, tr.head(3)])
    assert not trainleak.check(tr, te, ignore=["tl101"]).findings


def test_json_schema():
    tr, te = clean_split()
    j = json.loads(trainleak.check(tr, te).json())
    assert j["schema_version"] == 1 and j["ok"] is True and j["checks"]


def test_markdown_escapes_user_data():
    df = pd.DataFrame({"t": ["<script>alert(1)</script> `x` long text here ok"] * 2, "label": [0, 1]})
    md = trainleak.check(df.iloc[:1], df.iloc[1:], text=["t"], near=False).markdown()
    assert "<script>" not in md and "&lt;script&gt;" in md


def test_never_executes_notebook_code(tmp_path):
    marker = tmp_path / "pwned"
    p = nb(
        tmp_path,
        f"open({str(marker)!r}, 'w').write('x')\nimport os; os.system('touch {marker}')",
        "X_train, X_test = train_test_split(X)",
    )
    trainleak.check_code(p)
    assert not marker.exists()


def test_no_network(monkeypatch):
    import socket

    def boom(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", boom)
    tr, te = clean_split()
    trainleak.check(tr, te)


def test_refuses_pickle(tmp_path):
    p = tmp_path / "x.pkl"
    pd.DataFrame({"a": [1]}).to_pickle(p)
    assert main([str(p)]) == 2


def test_quiet_and_no_color(tmp_path, capsys, monkeypatch):
    tr, te = clean_split()
    te = pd.concat([te, tr.head(3)])
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    tr.to_csv(a, index=False)
    te.to_csv(b, index=False)
    monkeypatch.setenv("NO_COLOR", "1")
    assert main([str(a), str(b), "-q"]) == 1
    out = capsys.readouterr().out.strip()
    assert out.startswith("LEAK:") and "\033[" not in out and len(out.splitlines()) == 1


# ------------------------------------------------------------------ paths not covered above
def test_near_duplicate_text_is_a_risk():
    tr = pd.DataFrame(
        {
            "text": [
                "City Bank ATM suspicious withdrawal detected, protect it now by visiting http",
                "see you tomorrow at the library",
            ],
            "label": [1, 0],
        }
    )
    te = pd.DataFrame(
        {
            "text": [
                "City Bank ATM suspicious withdrawal detected, protect it now by visiting http now",
                "what time is the football match tonight",
            ],
            "label": [1, 0],
        }
    )
    r = trainleak.check(tr, te, text=["text"])
    f = next(x for x in r.findings if x.code == "TL104")
    assert f.level == "risk"
    assert "similarity" in f.examples[0]


def test_html_card_escapes_and_shows_verdict():
    df = pd.DataFrame({"t": ["<b>same long review text here</b>"] * 2, "label": [0, 1]})
    html = trainleak.check(df.iloc[:1], df.iloc[1:], text=["t"], near=False)._repr_html_()
    assert "<b>same" not in html
    assert "background:#0B1220" in html
    assert "leak" in html.lower()


def test_more_code_patterns(tmp_path):
    p = nb(
        tmp_path,
        "X_train, X_test, y_train, y_test = train_test_split(X, y)",
        "X_test, y_test = SMOTE().fit_resample(X_test, y_test)",
        "X2 = MinMaxScaler().fit_transform(X_test)",
        "from imblearn.over_sampling import SMOTE\nX_train, y_train = SMOTE().fit_resample(X_train, y_train)\nprint(cross_val_score(model, X_train, y_train))",
    )
    codes = {f.code for f in trainleak.check_code(p).findings}
    assert {"TL402", "TL403", "TL406"} <= codes


def test_feature_selection_by_correlation_before_split(tmp_path):
    p = nb(tmp_path, "keep = df.corr()['label'].abs() > 0.1", "X_train, X_test = train_test_split(df[keep.index])")
    assert trainleak.check_code(p).findings[0].code == "TL405"


def test_inputs_series_json_and_wrong_suffix(tmp_path):
    s = pd.Series(rng.normal(size=50), name="x")
    assert trainleak.check(s).ok
    j = tmp_path / "d.json"  # tiny data: the target check must skip politely, not crash
    pd.DataFrame({"a": [1, 2, 3], "label": [0, 1, 0]}).to_json(j)
    assert trainleak.check(j).ok
    with pytest.raises(ValueError, match=r"\.ipynb or \.py"):
        trainleak.check_code(j)


def test_cli_code_and_json_output(tmp_path, capsys):
    p = nb(tmp_path, "X_res, y_res = SMOTE().fit_resample(X, y)", "X_train, X_test = train_test_split(X_res)")
    assert main([str(p), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["findings"][0]["code"] == "TL401"


def test_demo_finds_exactly_what_was_planted():
    r = trainleak.demo()
    leaks = sorted(f.code for f in r.findings if f.level == "leak")
    assert leaks == ["TL101", "TL102", "TL201"], r.text(color=False)
    assert {f.code for f in r.findings if f.level != "leak"} <= {
        "TL104"
    }  # template-like SMS are genuinely near-duplicates
    assert next(f for f in r.findings if f.code == "TL102").what.startswith("2 more")


def test_unicode_forms_match():
    from trainleak.data import _norm_text

    composed, decomposed = "আমার নাম \u09df", "আমার নাম \u09af\u09bc"  # য় written two ways
    a, b = _norm_text(pd.Series([composed, decomposed])).tolist()
    assert a == b


def test_flags_and_counts_are_not_entities():
    n = 400
    df = pd.DataFrame(
        {
            "host_id": rng.integers(0, 60, n),
            "superhost": rng.integers(0, 2, n),
            "calculated_host_listings_count": rng.integers(1, 80, n),
            "price": rng.normal(100, 20, n),
        }
    )
    hints = [f.title for f in trainleak.check(df, target="price").findings if f.code == "TL304"]
    assert hints == ["Many rows share the same 'host_id'"], hints
