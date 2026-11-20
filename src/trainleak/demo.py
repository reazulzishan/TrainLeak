"""trainleak.demo(): a first result in one line, with no data of your own."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import check
from .report import Report


def demo() -> Report:
    """Builds a small, realistic leaky split in memory (scam-SMS style) and checks it.

    Planted: 1 exact copy (TL101), 2 SMS that differ from a training SMS only by a number (TL102), and a column
    ('reported_by_bank') filled in after the label was known (TL201). The checks find exactly these leaks; the
    template-like SMS also show up as near-duplicates (TL104), which is true of real scam data too.
    """
    rng = np.random.default_rng(7)
    who = ["Dear customer", "Sir", "Hello", "Bhai", "Apu", "Dear user", "Attention", "Hi"]
    scam_do = [
        "your account will be blocked",
        "you won a prize",
        "your SIM will be closed",
        "your parcel is on hold",
        "you got a refund",
        "your card is locked",
    ]
    scam_ask = [
        "send your PIN",
        "share the OTP",
        "click this link",
        "call this agent",
        "pay a small fee",
        "verify your NID",
    ]
    safe_do = [
        "your salary is credited",
        "the meeting moved",
        "your order is shipped",
        "class starts late",
        "the bill is paid",
        "your ticket is booked",
    ]
    safe_end = ["thank you", "see you soon", "have a nice day", "no action needed", "check the app", "regards"]
    rows: list[dict] = []
    seen: set[str] = set()
    while len(rows) < 240:
        is_scam = len(rows) % 2 == 0
        parts = (who, scam_do, scam_ask) if is_scam else (who, safe_do, safe_end)
        t = ", ".join(str(rng.choice(p)) for p in parts)
        if t in seen:
            continue
        seen.add(t)
        rows.append({"text": t, "amount": round(float(rng.gamma(2, 900)), 2), "label": int(is_scam)})
    df = pd.DataFrame(rows)
    df["reported_by_bank"] = df["label"]  # filled in after the fact: a column that IS the answer
    train, test = df.iloc[:200].copy(), df.iloc[200:].copy()
    # two test SMS that are training SMS with a new number added (same template), and one exact copy
    test.iloc[0] = train.iloc[0].to_dict() | {"text": train.iloc[0]["text"] + " 01711-223344"}
    test.iloc[1] = train.iloc[2].to_dict() | {"text": train.iloc[2]["text"] + " Tk 5,000"}
    train.iloc[0, 0] = train.iloc[0, 0] + " 01899-765432"
    train.iloc[2, 0] = train.iloc[2, 0] + " Tk 500"
    test = pd.concat([test, train.iloc[[5]]])
    return check(train, test, target="label", text=["text"])
