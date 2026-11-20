# TrainLeak on real projects (validation, Nov 20, 2026)

TrainLeak was run, with no tuning, on four real CS Funda projects where the leaks were already known from manual analysis.
Times are on a MacBook (Apple Silicon), Python 3.11, pandas 3.1, scikit-learn 1.9.

| Dataset | What we knew (manual) | What TrainLeak reported | Time |
|---|---|---|---|
| Bengali SMS Smishing (Hugging Face, MIT): official train+validation vs test, 5,604 vs 1,401 | 81 test SMS were the same template as a training SMS once numbers are removed | **93** template copies across the split (92 whole rows + 1 same text with a different `source` label) These include 79 of the manual 81, plus 14 the manual check missed (same SMS with different spacing, Bangla vs English digits, or a different number); 48 more near-duplicates (≥ 90 % similar). The other 2 of the 81 are Bangla spelling variants (কী / কি) that the manual method matched only by deleting vowel signs | ~3 s |
| Inside Airbnb NYC listings (CC BY 4.0), 19,205 rows, target `price` | `price_quote_price_per_night` gives R² = 1.000: it is the price | **Leak:** `price_quote_price_per_night` alone → R² 1.000. Also 164 rows that are exact copies apart from their ID | ~2 s |
| Bangladeshi e-commerce reviews (Rashid et al. 2023, Mendeley Data, CC BY 4.0), 78,130 rows | 13,983 duplicate review texts | **13,983** repeated `Review` texts (18,295 once numbers/case/punctuation are ignored). New: **`Emotion` alone predicts `Sentiment` with balanced accuracy 1.000**: one label is derived from the other, so using it as a feature is a leak. Hint: 2,476 products for 78,130 rows → split by product if predicting for new products | ~3 s |
| Our 5 project notebooks (CommerceGuard ×2, DataPipeline, MLModels, ScamGuard) | Written leak-safe | **No findings** (no false alarms). One notebook splits by date with custom code; TrainLeak says it found no split instead of guessing | < 1 s each |
| `examples/smote_mistake.ipynb` vs `examples/smote_fixed.ipynb` | SMOTE + scaler before the split | **Leak:** SMOTE before split (cell 4, line 3); **risk:** StandardScaler before split. Fixed notebook: no findings | < 1 s |

Bugs found by this validation and fixed before release (each has a regression test):
1. With pandas 3, a missing value made every row's fingerprint "missing", so unrelated rows counted as copies (5,513 instead of 164).
2. pandas 3 stores text as the new `str` type, so text columns weren't detected.
3. Python's `\w` treats Bangla vowel signs as punctuation; text normalisation now keeps every letter and mark and only drops punctuation, symbols and spaces.
4. Numbers written with commas or dashes (`50,000`, `0171-234`) became several numbers; they now count as one.
5. A repeating product name was reported as "duplicate text"; long columns now count as free text only if mostly unique, and name-like columns (product, user, patient…) get a split-by-group hint instead.
