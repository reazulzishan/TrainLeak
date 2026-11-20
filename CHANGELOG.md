# Changelog

All notable changes are listed here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions: [SemVer](https://semver.org/).

## [0.1.0] - unreleased

### Added
- `trainleak.check()` for one dataset or a train/test split (DataFrames, arrays, CSV/TSV/Parquet/JSON/Excel files): exact and
  template copies across the split (numbers, case, punctuation, spacing ignored; Bangla letters and marks kept), repeated texts,
  near-duplicate texts (character TF-IDF), shared groups/IDs/entities, time order, single columns that predict the target.
- `trainleak.check_code()` for `.ipynb`/`.py`: SMOTE and other resamplers before the split or on test data, preprocessing
  fitted before the split or on test data (target-based steps flagged as leaks), feature selection with the target on all
  data, resampling + cross-validation without an imblearn Pipeline. Code is read, never run.
- Stable finding codes (TL101…TL406), `ignore=`, JSON output with `schema_version`, Markdown report, notebook HTML card.
- `trainleak.demo()`: a first result with no data of your own. Colab quickstart notebook.
- Unicode NFKC normalisation so text typed in two forms (e.g. Bangla য়) matches.
- `trainleak` command: exit codes 0/1/2, `--quiet`, `--json`, `--report`, `--ignore`, `--no-color`/`NO_COLOR`.
