# Security policy

## What TrainLeak does with your files

- **Runs locally.** No network calls, no telemetry, no uploads (tested: `test_no_network`).
- **Never runs your code.** Notebooks and scripts are read as text; nothing is imported or executed (tested: `test_never_executes_notebook_code`).
- **Refuses formats that can run code when loaded:** `.pkl`, `.pickle`, `.joblib`, `.npy`, `.npz`.
- **Excel** (`.xlsx`) is read only with the `excel` extra, which includes `defusedxml` to block XML attacks.
- **Size limits:** data files over 2 GB and code files over 50 MB are refused with a clear message.
- **Reports** escape your data (HTML and Markdown), so a value like `<script>` in a dataset can't run in a report viewer.
- Reports include a few example rows from your data. Check them before you share a report publicly.

## Supported versions

The latest minor release gets security fixes.

## Reporting a vulnerability

Please **don't** open a public issue. Use GitHub's private reporting: *Security → Report a vulnerability* on this repository.
You'll get a reply within 7 days. Please include the version, how to reproduce it, and the impact.

## Release integrity

Releases are built and published by GitHub Actions with PyPI trusted publishing (no stored tokens); PyPI attestations are
generated for every file. Actions are pinned to commit SHAs and checked with `zizmor` and `actionlint`.
