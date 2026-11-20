#!/usr/bin/env bash
# Run the test suite across the supported Python / pandas range, the same combos as CI.
set -u
run() {
  local py="$1"; shift
  local args=()
  for r in "$@"; do args+=(--with "$r"); done
  local ver res
  ver=$(uv run -q --isolated --no-project --python "$py" "${args[@]}" --with-editable . -- python -c 'import sys,pandas,numpy,sklearn;print(sys.version.split()[0],"pandas",pandas.__version__,"numpy",numpy.__version__,"sklearn",sklearn.__version__)' 2>&1 | tail -1)
  res=$(uv run -q --isolated --no-project --python "$py" "${args[@]}" --with-editable . --with pytest --with nbformat --with pyarrow -- python -m pytest -q -p no:cacheprovider tests 2>&1 | tail -1)
  echo "$ver  ->  $res"
}
run 3.10 "pandas>=2.0,<2.1" "numpy<2" "scikit-learn>=1.2,<1.3"   # oldest supported
run 3.10 "pandas>=2.2,<3"
run 3.12 "pandas>=2.2,<3"
run 3.12 "pandas>=3"
run 3.14 "pandas>=3"
