#!/usr/bin/env bash
# Assemble the static web app: site/ + the freshly built wheel + example notebooks -> _site/
set -euo pipefail
rm -rf _site dist && uv build --wheel -q
mkdir -p _site/examples
cp site/index.html site/app.js site/theme.js _site/
cp dist/*.whl _site/
basename dist/*.whl > _site/wheel.txt
cp examples/smote_mistake.ipynb examples/smote_fixed.ipynb _site/examples/
touch _site/.nojekyll
echo "built _site with $(cat _site/wheel.txt)"
