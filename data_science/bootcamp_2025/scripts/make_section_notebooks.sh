#!/usr/bin/env bash
set -euo pipefail
# run from repo root (script adjusts if launched from scripts/)
cd "$(dirname "$0")/.."

sections=(01_intro 02_math 03_statistics 04_python_basics 05_tableau 06_advanced_stats 07_ml 08_deep_learning)

for sec in "${sections[@]}"; do
  mkdir -p "notebooks/$sec"
  title="${sec//_/ } — Overview"
  cat > "notebooks/$sec/${sec}__overview.ipynb" <<EOF
{
 "cells": [
  { "cell_type": "markdown", "metadata": {}, "source": ["# $title"] },
  { "cell_type": "markdown", "metadata": {}, "source": ["## Notes\\n","- Add your learning notes here\\n"] },
  { "cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["# First experiments for $sec"] }
 ],
 "metadata": {
  "kernelspec": { "display_name": "Python (bootcamp_2025)", "language": "python", "name": "bootcamp_2025" },
  "language_info": { "name": "python" }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
EOF
done

echo "✅ Notebooks created under notebooks/{01..08}_*/"
