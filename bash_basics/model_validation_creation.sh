#!/bin/bash
# model_validation_creation.sh - create project folder structure

mkdir -p model_validation/{data,scripts,notebooks,reports} \
&& touch model_validation/data/sample_data.csv \
&& touch model_validation/scripts/{generate_mock_data.py,train_model.py,validate_model.py,utils.py} \
&& touch model_validation/notebooks/model_validation.ipynb \
&& touch model_validation/reports/validation_report.md \
&& touch model_validation/README.md

echo "✅ model_validation project structure created."
