#!/bin/bash
# project_creation.sh - create generic project skeleton
# Usage: ./project_creation.sh <project_name>

# 1. Check if project name was provided
if [ -z "$1" ]; then
  echo "❌ Error: No project name provided."
  echo "Usage: ./project_creation.sh <project_name>"
  exit 1
fi

# 2. Capture the argument
PROJECT_NAME=$1

# 3. Create the structure
mkdir -p $PROJECT_NAME/{data,scripts,notebooks,reports} \
&& touch $PROJECT_NAME/data/sample_data.csv \
&& touch $PROJECT_NAME/scripts/{generate_mock_data.py,train_model.py,validate_model.py,utils.py} \
&& touch $PROJECT_NAME/notebooks/${PROJECT_NAME}.ipynb \
&& touch $PROJECT_NAME/reports/validation_report.md \
&& touch $PROJECT_NAME/README.md

echo "✅ Project skeleton '$PROJECT_NAME' created."
