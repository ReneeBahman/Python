#!/bin/bash
# Reset (nuke + rebuild) the virtual environment for this project

echo "🔥 Nuking old .venv..."
rm -rf .venv

echo "🛠️  Creating new .venv..."
python -m venv .venv

echo "✅ Activating .venv..."
source .venv/Scripts/activate

echo "📦 Installing base packages..."
pip install -r requirements_base.txt

echo "📦 Installing project-specific packages (if available)..."
if [ -f model_validation/requirements.txt ]; then
    pip install -r model_validation/requirements.txt
fi

echo "🎉 All done! Fresh .venv is ready."
