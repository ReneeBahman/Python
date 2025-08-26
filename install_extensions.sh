#!/usr/bin/env bash
echo "Installing VS Code extensions..."

while IFS= read -r ext || [ -n "$ext" ]; do
    # Skip empty lines or comments
    if [[ -z "$ext" || "$ext" =~ ^# ]]; then
        continue
    fi

    echo -n "→ Installing $ext ... "
    if code --install-extension "$ext" --force >/dev/null 2>&1; then
        echo "✅"
    else
        echo "❌ (failed)"
    fi
done < "$(dirname "$0")/vscode_extensions_install.txt"

echo "All done 🎉"
