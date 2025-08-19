#!/bin/bash
# Install all VS Code extensions listed in vscode_extensions_install.txt
# without spawning multiple VS Code windows (Hydra mode 🐙)

echo "Installing VS Code extensions..."
while read extension; do
    if [[ ! -z "$extension" && ! "$extension" =~ ^# ]]; then
        echo "→ Installing $extension"
        code --install-extension "$extension" --force
    fi
done < vscode_extensions_install.txt
echo "Done ✅"
