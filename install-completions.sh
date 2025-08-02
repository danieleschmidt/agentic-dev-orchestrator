#!/usr/bin/env bash
# Installation script for ADO shell completions

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPLETIONS_DIR="${SCRIPT_DIR}/completions"

install_bash_completion() {
    local completion_dirs=(
        "$HOME/.bash_completion.d"
        "/usr/local/etc/bash_completion.d"
        "/etc/bash_completion.d"
    )
    
    for dir in "${completion_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            echo "Installing bash completion to $dir"
            cp "${COMPLETIONS_DIR}/ado.bash" "$dir/"
            return 0
        fi
    done
    
    echo "Warning: No bash completion directory found. You can manually source:"
    echo "  source ${COMPLETIONS_DIR}/ado.bash"
}

install_zsh_completion() {
    local completion_dirs=(
        "${fpath[1]}"
        "/usr/local/share/zsh/site-functions"
        "/usr/share/zsh/site-functions"
    )
    
    for dir in "${completion_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            echo "Installing zsh completion to $dir"
            cp "${COMPLETIONS_DIR}/ado.zsh" "$dir/_ado"
            return 0
        fi
    done
    
    echo "Warning: No zsh completion directory found. You can manually add to fpath:"
    echo "  fpath=(${COMPLETIONS_DIR} \$fpath)"
    echo "  autoload -U compinit && compinit"
}

install_fish_completion() {
    local completion_dirs=(
        "$HOME/.config/fish/completions"
        "/usr/local/share/fish/completions"
        "/usr/share/fish/completions"
    )
    
    for dir in "${completion_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            echo "Installing fish completion to $dir"
            cp "${COMPLETIONS_DIR}/ado.fish" "$dir/"
            return 0
        fi
    done
    
    echo "Warning: No fish completion directory found. You can manually copy:"
    echo "  cp ${COMPLETIONS_DIR}/ado.fish ~/.config/fish/completions/"
}

main() {
    echo "Installing ADO shell completions..."
    
    # Detect shells and install completions
    if command -v bash >/dev/null 2>&1; then
        install_bash_completion
    fi
    
    if command -v zsh >/dev/null 2>&1; then
        install_zsh_completion
    fi
    
    if command -v fish >/dev/null 2>&1; then
        install_fish_completion
    fi
    
    echo ""
    echo "Installation complete! You may need to restart your shell or source your shell configuration."
    echo ""
    echo "To test completions:"
    echo "  ado <TAB><TAB>"
}

main "$@"