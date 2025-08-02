#!/usr/bin/env bash
# Bash completion for ado (Agentic Development Orchestrator)

_ado_complete() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Available commands
    opts="init run status discover help --help -h"

    # Complete based on previous word
    case "${prev}" in
        ado)
            COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
            return 0
            ;;
        *)
            ;;
    esac

    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
    return 0
}

# Register completion function
complete -F _ado_complete ado
complete -F _ado_complete agentic-dev-orchestrator