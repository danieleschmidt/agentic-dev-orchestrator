#compdef ado agentic-dev-orchestrator
# Zsh completion for ado (Agentic Development Orchestrator)

_ado() {
    local context state line
    typeset -A opt_args

    _arguments -C \
        '1: :_ado_commands' \
        '*::arg:->args'

    case $state in
        args)
            case $line[1] in
                init|run|status|discover|help)
                    # No additional arguments for these commands
                    ;;
            esac
            ;;
    esac
}

_ado_commands() {
    local commands
    commands=(
        'init:Initialize ADO in current directory'
        'run:Execute autonomous backlog processing'
        'status:Show current backlog status'
        'discover:Discover new backlog items from code'
        'help:Show help information'
    )
    _describe 'commands' commands
}

_ado "$@"